#!/usr/bin/env python3
"""
基于全局性价比排序的混合结构化剪枝

核心思想：
- 将剪枝问题建模为分数背包问题
- Score = Importance / Cost
- 全局排序，优先剪除"性价比"最低的 groups
- 自动实现深度剪枝（层移除）+ 宽度剪枝（神经元剪除）的混合策略
"""

import os
import torch
import argparse
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.methods.global_pruning import (
    build_global_group_table,
    select_groups_to_prune
)
from core.methods.gqa_aware import prune_attention_by_gqa_groups
from core.datasets import DatasetManager
from core.models import IdentityDecoderLayer
from evaluation.metrics.ppl import PPLMetric
from core.trainer.finetuner import FineTuner
from core.utils.logger import LoggerWithDepth


def collect_layer_activations(model, input_ids, device='cuda'):
    """
    收集每层的激活值用于 Wanda 方法

    Returns:
        activations: Dict[layer_idx -> Dict[name -> Tensor]]
    """
    activations = {}
    hooks = []

    def get_activation_hook(layer_idx, name):
        def hook(module, input, output):
            if layer_idx not in activations:
                activations[layer_idx] = {}
            # 存储输入激活值的平均值（用于 Wanda）
            if isinstance(input, tuple):
                act = input[0].detach()
            else:
                act = input.detach()
            # 计算所有维度的平均（除了最后的特征维度）
            if act.dim() > 1:
                act = act.abs().mean(dim=tuple(range(act.dim() - 1)))
            activations[layer_idx][name] = act.cpu()
        return hook

    # 为每层的关键模块注册 hooks
    for layer_idx, layer in enumerate(model.model.layers):
        # Attention 的输入激活
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'q_proj')))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'k_proj')))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'v_proj')))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'o_proj')))

        # MLP 的输入激活
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'mlp_input')))

        # MLP 中间激活（用于 down_proj）
        def get_mlp_intermediate_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx not in activations:
                    activations[layer_idx] = {}
                act = output.detach().abs().mean(dim=tuple(range(output.dim() - 1)))
                activations[layer_idx]['intermediate'] = act.cpu()
            return hook

        hooks.append(layer.mlp.up_proj.register_forward_hook(
            get_mlp_intermediate_hook(layer_idx)))

    # 执行前向传播
    with torch.no_grad():
        model(input_ids)

    # 移除所有 hooks
    for hook in hooks:
        hook.remove()

    return activations


def apply_global_pruning(model, groups_to_prune_df, head_dim=128, gqa_ratio=4, logger=None):
    """
    根据全局分析表执行实际剪枝

    Args:
        model: 模型
        groups_to_prune_df: 要剪枝的 groups DataFrame
        head_dim: attention head 维度
        gqa_ratio: Q:KV 比例
        logger: 日志记录器

    Returns:
        pruned_layers: 被完全剪空的层列表
        pruning_stats: 剪枝统计信息
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    log("\n" + "="*60)
    log("执行全局剪枝")
    log("="*60)

    num_layers = len(model.model.layers)
    pruning_stats = {
        'attention': {},  # {layer_idx: (old_kv, new_kv)}
        'mlp': {},        # {layer_idx: (old_channels, new_channels)}
        'empty_layers': []
    }

    # 按层组织要剪枝的 groups
    layer_prune_info = {}
    for layer_idx in range(num_layers):
        layer_data = groups_to_prune_df[groups_to_prune_df['layer_idx'] == layer_idx]

        attn_groups = layer_data[layer_data['group_type'] == 'attention']['group_idx'].tolist()
        mlp_groups = layer_data[layer_data['group_type'] == 'mlp']['group_idx'].tolist()

        layer_prune_info[layer_idx] = {
            'attention': attn_groups,
            'mlp': mlp_groups
        }

    # 执行剪枝
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        prune_info = layer_prune_info[layer_idx]

        log(f"\n处理 Layer {layer_idx}:")

        # ========== Attention 剪枝 ==========
        attn_prune_indices = prune_info['attention']

        if len(attn_prune_indices) > 0:
            # 获取当前 KV heads 数量（从权重形状推断）
            k_proj_out_features = layer.self_attn.k_proj.out_features
            num_kv_heads = k_proj_out_features // head_dim

            # 计算保留的 indices
            all_kv_indices = set(range(num_kv_heads))
            keep_kv_indices = sorted(list(all_kv_indices - set(attn_prune_indices)))

            # 从权重形状获取 Q heads 数量
            q_proj_out_features = layer.self_attn.q_proj.out_features
            old_q = q_proj_out_features // head_dim
            old_kv = num_kv_heads

            if len(keep_kv_indices) > 0:
                # 执行剪枝
                new_q, new_kv = prune_attention_by_gqa_groups(
                    layer,
                    keep_kv_indices,
                    head_dim=head_dim,
                    gqa_ratio=gqa_ratio
                )
                log(f"  Attention: {old_q}Q:{old_kv}KV → {new_q}Q:{new_kv}KV")
                pruning_stats['attention'][layer_idx] = (old_kv, new_kv)
            else:
                # 该层 Attention 被完全剪空
                log(f"  ⚠️ Attention 被完全剪空（{old_kv} → 0 KV heads）")
                pruning_stats['attention'][layer_idx] = (old_kv, 0)

        # ========== MLP 剪枝 ==========
        mlp_prune_indices = prune_info['mlp']

        if len(mlp_prune_indices) > 0:
            intermediate_size = layer.mlp.gate_proj.out_features

            # 计算保留的 indices
            all_mlp_indices = set(range(intermediate_size))
            keep_mlp_indices = sorted(list(all_mlp_indices - set(mlp_prune_indices)))

            if len(keep_mlp_indices) > 0:
                # 执行 MLP 剪枝
                keep_mlp_indices_tensor = torch.tensor(keep_mlp_indices, device=layer.mlp.gate_proj.weight.device)

                # 剪枝 gate_proj 和 up_proj（保留对应的行）
                layer.mlp.gate_proj.weight = torch.nn.Parameter(
                    layer.mlp.gate_proj.weight[keep_mlp_indices_tensor, :]
                )
                layer.mlp.up_proj.weight = torch.nn.Parameter(
                    layer.mlp.up_proj.weight[keep_mlp_indices_tensor, :]
                )

                # 剪枝 down_proj（保留对应的列）
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    layer.mlp.down_proj.weight[:, keep_mlp_indices_tensor]
                )

                # 更新 intermediate_size
                new_intermediate_size = len(keep_mlp_indices)
                layer.mlp.gate_proj.out_features = new_intermediate_size
                layer.mlp.up_proj.out_features = new_intermediate_size
                layer.mlp.down_proj.in_features = new_intermediate_size

                log(f"  MLP: {intermediate_size} → {new_intermediate_size} channels")
                pruning_stats['mlp'][layer_idx] = (intermediate_size, new_intermediate_size)
            else:
                # 该层 MLP 被完全剪空
                log(f"  ⚠️ MLP 被完全剪空（{intermediate_size} → 0 channels）")
                pruning_stats['mlp'][layer_idx] = (intermediate_size, 0)

        # 检查是否整层被剪空
        attn_empty = (layer_idx in pruning_stats['attention'] and
                     pruning_stats['attention'][layer_idx][1] == 0)
        mlp_empty = (layer_idx in pruning_stats['mlp'] and
                    pruning_stats['mlp'][layer_idx][1] == 0)

        if attn_empty and mlp_empty:
            log(f"  🔴 Layer {layer_idx} 被完全剪空（自动深度剪枝）")
            pruning_stats['empty_layers'].append(layer_idx)

    return pruning_stats


def remove_empty_layers(model, empty_layers, logger=None):
    """
    "移除"被完全剪空的层 - 通过替换为 identity 层

    注意：由于 HuggingFace Transformers 的内部实现可能在多处假设层数固定，
    完全删除层可能导致 "list index out of range" 错误。
    因此我们采用更安全的策略：将空层替换为简单的 pass-through 层。

    Args:
        model: 模型
        empty_layers: 要移除的层索引列表
        logger: 日志记录器
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    if len(empty_layers) == 0:
        log("\n✓ 没有层被完全剪空，跳过层移除")
        return

    log(f"\n{'='*60}")
    log(f"移除完全剪空的层")
    log(f"{'='*60}")
    log(f"要移除的层: {empty_layers}")
    log(f"策略: 替换为 Identity 层（保持模型结构完整）")

    # 为了避免 HuggingFace Transformers 内部的各种假设被打破
    # 我们不删除层，而是将它们替换为全局定义的 IdentityDecoderLayer
    num_layers = len(model.model.layers)

    # 替换空层为 identity 层
    for layer_idx in empty_layers:
        if layer_idx < num_layers:
            log(f"  替换 Layer {layer_idx} 为 Identity 层")
            model.model.layers[layer_idx] = IdentityDecoderLayer()

    log(f"✓ 已替换 {len(empty_layers)} 层为 Identity 层")
    log(f"  物理层数: {num_layers} (保持不变)")
    log(f"  有效层数: {num_layers - len(empty_layers)}")

    # 确保模型在正确的设备上并处于eval模式
    device = next(model.parameters()).device
    model.eval()

    log(f"✓ 模型状态已刷新")

    # 验证模型是否可以正常forward（使用一个小的dummy输入）
    try:
        with torch.no_grad():
            dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
            _ = model(dummy_input)
        log(f"✓ 模型forward验证通过")
    except Exception as e:
        log(f"⚠️  模型forward验证失败: {e}")
        log(f"   这可能会导致后续PPL计算出错")
        import traceback
        log(f"   错误详情:\n{traceback.format_exc()}")


def auto_collapse(model, pruning_stats, collapse_threshold=0.15, logger=None):
    """
    自动坍缩：检测稀疏层并强制移除整层

    H-GSP 核心思想：避免"留 10% 不如不留"的情况
    当某层的参数保留率低于阈值时，直接移除整层

    Args:
        model: 模型
        pruning_stats: 剪枝统计信息
            {'attention': {layer_idx: (old, new)}, 'mlp': {layer_idx: (old, new)}, 'empty_layers': []}
        collapse_threshold: 坍缩阈值（默认 0.15 = 15%）
        logger: 日志记录器

    Returns:
        additional_empty_layers: 需要额外移除的层列表
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    log(f"\n{'='*60}")
    log(f"自动坍缩检测 (Auto-Collapse)")
    log(f"{'='*60}")
    log(f"坍缩阈值: {collapse_threshold:.1%}")
    log(f"检测逻辑: 当层参数保留率 < {collapse_threshold:.1%} 时，强制移除整层")

    num_layers = len(model.model.layers)
    additional_empty_layers = []

    for layer_idx in range(num_layers):
        # 跳过已经被剪空的层
        if layer_idx in pruning_stats.get('empty_layers', []):
            continue

        # 计算该层的参数保留率
        attn_info = pruning_stats['attention'].get(layer_idx)
        mlp_info = pruning_stats['mlp'].get(layer_idx)

        # 计算 Attention 保留率
        if attn_info:
            old_kv, new_kv = attn_info
            attn_retain_rate = new_kv / old_kv if old_kv > 0 else 1.0
        else:
            attn_retain_rate = 1.0

        # 计算 MLP 保留率
        if mlp_info:
            old_channels, new_channels = mlp_info
            mlp_retain_rate = new_channels / old_channels if old_channels > 0 else 1.0
        else:
            mlp_retain_rate = 1.0

        # 计算综合保留率（取两者的平均）
        avg_retain_rate = (attn_retain_rate + mlp_retain_rate) / 2.0

        # 判断是否触发坍缩
        if avg_retain_rate < collapse_threshold:
            log(f"  🔻 Layer {layer_idx} 触发坍缩:")
            log(f"     Attn 保留率: {attn_retain_rate:.1%}, MLP 保留率: {mlp_retain_rate:.1%}")
            log(f"     平均保留率: {avg_retain_rate:.1%} < {collapse_threshold:.1%}")
            log(f"     决策: 强制移除整层")
            additional_empty_layers.append(layer_idx)

    if len(additional_empty_layers) == 0:
        log(f"\n✓ 没有层触发坍缩阈值")
    else:
        log(f"\n✓ 检测到 {len(additional_empty_layers)} 层需要坍缩: {additional_empty_layers}")
        log(f"  这些层将被强制移除（利用残差悖论）")

    return additional_empty_layers


def main():
    parser = argparse.ArgumentParser(description='基于全局性价比的混合结构化剪枝')

    # 模型参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--save_ckpt_log_name', type=str, default='llama_global_prune',
                       help='实验名称')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率（相对于模型总参数）')
    parser.add_argument('--importance_method', type=str, default='taylor',
                       choices=['taylor', 'wanda', 'taylor_2nd'],
                       help='重要性计算方法: taylor(一阶), wanda(权重×激活), taylor_2nd(二阶)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'ptb', 'c4'],
                       help='数据集选择（用于所有重要性计算和评估）')
    parser.add_argument('--gradient_batch_size', type=int, default=4,
                       help='梯度计算时的批次大小（用于节省内存）')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                       help='使用梯度检查点（节省显存但会慢一些）')
    parser.add_argument('--remove_empty_layers', action='store_true',
                       help='是否移除被完全剪空的层（自动深度剪枝）')

    # H-GSP 核心参数
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='H-GSP 温度参数 T：控制敏感度加权强度 (T=0: 纯Taylor, T=1: 推荐平衡, T>1: 激进强化首尾)')
    parser.add_argument('--tau', type=float, default=None,
                       help='H-GSP 门控阈值 τ：Layer/Block 模式切换点 (None: 自动计算25分位数, 0: 纯Block, inf: 纯Layer)')
    parser.add_argument('--epsilon', type=float, default=0.15,
                       help='H-GSP 坍缩阈值 ε：层剩余参数率低于此值时自动坍缩整层（默认0.15）')

    # GQA 配置
    parser.add_argument('--head_dim', type=int, default=128,
                       help='Attention head 维度')
    parser.add_argument('--gqa_ratio', type=int, default=4,
                       help='Q:KV 比例')

    # 评估参数
    parser.add_argument('--test_before_prune', action='store_true',
                       help='剪枝前评估基线 PPL')
    parser.add_argument('--test_after_prune', action='store_true',
                       help='剪枝后评估 PPL')

    # 微调参数
    parser.add_argument('--finetune', action='store_true',
                       help='剪枝后进行微调')
    parser.add_argument('--finetune_method', type=str, default='lora',
                       choices=['full', 'lora'],
                       help='微调方法')
    parser.add_argument('--finetune_samples', type=int, default=500,
                       help='微调样本数')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                       help='微调学习率')
    parser.add_argument('--finetune_epochs', type=int, default=1,
                       help='微调轮数')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')

    # 保存参数
    parser.add_argument('--save_model', action='store_true',
                       help='保存剪枝后的模型')

    # 其他
    from core.utils.get_best_gpu import get_best_gpu
    bestDevice = "cuda:"+str(get_best_gpu())  # 自动选择显存最大的GPU
    # bestDevice = "cpu"  # 如果要用CPU，取消注释这行
    parser.add_argument('--device', type=str, default=bestDevice,
                       help='设备')
    parser.add_argument('--layer_start', type=int, default=0,
                       help='起始层（debug用）')
    parser.add_argument('--layer_end', type=int, default=None,
                       help='结束层（debug用）')

    args = parser.parse_args()

    # 设置 logger
    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir='prune_log'
    )

    logger.log("="*60)
    logger.log("基于全局性价比的混合结构化剪枝")
    logger.log("="*60)
    logger.log(f"模型: {args.base_model}")
    logger.log(f"剪枝率: {args.pruning_ratio:.1%}")
    logger.log(f"重要性方法: {args.importance_method}")
    logger.log(f"数据集: {args.dataset}")

    # ========== Step 1: 加载模型 ==========
    logger.log("\n[Step 1] 加载模型...")

    # 根据设备选择加载方式
    if 'cpu' in args.device.lower():
        device_map = args.device
    else:
        # 单GPU：直接指定设备，避免多GPU分布
        device_map = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 启用梯度检查点（节省显存）
    if args.use_gradient_checkpointing:
        logger.log("  启用梯度检查点（Gradient Checkpointing）...")
        model.gradient_checkpointing_enable()

    # 获取实际使用的设备
    if hasattr(model, 'hf_device_map'):
        logger.log(f"  模型分布: {model.hf_device_map}")
        # 获取第一个模块的设备（输入数据应该发送到这里）
        first_device = next(iter(model.hf_device_map.values()))
        args.device = f'cuda:{first_device}' if isinstance(first_device, int) else first_device
        logger.log(f"  输入设备: {args.device}")
    else:
        args.device = next(model.parameters()).device

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"✓ 模型加载完成")
    logger.log(f"  总参数量: {total_params:,}")

    # 显示GPU显存使用情况
    if torch.cuda.is_available() and 'cuda' in str(args.device).lower():
        device_str = str(args.device)
        gpu_id = int(device_str.split(':')[-1]) if ':' in device_str else 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.log(f"  GPU 显存: {allocated:.2f}GB / {total_mem:.2f}GB (已分配)")
        logger.log(f"  GPU 显存: {reserved:.2f}GB / {total_mem:.2f}GB (已预留)")

    # 创建数据集管理器（统一管理所有数据集加载）
    logger.log(f"\n✓ 初始化数据集管理器: {args.dataset}")
    dataset_manager = DatasetManager(dataset_name=args.dataset, tokenizer=tokenizer)

    # ========== Step 2: 评估基线 ==========
    if args.test_before_prune:
        logger.log("\n[Step 2] 评估基线 PPL...")
        baseline_ppl = PPLMetric(model, tokenizer, datasets=[args.dataset], device=args.device)
        logger.log(f"✓ 基线 PPL: {baseline_ppl}")

    # ========== Step 3: 计算重要性（梯度或激活） ==========
    activations = None
    hessian_diag = None

    # H-GSP 内部固定参数（不对外暴露）
    TAYLOR_NUM_SAMPLES = 128
    TAYLOR_SEQ_LEN = 128
    LAYER_IMPORTANCE_NUM_SAMPLES = 50
    LAYER_IMPORTANCE_SEQ_LEN = 128
    BLOCK_IMPORTANCE_NUM_SAMPLES = 50
    BLOCK_IMPORTANCE_SEQ_LEN = 128

    if args.importance_method in ['taylor', 'taylor_2nd']:
        logger.log(f"\n[Step 3] 计算梯度（{'一阶' if args.importance_method == 'taylor' else '二阶'} Taylor importance）...")
        logger.log(f"  样本数: {TAYLOR_NUM_SAMPLES}, 序列长度: {TAYLOR_SEQ_LEN} (内部固定)")

        # 分批计算梯度以节省内存
        batch_size = args.gradient_batch_size
        num_batches = (TAYLOR_NUM_SAMPLES + batch_size - 1) // batch_size
        logger.log(f"  批次大小: {batch_size}, 总批次数: {num_batches}")

        model.zero_grad()
        total_loss = 0.0
        start_time = time.time()

        # 如果在 CPU 上运行，给出提示
        if 'cpu' in str(args.device).lower():
            logger.log(f"  ⚠️ 在 CPU 上运行，速度会非常慢！")
            logger.log(f"  预计每个批次需要 5-10 分钟（取决于 CPU 性能）")
            logger.log(f"  总预计时间: {num_batches * 7:.0f} 分钟左右")
            logger.log("")

        # 二阶泰勒需要累积 Hessian 对角线近似
        # ⚠️ 存储在CPU上以避免GPU OOM
        if args.importance_method == 'taylor_2nd':
            hessian_diag = {}
            logger.log("  初始化 Hessian 对角线存储（在CPU上以节省GPU显存）...")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 存储在CPU上，避免占用GPU显存
                    hessian_diag[name] = torch.zeros_like(param.data, device='cpu')

        # 使用 tqdm 显示进度条
        pbar = tqdm(range(num_batches), desc="计算梯度", ncols=100)

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, TAYLOR_NUM_SAMPLES)
            current_batch_size = end_idx - start_idx

            batch_start_time = time.time()

            # 加载当前批次
            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 加载数据...")
            input_ids = dataset_manager.get_gradient_samples(
                num_samples=current_batch_size,
                seq_len=TAYLOR_SEQ_LEN
            )
            input_ids = input_ids.to(args.device)

            # 前向传播
            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 前向传播...")
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss / num_batches  # 归一化

            # 反向传播
            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 反向传播...")
            loss.backward()

            # 二阶泰勒：累积 Hessian 对角线（使用梯度平方近似）
            if args.importance_method == 'taylor_2nd':
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # 将梯度平方移动到CPU后累加，避免GPU OOM
                        hessian_diag[name] += (param.grad ** 2).cpu() / num_batches

            batch_time = time.time() - batch_start_time
            total_loss += loss.item() * num_batches

            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 完成！耗时: {batch_time:.2f}s, loss: {loss.item() * num_batches:.4f}")

            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item() * num_batches:.4f}',
                'batch_time': f'{batch_time:.2f}s'
            })

            # 清理内存
            del input_ids, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        total_time = time.time() - start_time
        logger.log(f"✓ 梯度计算完成")
        logger.log(f"  平均 loss: {total_loss:.4f}")
        logger.log(f"  总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.log(f"  平均每批次: {total_time/num_batches:.2f}s")

        if args.importance_method == 'taylor_2nd':
            logger.log(f"  ✓ Hessian 对角线近似计算完成")

    elif args.importance_method == 'wanda':
        logger.log(f"\n[Step 3] 收集激活值（Wanda importance）...")
        logger.log(f"  样本数: {TAYLOR_NUM_SAMPLES}, 序列长度: {TAYLOR_SEQ_LEN} (内部固定)")

        # 分批收集激活
        batch_size = args.gradient_batch_size
        num_batches = (TAYLOR_NUM_SAMPLES + batch_size - 1) // batch_size
        logger.log(f"  批次大小: {batch_size}, 总批次数: {num_batches}")

        all_activations = {}
        start_time = time.time()

        pbar = tqdm(range(num_batches), desc="收集激活", ncols=100)

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, TAYLOR_NUM_SAMPLES)
            current_batch_size = end_idx - start_idx

            batch_start_time = time.time()

            # 加载当前批次
            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 加载数据...")
            input_ids = dataset_manager.get_gradient_samples(
                num_samples=current_batch_size,
                seq_len=TAYLOR_SEQ_LEN
            )
            input_ids = input_ids.to(args.device)

            # 收集激活
            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 收集激活...")
            batch_activations = collect_layer_activations(model, input_ids, args.device)

            # 累加激活值
            for layer_idx, layer_acts in batch_activations.items():
                if layer_idx not in all_activations:
                    all_activations[layer_idx] = {}
                for name, act in layer_acts.items():
                    if name not in all_activations[layer_idx]:
                        all_activations[layer_idx][name] = act.to(args.device)
                    else:
                        all_activations[layer_idx][name] += act.to(args.device)

            batch_time = time.time() - batch_start_time
            logger.log(f"  [批次 {batch_idx + 1}/{num_batches}] 完成！耗时: {batch_time:.2f}s")

            pbar.set_postfix({'batch_time': f'{batch_time:.2f}s'})

            del input_ids, batch_activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        # 平均激活值
        for layer_idx in all_activations:
            for name in all_activations[layer_idx]:
                all_activations[layer_idx][name] /= num_batches

        activations = all_activations

        total_time = time.time() - start_time
        logger.log(f"✓ 激活值收集完成")
        logger.log(f"  总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.log(f"  平均每批次: {total_time/num_batches:.2f}s")

    # ========== Step 3.5: 计算层移除困惑度（H-GSP 必需）==========
    logger.log(f"\n[Step 3.5] 计算层移除困惑度（H-GSP Layer-wise 重要性）...")
    logger.log(f"  样本数: {LAYER_IMPORTANCE_NUM_SAMPLES}, 序列长度: {LAYER_IMPORTANCE_SEQ_LEN} (内部固定)")

    from core.importance.layer_analyzer import LayerImportanceAnalyzer

    # 加载用于层重要性分析的样本（文本格式）
    layer_texts_list = dataset_manager.get_layer_importance_samples(
        num_samples=LAYER_IMPORTANCE_NUM_SAMPLES,
        seq_len=LAYER_IMPORTANCE_SEQ_LEN
    )

    # 创建分析器
    analyzer = LayerImportanceAnalyzer(model, tokenizer, device=args.device)

    # 计算每层的移除困惑度
    num_layers = len(model.model.layers)
    layer_removal_ppl = analyzer.measure_layer_importance_by_removal(
        texts=layer_texts_list,
        num_layers=num_layers
    )

    logger.log(f"✓ 层移除困惑度计算完成")
    logger.log(f"  示例 - Layer 0: Removal PPL = {layer_removal_ppl[0]:.4f}")
    logger.log(f"  示例 - Layer {num_layers//2}: Removal PPL = {layer_removal_ppl[num_layers//2]:.4f}")
    logger.log(f"  示例 - Layer {num_layers-1}: Removal PPL = {layer_removal_ppl[num_layers-1]:.4f}")

    # 保存层移除困惑度到文件
    import json
    if not hasattr(logger, 'env_name'):
        logger.env_name = 'global_results'
    if not os.path.exists(logger.env_name):
        os.makedirs(logger.env_name, exist_ok=True)

    layer_ppl_path = os.path.join(logger.env_name, 'layer_removal_ppl.json')
    with open(layer_ppl_path, 'w') as f:
        json.dump(layer_removal_ppl, f, indent=2)
    logger.log(f"✓ 层移除困惑度已保存: {layer_ppl_path}")

    # ========== Step 3.6: 计算块移除困惑度（H-GSP 必需）==========
    logger.log(f"\n[Step 3.6] 计算块移除困惑度（H-GSP Block-wise 重要性）...")
    logger.log(f"  样本数: {BLOCK_IMPORTANCE_NUM_SAMPLES}, 序列长度: {BLOCK_IMPORTANCE_SEQ_LEN} (内部固定)")

    # 加载用于块重要性分析的样本（文本格式）
    block_texts_list = dataset_manager.get_layer_importance_samples(
        num_samples=BLOCK_IMPORTANCE_NUM_SAMPLES,
        seq_len=BLOCK_IMPORTANCE_SEQ_LEN
    )

    # 计算每层的 Attention 和 MLP 块移除困惑度
    block_removal_ppl = analyzer.measure_block_importance_by_removal(
        texts=block_texts_list,
        num_layers=num_layers
    )

    logger.log(f"✓ 块移除困惑度计算完成")
    logger.log(f"  示例 - Layer 0 Attention: {block_removal_ppl['attention'][0]:.4f}, MLP: {block_removal_ppl['mlp'][0]:.4f}")
    logger.log(f"  示例 - Layer {num_layers-1} Attention: {block_removal_ppl['attention'][num_layers-1]:.4f}, MLP: {block_removal_ppl['mlp'][num_layers-1]:.4f}")

    # 保存块移除困惑度到文件
    block_ppl_path = os.path.join(logger.env_name, 'block_removal_ppl.json')
    with open(block_ppl_path, 'w') as f:
        json.dump(block_removal_ppl, f, indent=2)
    logger.log(f"✓ 块移除困惑度已保存: {block_ppl_path}")

    # ========== Step 4: 构建全局分析表 ==========
    logger.log("\n[Step 4] 构建全局 Group 分析表...")

    layer_end = args.layer_end if args.layer_end else len(model.model.layers)

    # 传递重要性信息
    importance_info = {}
    if args.importance_method in ['taylor', 'taylor_2nd']:
        importance_info['gradients'] = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        if args.importance_method == 'taylor_2nd':
            importance_info['hessian_diag'] = hessian_diag
    elif args.importance_method == 'wanda':
        importance_info['activations'] = activations

    df = build_global_group_table(
        model=model,
        importance_method=args.importance_method,
        importance_info=importance_info,
        layer_start=args.layer_start,
        layer_end=layer_end,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
        device=args.device,
        layer_removal_ppl=layer_removal_ppl,    # H-GSP: 层级重要性
        block_removal_ppl=block_removal_ppl,    # H-GSP: 块级重要性
        temperature=args.temperature,           # H-GSP: 温度参数 T
        tau=args.tau                           # H-GSP: 门控阈值 τ
    )

    logger.log(f"✓ 分析表构建完成")

    # ========== Step 5: 选择要剪枝的 groups ==========
    logger.log(f"\n[Step 5] 根据剪枝率选择要剪枝的 groups...")

    groups_to_prune = select_groups_to_prune(
        df=df,
        pruning_ratio=args.pruning_ratio,
        total_params=total_params
    )

    logger.log(f"✓ 选中 {len(groups_to_prune)} 个 groups 进行剪枝")

    # 确保输出目录存在
    output_dir = 'global_results'
    if not hasattr(logger, 'env_name'):
        logger.env_name = output_dir
    if not os.path.exists(logger.env_name):
        os.makedirs(logger.env_name, exist_ok=True)

    # 保存分析表（按score排序）
    table_path = os.path.join(logger.env_name, 'global_group_table.csv')
    df.to_csv(table_path, index=False)
    logger.log(f"✓ 分析表已保存（按score排序）: {table_path}")

    prune_table_path = os.path.join(logger.env_name, 'groups_to_prune.csv')
    groups_to_prune.to_csv(prune_table_path, index=False)
    logger.log(f"✓ 剪枝列表已保存（按score排序）: {prune_table_path}")

    # 保存按层排序的分析表
    df_by_layer = df.sort_values(['layer_idx', 'group_type', 'group_idx']).reset_index(drop=True)
    table_by_layer_path = os.path.join(logger.env_name, 'global_group_table_by_layer.csv')
    df_by_layer.to_csv(table_by_layer_path, index=False)
    logger.log(f"✓ 分析表已保存（按层排序）: {table_by_layer_path}")

    # 保存按层排序的剪枝列表
    prune_by_layer = groups_to_prune.sort_values(['layer_idx', 'group_type', 'group_idx']).reset_index(drop=True)
    prune_by_layer_path = os.path.join(logger.env_name, 'groups_to_prune_by_layer.csv')
    prune_by_layer.to_csv(prune_by_layer_path, index=False)
    logger.log(f"✓ 剪枝列表已保存（按层排序）: {prune_by_layer_path}")

    # 生成层级统计摘要
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("各层剪枝统计摘要")
    summary_lines.append("="*80)
    summary_lines.append(f"{'Layer':<8} {'Attention剪枝':<20} {'MLP剪枝':<20} {'总参数剪枝':<20}")
    summary_lines.append("-"*80)

    for layer_idx in sorted(groups_to_prune['layer_idx'].unique()):
        layer_data = groups_to_prune[groups_to_prune['layer_idx'] == layer_idx]
        attn_data = layer_data[layer_data['group_type'] == 'attention']
        mlp_data = layer_data[layer_data['group_type'] == 'mlp']

        attn_count = len(attn_data)
        mlp_count = len(mlp_data)
        attn_params = attn_data['cost'].sum() if len(attn_data) > 0 else 0
        mlp_params = mlp_data['cost'].sum() if len(mlp_data) > 0 else 0
        total_params_prune = attn_params + mlp_params

        summary_lines.append(
            f"{layer_idx:<8} "
            f"{attn_count} groups ({attn_params:,} params)".ljust(20) + " "
            f"{mlp_count} channels ({mlp_params:,} params)".ljust(20) + " "
            f"{total_params_prune:,} params".ljust(20)
        )

    summary_lines.append("-"*80)
    summary_lines.append(f"总计: {len(groups_to_prune)} groups, "
                        f"{groups_to_prune['cost'].sum():,} params")
    summary_lines.append("="*80)

    # 保存摘要文件
    summary_path = os.path.join(logger.env_name, 'pruning_summary_by_layer.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    logger.log(f"✓ 层级统计摘要已保存: {summary_path}")

    # 也在日志中显示
    logger.log("\n" + '\n'.join(summary_lines))

    # ========== Step 6: 执行全局剪枝 ==========
    logger.log(f"\n[Step 6] 执行全局剪枝...")

    pruning_stats = apply_global_pruning(
        model=model,
        groups_to_prune_df=groups_to_prune,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
        logger=logger
    )

    logger.log("\n✓ 全局剪枝完成")

    # ========== Step 6.5: 自动坍缩（H-GSP 必需）==========
    logger.log(f"\n[Step 6.5] 自动坍缩检测（H-GSP Auto-Collapse, ε={args.epsilon}）...")
    additional_empty_layers = auto_collapse(
        model=model,
        pruning_stats=pruning_stats,
        collapse_threshold=args.epsilon,
        logger=logger
    )
    # 将额外的空层加入到 empty_layers 列表
    pruning_stats['empty_layers'].extend(additional_empty_layers)

    # ========== Step 7: 移除空层（可选）==========
    all_empty_layers = pruning_stats['empty_layers']
    if args.remove_empty_layers and len(all_empty_layers) > 0:
        logger.log(f"\n[Step 7] 移除空层...")
        logger.log(f"  原始空层: {len(all_empty_layers) - len(additional_empty_layers)}")
        if len(additional_empty_layers) > 0:
            logger.log(f"  坍缩触发: {len(additional_empty_layers)}")
        logger.log(f"  总计移除: {len(all_empty_layers)} 层")
        remove_empty_layers(model, all_empty_layers, logger)

    # ========== Step 8: 统计剪枝结果 ==========
    logger.log(f"\n{'='*60}")
    logger.log(f"剪枝统计")
    logger.log(f"{'='*60}")

    after_params = sum(p.numel() for p in model.parameters())
    actual_ratio = (total_params - after_params) / total_params

    logger.log(f"参数统计:")
    logger.log(f"  剪枝前: {total_params:,}")
    logger.log(f"  剪枝后: {after_params:,}")
    logger.log(f"  实际剪枝率: {actual_ratio:.2%}")

    if len(pruning_stats['empty_layers']) > 0:
        logger.log(f"\n自动深度剪枝:")
        logger.log(f"  替换为Identity的层: {pruning_stats['empty_layers']}")
        logger.log(f"  物理层数: {len(model.model.layers)} (保持不变)")
        logger.log(f"  有效层数: {len(model.model.layers) - len(pruning_stats['empty_layers'])}")

    # ========== Step 9: 评估剪枝后 PPL ==========
    if args.test_after_prune:
        logger.log(f"\n[Step 9] 评估剪枝后 PPL...")
        pruned_ppl = PPLMetric(model, tokenizer, datasets=[args.dataset], device=args.device)
        logger.log(f"✓ 剪枝后 PPL: {pruned_ppl}")

        if args.test_before_prune and len(pruned_ppl.results) > 0 and len(baseline_ppl.results) > 0:
            # 获取对应数据集的 PPL key（两者应该一致）
            pruned_key = list(pruned_ppl.results.keys())[0]
            baseline_key = list(baseline_ppl.results.keys())[0]

            # 确保都不是inf
            if pruned_ppl[pruned_key] != float('inf') and baseline_ppl[baseline_key] != float('inf'):
                degradation = (pruned_ppl[pruned_key] / baseline_ppl[baseline_key] - 1) * 100
                logger.log(f"  PPL 退化: {degradation:.2f}%")
            else:
                logger.log(f"  ⚠️  无法计算PPL退化（存在inf值）")

    # ========== Step 10: 微调恢复（可选）==========
    if args.finetune:
        logger.log(f"\n[Step 10] 微调恢复...")

        finetuner = FineTuner(model, tokenizer, device=args.device, logger=logger)

        finetuner.finetune(
            dataset_name=args.dataset,
            num_samples=args.finetune_samples,
            lr=args.finetune_lr,
            epochs=args.finetune_epochs,
            method=args.finetune_method,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha
        )

        logger.log(f"✓ 微调完成")

        # 评估微调后 PPL
        if args.test_after_prune:
            logger.log(f"\n评估微调后 PPL...")
            finetuned_ppl = PPLMetric(model, tokenizer, datasets=[args.dataset], device=args.device)
            logger.log(f"✓ 微调后 PPL: {finetuned_ppl}")

            if args.test_before_prune and len(finetuned_ppl.results) > 0 and len(baseline_ppl.results) > 0:
                finetuned_key = list(finetuned_ppl.results.keys())[0]
                baseline_key = list(baseline_ppl.results.keys())[0]

                if finetuned_ppl[finetuned_key] != float('inf') and baseline_ppl[baseline_key] != float('inf'):
                    final_degradation = (finetuned_ppl[finetuned_key] / baseline_ppl[baseline_key] - 1) * 100
                    logger.log(f"  最终 PPL 退化: {final_degradation:.2f}%")
                else:
                    logger.log(f"  ⚠️  无法计算最终PPL退化（存在inf值）")

    # ========== Step 11: 保存模型 ==========
    if args.save_model:
        logger.log(f"\n[Step 11] 保存模型...")

        save_path = os.path.join(logger.env_name, 'pytorch_model.bin')

        save_dict = {
            'model': model,
            'tokenizer': tokenizer,
            'pruning_stats': pruning_stats,
            'pruning_ratio': args.pruning_ratio,
            'actual_ratio': actual_ratio,
            'method': 'global_pruning',
            'config': args.__dict__
        }

        torch.save(save_dict, save_path)
        logger.log(f"✓ 模型已保存: {save_path}")

    logger.log(f"\n{'='*60}")
    logger.log(f"✓ 全部完成！")
    logger.log(f"{'='*60}")


if __name__ == '__main__':
    main()
