#!/usr/bin/env python3
"""
Llama-3 非均衡结构化剪枝脚本 (GQA-Aware)

核心改进：
1. 保留层重要性评估和per-layer剪枝率计算
2. Attention使用GQA-aware Taylor importance剪枝
3. MLP也使用Taylor importance剪枝（综合gate/up/down三个投影）
4. 不依赖torch_pruning，完全手动控制剪枝过程
5. 确保4:1 GQA比例自然保持，基于importance选择GQA组

"""

import os
import gc
import sys
import json
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM

from core import (
    # 剪枝方法
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups,
    # 重要性分析
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator,
    # 评估和数据
    PPLMetric,
    get_examples,
    get_examples_from_text,
    # 训练
    FineTuner,
    # 工具
    LoggerWithDepth,
)


def count_attention_params(layer):
    """统计单层 Attention 的参数量"""
    params = 0
    params += layer.self_attn.q_proj.weight.numel()
    params += layer.self_attn.k_proj.weight.numel()
    params += layer.self_attn.v_proj.weight.numel()
    params += layer.self_attn.o_proj.weight.numel()

    # bias（如果有）
    if layer.self_attn.q_proj.bias is not None:
        params += layer.self_attn.q_proj.bias.numel()
    if layer.self_attn.k_proj.bias is not None:
        params += layer.self_attn.k_proj.bias.numel()
    if layer.self_attn.v_proj.bias is not None:
        params += layer.self_attn.v_proj.bias.numel()
    if layer.self_attn.o_proj.bias is not None:
        params += layer.self_attn.o_proj.bias.numel()

    return params


def count_mlp_params(layer):
    """统计单层 MLP 的参数量"""
    params = 0
    params += layer.mlp.gate_proj.weight.numel()
    params += layer.mlp.up_proj.weight.numel()
    params += layer.mlp.down_proj.weight.numel()

    # bias（如果有）
    if layer.mlp.gate_proj.bias is not None:
        params += layer.mlp.gate_proj.bias.numel()
    if layer.mlp.up_proj.bias is not None:
        params += layer.mlp.up_proj.bias.numel()
    if layer.mlp.down_proj.bias is not None:
        params += layer.mlp.down_proj.bias.numel()

    return params


def compute_component_param_counts(model, layer_start=0, layer_end=None):
    """
    统计所有层的 Attention 和 MLP 参数量

    Returns:
        attention_params: Dict[layer_idx, param_count]
        mlp_params: Dict[layer_idx, param_count]
    """
    num_layers = len(model.model.layers)
    if layer_end is None:
        layer_end = num_layers

    attention_params = {}
    mlp_params = {}

    for layer_idx in range(layer_start, min(layer_end, num_layers)):
        layer = model.model.layers[layer_idx]
        attention_params[layer_idx] = count_attention_params(layer)
        mlp_params[layer_idx] = count_mlp_params(layer)

    return attention_params, mlp_params


def main():
    parser = argparse.ArgumentParser(description='Llama-3 GQA-Aware非均衡结构化剪枝')

    # 模型参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='原始模型路径')
    parser.add_argument('--save_ckpt_log_name', type=str, default='llama_gqa_aware_prune',
                       help='日志和模型保存目录名称')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.25,
                       help='目标剪枝率（相对于模型总参数量，0.25表示剪掉整个模型总参数的25%）')
    parser.add_argument('--pruning_distribution', type=str, default='5:5',
                       help='Attention和MLP的剪枝参数量比例（例如: 5:5表示各占一半, 10:0表示只剪MLP, 0:10表示只剪Attention）')

    # 层重要度评估
    parser.add_argument('--layer_importance_method', type=str, default='removal',
                       choices=['removal', 'activation'],
                       help='层重要度评估方法：removal(移除层) 或 activation(激活值)')
    parser.add_argument('--layer_importance_samples', type=int, default=50,
                       help='用于评估层重要度的样本数量')
    parser.add_argument('--skip_importance_analysis', action='store_true',
                       help='跳过层重要度分析，使用已保存的配置')
    parser.add_argument('--layer_importance_config', type=str, default='layer_importance_config.json',
                       help='层重要度配置文件路径')

    # 非均衡剪枝策略
    parser.add_argument('--pruning_strategy', type=str, default='inverse',
                       choices=['inverse', 'proportional', 'uniform'],
                       help='剪枝策略：inverse(重要层剪少), proportional(重要层剪多), uniform(均匀)')
    parser.add_argument('--layer_importance_weight', type=float, default=1.0,
                       help='层间剪枝率差异系数：越大层间差异越明显（推荐0.5-3.0）')
    parser.add_argument('--min_pruning_rate', type=float, default=0.0,
                       help='单层最小剪枝率（0表示允许不剪枝）')
    parser.add_argument('--max_pruning_rate', type=float, default=1.0,
                       help='单层最大剪枝率（1.0表示允许完全剪枝）')
    parser.add_argument('--freeze_top_n_layers', type=int, default=0,
                       help='冻结重要度最高的n层，这些层不参与剪枝（0表示不冻结任何层）')

    # 剪枝范围
    parser.add_argument('--layer_start', type=int, default=0,
                       help='剪枝起始层')
    parser.add_argument('--layer_end', type=int, default=32,
                       help='剪枝结束层')

    # 通道/头重要性评估
    parser.add_argument('--channel_importance_samples', type=int, default=10,
                       help='用于计算通道/头Taylor重要性的样本数（Attention层评估头，MLP层评估通道）')
    parser.add_argument('--taylor_seq_len', type=int, default=128,
                       help='Taylor重要性计算时的序列长度')

    # 其他参数
    parser.add_argument('--save_model', action='store_true',
                       help='是否保存模型')
    parser.add_argument('--test_original_ppl', action='store_true',
                       help='剪枝前是否评估原模型PPL（作为baseline）')
    parser.add_argument('--test_after_prune', action='store_true',
                       help='剪枝后是否评估PPL')
    parser.add_argument('--eval_seq_len', type=int, default=128,
                       help='PPL评估时的序列长度（与剪枝后、微调后保持一致）')

    # GQA配置
    parser.add_argument('--head_dim', type=int, default=128,
                       help='每个attention head的维度')
    parser.add_argument('--gqa_ratio', type=int, default=4,
                       help='Q:KV比例（Llama-3默认4:1）')

    # 微调参数
    parser.add_argument('--finetune', action='store_true',
                       help='剪枝后是否进行微调')
    parser.add_argument('--finetune_method', type=str, default='full',
                       choices=['full', 'lora'],
                       help='微调方法：full(全参数微调) 或 lora(LoRA微调)')
    parser.add_argument('--finetune_lr', type=float, default=1e-5,
                       help='微调学习率（LoRA建议2e-4，全参数建议1e-5）')
    parser.add_argument('--finetune_epochs', type=int, default=1,
                       help='微调轮数')
    parser.add_argument('--finetune_samples', type=int, default=500,
                       help='微调使用的样本数量')
    parser.add_argument('--finetune_batch_size', type=int, default=1,
                       help='微调batch size')
    parser.add_argument('--finetune_seq_len', type=int, default=512,
                       help='微调序列长度')
    parser.add_argument('--finetune_grad_accum', type=int, default=4,
                       help='梯度累积步数（有效batch size = batch_size * grad_accum）')
    parser.add_argument('--finetune_max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪阈值')
    parser.add_argument('--finetune_weight_decay', type=float, default=0.01,
                       help='权重衰减系数')
    parser.add_argument('--finetune_warmup_steps', type=int, default=0,
                       help='学习率预热步数')

    # LoRA专用参数
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA秩（越大效果越好但参数越多，建议4-16）')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA缩放系数（通常设为r的2倍）')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout率')
    parser.add_argument('--lora_target_attention', action='store_true', default=True,
                       help='LoRA是否应用到Attention层（q,k,v,o）')
    parser.add_argument('--lora_target_mlp', action='store_true', default=True,
                       help='LoRA是否应用到MLP层（gate,up,down）')

    args = parser.parse_args()

    # 解析 pruning_distribution 参数（支持浮点数）
    try:
        attn_ratio, mlp_ratio = map(float, args.pruning_distribution.split(':'))
        if attn_ratio < 0 or mlp_ratio < 0:
            raise ValueError("剪枝比例不能为负数")
        if attn_ratio == 0 and mlp_ratio == 0:
            raise ValueError("Attention 和 MLP 的剪枝比例不能同时为0")
    except ValueError as e:
        raise ValueError(f"无效的 pruning_distribution 参数 '{args.pruning_distribution}': {e}")

    args.attn_ratio = attn_ratio
    args.mlp_ratio = mlp_ratio
    total_ratio = attn_ratio + mlp_ratio

    # 自动选择最优 GPU
    try:
        from core.utils.get_best_gpu import get_best_gpu
        device = f"cuda:{get_best_gpu()}"
    except:
        device = "cuda:0"

    print(f"自动选择设备: {device}")
    args.device = device

    # 输出剪枝配置
    print(f"剪枝配置:")
    print(f"  总剪枝率: {args.pruning_ratio:.2%}")
    print(f"  Attention:MLP 剪枝比例 = {attn_ratio}:{mlp_ratio}")
    if total_ratio > 0:
        print(f"    -> Attention 占总剪枝量的 {attn_ratio/total_ratio:.1%}")
        print(f"    -> MLP 占总剪枝量的 {mlp_ratio/total_ratio:.1%}")

    # 创建日志
    logger = LoggerWithDepth(
        env_name=args.save_ckpt_log_name,
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )

    # ==================== 步骤1: 加载模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤1: 加载模型")
    logger.log("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # 确保 tokenizer 有 pad_token (Llama 等模型默认没有)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.log("设置 pad_token = eos_token")

    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    model.half()

    # 启用梯度
    for param in model.parameters():
        param.requires_grad_(True)

    num_layers = len(model.model.layers)
    logger.log(f"模型总层数: {num_layers}")

    # 统计剪枝前参数量（所有参数）
    before_pruning_parameters = sum(p.numel() for p in model.parameters())
    logger.log(f"模型总参数量: {before_pruning_parameters:,}")

    # 统计原始模型的 Attention 和 MLP 参数量
    logger.log("\n" + "=" * 60)
    logger.log("原始模型参数量统计")
    logger.log("=" * 60)

    original_attention_params, original_mlp_params = compute_component_param_counts(
        model, 0, num_layers
    )
    total_original_attention = sum(original_attention_params.values())
    total_original_mlp = sum(original_mlp_params.values())
    total_original_prunable = total_original_attention + total_original_mlp

    logger.log(f"\n所有层的参数量统计:")
    logger.log(f"  Attention 总参数量: {total_original_attention:,}")
    logger.log(f"  MLP 总参数量: {total_original_mlp:,}")
    logger.log(f"  Attention+MLP 总和: {total_original_prunable:,}")
    logger.log(f"  其他参数（embedding, norm等）: {before_pruning_parameters - total_original_prunable:,}")

    logger.log(f"\n参数量占比:")
    logger.log(f"  Attention: {total_original_attention/total_original_prunable:.2%} of (Attention+MLP)")
    logger.log(f"  MLP: {total_original_mlp/total_original_prunable:.2%} of (Attention+MLP)")

    # 计算归一化的比值（x+y=10），与 --pruning_distribution 格式一致
    ratio = total_original_attention / total_original_mlp
    ratio_sum = ratio + 1
    attn_normalized = ratio / ratio_sum * 10
    mlp_normalized = 1 / ratio_sum * 10
    logger.log(f"  Attention:MLP 比值 = {attn_normalized:.1f}:{mlp_normalized:.1f} (归一化，总和为10)")

    logger.log(f"\n占全局模型的比例:")
    logger.log(f"  Attention: {total_original_attention/before_pruning_parameters:.2%} of total")
    logger.log(f"  MLP: {total_original_mlp/before_pruning_parameters:.2%} of total")
    logger.log(f"  Attention+MLP: {total_original_prunable/before_pruning_parameters:.2%} of total")

    # ==================== 统一数据加载 ====================
    logger.log("\n" + "=" * 60)
    logger.log("加载评估数据")
    logger.log("=" * 60)

    # 计算所需的最大样本数
    max_samples = max(args.layer_importance_samples, args.channel_importance_samples) if not args.skip_importance_analysis else args.channel_importance_samples

    # 一次性加载所有需要的样本（从 wikitext2）
    logger.log(f"从 wikitext2 加载 {max_samples} 个样本...")
    all_samples = get_examples('wikitext', tokenizer, num_samples=max_samples, seq_len=512, split='test')
    logger.log(f"✅ 加载完成，shape: {all_samples.shape}")

    # ==================== 原模型PPL评估（可选） ====================
    ppl_original = None
    if args.test_original_ppl:
        logger.log("\n" + "=" * 60)
        logger.log("评估原始模型PPL（Baseline）")
        logger.log("=" * 60)

        model.eval()
        logger.log(f"使用数据集: wikitext2, seq_len={args.eval_seq_len}")
        ppl_original = PPLMetric(model, tokenizer, ['wikitext2'],
                                seq_len=args.eval_seq_len, device=args.device)
        logger.log(f"原始模型 PPL: {ppl_original}")

        # 重新启用梯度（为后续剪枝准备）
        for param in model.parameters():
            param.requires_grad_(True)
        model.train()

    # ==================== 步骤2: 评估层重要性 ====================
    if not args.skip_importance_analysis:
        logger.log("\n" + "=" * 60)
        logger.log("步骤2: 评估层重要性")
        logger.log("=" * 60)

        # 将 tokenized 样本转换回文本（用于层重要性评估）
        eval_samples = all_samples[:args.layer_importance_samples]
        eval_texts = [tokenizer.decode(sample, skip_special_tokens=True) for sample in eval_samples]
        logger.log(f"准备了 {len(eval_texts)} 个样本用于层重要性评估")

        analyzer = LayerImportanceAnalyzer(model, tokenizer, device=args.device)

        if args.layer_importance_method == 'removal':
            logger.log("使用层移除法评估重要性...")
            layer_importance = analyzer.measure_layer_importance_by_removal(
                eval_texts, num_layers=num_layers
            )
        else:
            logger.log("使用激活值法评估重要性...")
            layer_importance = analyzer.measure_layer_importance_by_activation(eval_texts)

        # 只打印统计信息和极值层
        importance_values = list(layer_importance.values())
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)

        logger.log(f"\n层重要性统计:")
        logger.log(f"  平均: {np.mean(importance_values):.6f}")
        logger.log(f"  标准差: {np.std(importance_values):.6f}")
        logger.log(f"  最大: {max(importance_values):.6f}")
        logger.log(f"  最小: {min(importance_values):.6f}")

        logger.log(f"\n最重要的5层:")
        for layer_idx, importance in sorted_layers[:5]:
            logger.log(f"  Layer {layer_idx}: {importance:.6f}")

        logger.log(f"最不重要的5层:")
        for layer_idx, importance in sorted_layers[-5:]:
            logger.log(f"  Layer {layer_idx}: {importance:.6f}")

    else:
        logger.log("跳过层重要度分析，加载已保存的配置...")
        calculator = UnbalancedStructuredPruningCalculator({}, num_layers)
        layer_pruning_rates = calculator.load_pruning_rates(args.layer_importance_config)
        layer_importance = {i: 1.0 for i in range(num_layers)}

    # ==================== 步骤3: 统计参数量并计算剪枝策略 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤3: 统计参数量并计算剪枝策略")
    logger.log("=" * 60)

    # 统计 Attention 和 MLP 的参数量
    attention_param_counts, mlp_param_counts = compute_component_param_counts(
        model, args.layer_start, args.layer_end
    )

    total_attention_params = sum(attention_param_counts.values())
    total_mlp_params = sum(mlp_param_counts.values())
    total_prunable_params = total_attention_params + total_mlp_params

    logger.log(f"\n参与剪枝的层范围: [{args.layer_start}, {args.layer_end})")
    logger.log(f"Attention 总参数量: {total_attention_params:,} ({total_attention_params/total_prunable_params:.1%})")
    logger.log(f"MLP 总参数量: {total_mlp_params:,} ({total_mlp_params/total_prunable_params:.1%})")
    logger.log(f"可剪枝总参数量: {total_prunable_params:,}")

    # 根据 pruning_distribution 分配剪枝参数量
    logger.log(f"\n" + "-" * 60)
    logger.log("剪枝参数量分配计算过程:")
    logger.log("-" * 60)

    # 计算总目标剪枝参数量（基于整个模型的总参数量）
    total_pruned_params = int(before_pruning_parameters * args.pruning_ratio)
    logger.log(f"\n1. 计算总目标剪枝参数量:")
    logger.log(f"   total_pruned = {before_pruning_parameters:,} (模型总参数) × {args.pruning_ratio}")
    logger.log(f"   total_pruned = {total_pruned_params:,}")

    logger.log(f"\n2. 根据分布比例 {args.attn_ratio}:{args.mlp_ratio} 分配:")
    total_ratio_sum = args.attn_ratio + args.mlp_ratio
    logger.log(f"   Attention 占比 = {args.attn_ratio}/{total_ratio_sum} = {args.attn_ratio/total_ratio_sum:.2%}")
    logger.log(f"   MLP 占比 = {args.mlp_ratio}/{total_ratio_sum} = {args.mlp_ratio/total_ratio_sum:.2%}")

    attn_pruned_params = int(total_pruned_params * args.attn_ratio / total_ratio_sum)
    mlp_pruned_params = int(total_pruned_params * args.mlp_ratio / total_ratio_sum)

    logger.log(f"\n3. 计算各组件目标剪枝量:")
    logger.log(f"   Attention 目标剪枝量 = {total_pruned_params:,} × {args.attn_ratio/total_ratio_sum:.2%} = {attn_pruned_params:,}")
    logger.log(f"   MLP 目标剪枝量 = {total_pruned_params:,} × {args.mlp_ratio/total_ratio_sum:.2%} = {mlp_pruned_params:,}")

    logger.log(f"\n4. 计算各组件剪枝率:")
    attn_prune_rate = attn_pruned_params / total_attention_params if total_attention_params > 0 else 0
    mlp_prune_rate = mlp_pruned_params / total_mlp_params if total_mlp_params > 0 else 0
    logger.log(f"   Attention 剪枝率 = {attn_pruned_params:,} / {total_attention_params:,} = {attn_prune_rate:.2%}")
    logger.log(f"   MLP 剪枝率 = {mlp_pruned_params:,} / {total_mlp_params:,} = {mlp_prune_rate:.2%}")

    # 异常检查
    logger.log(f"\n5. 参数量有效性检查:")
    errors = []
    if attn_pruned_params > total_attention_params:
        errors.append(f"Attention 目标剪枝量 ({attn_pruned_params:,}) 超过实际参数量 ({total_attention_params:,})")
    if mlp_pruned_params > total_mlp_params:
        errors.append(f"MLP 目标剪枝量 ({mlp_pruned_params:,}) 超过实际参数量 ({total_mlp_params:,})")

    if errors:
        logger.log("   ❌ 发现错误:")
        for error in errors:
            logger.log(f"      - {error}")
        raise ValueError(f"剪枝参数量配置错误，请调整 --pruning_ratio 或 --pruning_distribution。\n详细信息:\n" + "\n".join(errors))
    else:
        logger.log("   ✅ 参数量配置有效")

    logger.log(f"\n6. 验证全局剪枝率:")
    global_prune_rate = total_pruned_params / before_pruning_parameters
    logger.log(f"   实际全局剪枝率 = {total_pruned_params:,} / {before_pruning_parameters:,} = {global_prune_rate:.2%}")
    logger.log(f"   目标全局剪枝率 = {args.pruning_ratio:.1%}")

    logger.log(f"\n" + "=" * 60)
    logger.log("剪枝配置总结:")
    logger.log("=" * 60)
    logger.log(f"总目标剪枝量: {total_pruned_params:,} ({args.pruning_ratio:.1%} of total 模型参数)")
    logger.log(f"  -> Attention: {attn_pruned_params:,} ({attn_prune_rate:.2%} of Attention, {attn_pruned_params/before_pruning_parameters:.2%} of total)")
    logger.log(f"  -> MLP: {mlp_pruned_params:,} ({mlp_prune_rate:.2%} of MLP, {mlp_pruned_params/before_pruning_parameters:.2%} of total)")

    # 创建计算器
    calculator = UnbalancedStructuredPruningCalculator(layer_importance, num_layers)

    # ========== 层冻结机制 ==========
    frozen_layers = set()
    if args.freeze_top_n_layers > 0:
        logger.log(f"\n" + "=" * 60)
        logger.log(f"层冻结机制: 冻结重要度最高的 {args.freeze_top_n_layers} 层")
        logger.log("=" * 60)

        # 按重要度排序，选择最重要的n层
        sorted_layers = sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)
        frozen_layers = set([layer_idx for layer_idx, _ in sorted_layers[:args.freeze_top_n_layers]])

        logger.log(f"\n冻结的层索引: {sorted(frozen_layers)}")
        logger.log(f"冻结层的重要度:")
        for layer_idx in sorted(frozen_layers):
            logger.log(f"  Layer {layer_idx}: {layer_importance[layer_idx]:.6f}")

        # 计算冻结层的参数量
        frozen_attn_params = sum(attention_param_counts[i] for i in frozen_layers if i in attention_param_counts)
        frozen_mlp_params = sum(mlp_param_counts[i] for i in frozen_layers if i in mlp_param_counts)
        logger.log(f"\n冻结层总参数量:")
        logger.log(f"  Attention: {frozen_attn_params:,}")
        logger.log(f"  MLP: {frozen_mlp_params:,}")
        logger.log(f"  合计: {frozen_attn_params + frozen_mlp_params:,}")

        # 创建过滤后的参数计数字典（只包含未冻结的层）
        attention_param_counts_unfrozen = {k: v for k, v in attention_param_counts.items() if k not in frozen_layers}
        mlp_param_counts_unfrozen = {k: v for k, v in mlp_param_counts.items() if k not in frozen_layers}

        unfrozen_attn_params = sum(attention_param_counts_unfrozen.values())
        unfrozen_mlp_params = sum(mlp_param_counts_unfrozen.values())
        logger.log(f"\n未冻结层总参数量:")
        logger.log(f"  Attention: {unfrozen_attn_params:,}")
        logger.log(f"  MLP: {unfrozen_mlp_params:,}")
        logger.log(f"  合计: {unfrozen_attn_params + unfrozen_mlp_params:,}")

        logger.log(f"\n⚠️  剪枝任务将由 {len(attention_param_counts_unfrozen)} 个未冻结层承担")
        logger.log("=" * 60)
    else:
        attention_param_counts_unfrozen = attention_param_counts
        mlp_param_counts_unfrozen = mlp_param_counts

    # 分别计算 Attention 和 MLP 的层级剪枝率（仅针对未冻结的层）
    if attn_pruned_params > 0 and len(attention_param_counts_unfrozen) > 0:
        attn_layer_pruning_rates_unfrozen = calculator.compute_layer_pruning_rates_by_target_params(
            layer_param_counts=attention_param_counts_unfrozen,
            target_total_pruned_params=attn_pruned_params,
            strategy=args.pruning_strategy,
            alpha=args.layer_importance_weight,
            min_rate=args.min_pruning_rate,
            max_rate=args.max_pruning_rate,
            use_log_transform=True
        )
        # 为所有层创建剪枝率字典，冻结层设为0
        attn_layer_pruning_rates = {i: 0.0 for i in attention_param_counts.keys()}
        attn_layer_pruning_rates.update(attn_layer_pruning_rates_unfrozen)
    else:
        attn_layer_pruning_rates = {i: 0.0 for i in attention_param_counts.keys()}

    if mlp_pruned_params > 0 and len(mlp_param_counts_unfrozen) > 0:
        mlp_layer_pruning_rates_unfrozen = calculator.compute_layer_pruning_rates_by_target_params(
            layer_param_counts=mlp_param_counts_unfrozen,
            target_total_pruned_params=mlp_pruned_params,
            strategy=args.pruning_strategy,
            alpha=args.layer_importance_weight,
            min_rate=args.min_pruning_rate,
            max_rate=args.max_pruning_rate,
            use_log_transform=True
        )
        # 为所有层创建剪枝率字典，冻结层设为0
        mlp_layer_pruning_rates = {i: 0.0 for i in mlp_param_counts.keys()}
        mlp_layer_pruning_rates.update(mlp_layer_pruning_rates_unfrozen)
    else:
        mlp_layer_pruning_rates = {i: 0.0 for i in mlp_param_counts.keys()}

    # ========== GQA 离散化映射和 MLP 修正 ==========
    if attn_pruned_params > 0:
        logger.log(f"\n" + "=" * 60)
        logger.log("GQA 离散化映射和 MLP 修正过程:")
        logger.log("=" * 60)

        # 获取第一层的KV头数量（所有层相同）
        num_kv_heads = model.model.layers[0].self_attn.k_proj.out_features // args.head_dim
        logger.log(f"\n每层KV头数量: {num_kv_heads}")
        logger.log(f"GQA可用剪枝档位: {[i/num_kv_heads for i in range(num_kv_heads+1)]}")

        # 映射Attention剪枝率到离散档位
        logger.log(f"\n1. 映射 Attention 各层剪枝率到 GQA 离散档位:")
        logger.log(f"   {'Layer':<8} {'原始剪枝率':<12} {'映射后剪枝率':<14} {'剪枝KV头数':<12}")
        logger.log(f"   {'-'*8} {'-'*12} {'-'*14} {'-'*12}")

        attn_layer_pruning_rates_discretized = {}
        for layer_idx, original_rate in attn_layer_pruning_rates.items():
            # 计算要剪枝的KV头数量（四舍五入到最近的整数）
            num_pruned_kv_heads = round(original_rate * num_kv_heads)
            num_pruned_kv_heads = max(0, min(num_kv_heads, num_pruned_kv_heads))  # 限制在 [0, num_kv_heads]

            # 映射后的剪枝率
            discretized_rate = num_pruned_kv_heads / num_kv_heads
            attn_layer_pruning_rates_discretized[layer_idx] = discretized_rate

            logger.log(f"   {layer_idx:<8} {original_rate:<12.4f} {discretized_rate:<14.4f} {num_pruned_kv_heads:<12}")

        # 计算映射后的实际Attention剪枝量
        logger.log(f"\n2. 计算映射后的实际 Attention 剪枝量:")
        actual_attn_pruned_params = sum(
            attn_layer_pruning_rates_discretized[i] * attention_param_counts[i]
            for i in attn_layer_pruning_rates_discretized.keys()
        )
        actual_attn_pruned_params = int(actual_attn_pruned_params)

        logger.log(f"   原始计划剪枝量: {attn_pruned_params:,}")
        logger.log(f"   映射后实际剪枝量: {actual_attn_pruned_params:,}")

        deviation = attn_pruned_params - actual_attn_pruned_params
        logger.log(f"   偏差量: {deviation:,} ({deviation/attn_pruned_params*100:+.2f}%)")

        # 将偏差补偿到MLP
        logger.log(f"\n3. 将偏差补偿到 MLP:")
        logger.log(f"   MLP 原始目标剪枝量: {mlp_pruned_params:,}")
        mlp_pruned_params_adjusted = mlp_pruned_params + deviation
        logger.log(f"   MLP 修正后目标剪枝量: {mlp_pruned_params_adjusted:,} (补偿 {deviation:+,})")

        # 检查修正后的MLP剪枝量是否有效
        if mlp_pruned_params_adjusted < 0:
            logger.log(f"   ⚠️  修正后的 MLP 剪枝量为负，设为 0")
            mlp_pruned_params_adjusted = 0
        elif mlp_pruned_params_adjusted > total_mlp_params:
            logger.log(f"   ⚠️  修正后的 MLP 剪枝量超过总量，设为 {total_mlp_params:,}")
            mlp_pruned_params_adjusted = total_mlp_params

        # 重新计算MLP层级剪枝率（仅对未冻结层）
        if mlp_pruned_params_adjusted > 0 and len(mlp_param_counts_unfrozen) > 0:
            logger.log(f"\n4. 重新计算 MLP 各层剪枝率（仅未冻结层）:")
            mlp_layer_pruning_rates_unfrozen = calculator.compute_layer_pruning_rates_by_target_params(
                layer_param_counts=mlp_param_counts_unfrozen,  # 只计算未冻结层
                target_total_pruned_params=mlp_pruned_params_adjusted,
                strategy=args.pruning_strategy,
                alpha=args.layer_importance_weight,
                min_rate=args.min_pruning_rate,
                max_rate=args.max_pruning_rate,
                use_log_transform=True
            )
            # 为所有层创建剪枝率字典，冻结层设为0
            mlp_layer_pruning_rates = {i: 0.0 for i in mlp_param_counts.keys()}
            mlp_layer_pruning_rates.update(mlp_layer_pruning_rates_unfrozen)
            logger.log(f"   ✅ MLP 层级剪枝率已根据修正后的目标量重新计算（冻结层剪枝率=0）")
        else:
            mlp_layer_pruning_rates = {i: 0.0 for i in mlp_param_counts.keys()}
            logger.log(f"   MLP 剪枝量为 0 或无未冻结层，所有层剪枝率设为 0")

        # 更新Attention剪枝率为离散化后的值
        attn_layer_pruning_rates = attn_layer_pruning_rates_discretized

        # 更新实际剪枝参数量
        attn_pruned_params = actual_attn_pruned_params
        mlp_pruned_params = mlp_pruned_params_adjusted

        logger.log(f"\n" + "=" * 60)
        logger.log("修正后的最终剪枝配置:")
        logger.log("=" * 60)
        logger.log(f"Attention 实际剪枝量: {attn_pruned_params:,} ({attn_pruned_params/total_attention_params:.2%} of Attention)")
        logger.log(f"MLP 实际剪枝量: {mlp_pruned_params:,} ({mlp_pruned_params/total_mlp_params:.2%} of MLP)")
        logger.log(f"总计剪枝量: {attn_pruned_params + mlp_pruned_params:,}")
        logger.log("=" * 60)

    logger.log(f"\nAttention 剪枝率统计 (离散化后):")
    logger.log(f"  平均: {np.mean(list(attn_layer_pruning_rates.values())):.4f}")
    logger.log(f"  最小: {np.min(list(attn_layer_pruning_rates.values())):.4f}")
    logger.log(f"  最大: {np.max(list(attn_layer_pruning_rates.values())):.4f}")

    logger.log(f"\nMLP 剪枝率统计:")
    logger.log(f"  平均: {np.mean(list(mlp_layer_pruning_rates.values())):.4f}")
    logger.log(f"  最小: {np.min(list(mlp_layer_pruning_rates.values())):.4f}")
    logger.log(f"  最大: {np.max(list(mlp_layer_pruning_rates.values())):.4f}")

    # 保存配置
    config_path = os.path.join(logger.log_dir, 'pruning_strategy_config.json')
    config = {
        'attention_pruning_rates': {str(k): v for k, v in attn_layer_pruning_rates.items()},
        'mlp_pruning_rates': {str(k): v for k, v in mlp_layer_pruning_rates.items()},
        'layer_importance': {str(k): v for k, v in layer_importance.items()},
        'pruning_distribution': args.pruning_distribution,
        'target_pruning_ratio': args.pruning_ratio,
        'attention_pruned_params': attn_pruned_params,
        'mlp_pruned_params': mlp_pruned_params,
        'total_attention_params': total_attention_params,
        'total_mlp_params': total_mlp_params
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.log(f"\n剪枝策略配置已保存到: {config_path}")

    # 可视化（使用 Attention 的剪枝率作为示例）
    viz_path = os.path.join(logger.log_dir, 'pruning_strategy.png')
    calculator.visualize_pruning_strategy(attn_layer_pruning_rates, save_path=viz_path)

    # ==================== 步骤4: GQA-Aware剪枝 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤4: GQA-Aware结构化剪枝")
    logger.log("=" * 60)

    # 确定剪枝目标
    prune_targets = []
    if attn_pruned_params > 0:
        prune_targets.append("Attention(GQA组级)")
    if mlp_pruned_params > 0:
        prune_targets.append("MLP")

    if not prune_targets:
        logger.log("⚠️ 警告：Attention 和 MLP 的目标剪枝量都为0")
        logger.log("跳过剪枝步骤\n")
    else:
        logger.log(f"剪枝目标: {', '.join(prune_targets)}")
        logger.log(f"剪枝分布 (Attention:MLP) = {args.attn_ratio}:{args.mlp_ratio}")
        logger.log("使用Taylor Importance，保持4:1 Q:KV比例\n")

    # 只有在有剪枝目标时才执行剪枝
    if prune_targets:
        # 准备样本数据用于计算梯度（复用已加载的数据，截断到64 token）
        example_prompts = all_samples[:args.channel_importance_samples, :64].to(args.device)
        logger.log(f"准备了 {args.channel_importance_samples} 个样本用于Taylor importance计算")

        # 确定要剪枝的层（只要 Attention 或 MLP 的剪枝率 >= min_rate）
        pruning_layers = [i for i in range(args.layer_start, min(args.layer_end, num_layers))
                         if (attn_layer_pruning_rates.get(i, 0.0) >= args.min_pruning_rate or
                             mlp_layer_pruning_rates.get(i, 0.0) >= args.min_pruning_rate)]

        logger.log(f"\n实际参与剪枝的层: {pruning_layers}")
        logger.log(f"跳过的层: {[i for i in range(args.layer_start, min(args.layer_end, num_layers)) if i not in pruning_layers]}\n")

        # 记录已剪枝的层（用于禁用梯度计算）
        pruned_layer_indices = []

        # 逐层剪枝
        for layer_idx in pruning_layers:
            attn_rate = attn_layer_pruning_rates.get(layer_idx, 0.0)
            mlp_rate = mlp_layer_pruning_rates.get(layer_idx, 0.0)

            logger.log(f"\n处理 Layer {layer_idx}")
            logger.log(f"  Attention 剪枝率: {attn_rate:.2%}, MLP 剪枝率: {mlp_rate:.2%}")

            layer = model.model.layers[layer_idx]

            # 禁用已剪枝层的梯度计算（避免形状不匹配）
            for pruned_idx in pruned_layer_indices:
                for param in model.model.layers[pruned_idx].parameters():
                    param.requires_grad = False

            # 计算梯度（如果需要）
            prune_attn = attn_rate >= args.min_pruning_rate
            prune_mlp = mlp_rate >= args.min_pruning_rate

            if prune_attn or prune_mlp:
                model.zero_grad()
                loss = model(example_prompts, labels=example_prompts).loss
                loss.backward()

            # 初始化默认值
            num_q, num_kv = None, None
            target_channels = None
            original_mlp_channels = None

            # 执行 Attention 剪枝
            if prune_attn:
                group_imp = compute_gqa_group_importance(layer, args.head_dim, args.gqa_ratio)
                num_kv_heads = len(group_imp)
                num_groups_to_prune = int(num_kv_heads * attn_rate)
                target_num_kv_heads = max(1, num_kv_heads - num_groups_to_prune)

                keep_indices, _ = select_gqa_groups_to_prune(group_imp, target_num_kv_heads)
                num_q, num_kv = prune_attention_by_gqa_groups(layer, keep_indices, args.head_dim, args.gqa_ratio)

            # 执行 MLP 剪枝
            if prune_mlp:
                gate_salience = (layer.mlp.gate_proj.weight * layer.mlp.gate_proj.weight.grad).abs().sum(1)
                up_salience = (layer.mlp.up_proj.weight * layer.mlp.up_proj.weight.grad).abs().sum(1)
                down_salience = (layer.mlp.down_proj.weight * layer.mlp.down_proj.weight.grad).abs().sum(0)
                mlp_importance = gate_salience + up_salience + down_salience

                original_mlp_channels = mlp_importance.shape[0]
                num_channels_to_prune = int(original_mlp_channels * mlp_rate)
                num_channels_to_prune = (num_channels_to_prune // args.head_dim) * args.head_dim
                target_channels = max(args.head_dim, original_mlp_channels - num_channels_to_prune)

                _, sorted_indices = torch.sort(mlp_importance, descending=True)
                keep_indices_mlp = sorted(sorted_indices[:target_channels].tolist())

                layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[keep_indices_mlp, :]
                layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[keep_indices_mlp, :]

                if layer.mlp.gate_proj.bias is not None:
                    layer.mlp.gate_proj.bias.data = layer.mlp.gate_proj.bias.data[keep_indices_mlp]
                if layer.mlp.up_proj.bias is not None:
                    layer.mlp.up_proj.bias.data = layer.mlp.up_proj.bias.data[keep_indices_mlp]

                layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, keep_indices_mlp]

                layer.mlp.gate_proj.out_features = target_channels
                layer.mlp.up_proj.out_features = target_channels
                layer.mlp.down_proj.in_features = target_channels

            # 输出日志
            log_parts = []
            if prune_attn:
                log_parts.append(f"Attention: {32}Q:{8}KV → {num_q}Q:{num_kv}KV")
            if prune_mlp:
                log_parts.append(f"MLP: {original_mlp_channels}→{target_channels}")

            if log_parts:
                logger.log(f"  结果: {', '.join(log_parts)}")
            else:
                logger.log(f"  ⚠️ 跳过（剪枝率低于阈值）")

            # 清理
            if prune_attn or prune_mlp:
                model.zero_grad()
                for param in layer.parameters():
                    if param.grad is not None:
                        param.grad = None
            torch.cuda.empty_cache()

            pruned_layer_indices.append(layer_idx)

    # ==================== 步骤5: 保存模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤5: 保存剪枝后的模型")
    logger.log("=" * 60)

    if args.save_model:
        model.half()
        save_dict = {
            'model': model,
            'tokenizer': tokenizer,
            'attention_pruning_rates': attn_layer_pruning_rates,
            'mlp_pruning_rates': mlp_layer_pruning_rates,
            'layer_importance': layer_importance,
            'pruning_method': 'gqa_aware_taylor_distributed',
            'pruning_distribution': args.pruning_distribution,
            'attention_pruned_params': attn_pruned_params,
            'mlp_pruned_params': mlp_pruned_params,
            'config': args.__dict__
        }

        torch.save(save_dict, logger.best_checkpoint_path)
        logger.log(f"✅ 模型已保存到: {logger.best_checkpoint_path}")
    else:
        logger.log("⚠️ 未启用 --save_model，跳过模型保存")

    # ==================== 步骤6: 重新加载模型 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤6: 重新加载模型")
    logger.log("=" * 60)

    if args.save_model:
        # 删除原模型，释放内存
        logger.log("删除原模型副本，释放显存...")
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 重新加载保存的模型
        logger.log(f"从检查点重新加载模型: {logger.best_checkpoint_path}")
        checkpoint = torch.load(logger.best_checkpoint_path, weights_only=False)
        model = checkpoint['model']
        tokenizer = checkpoint['tokenizer']
        logger.log("✅ 模型重新加载成功")
    else:
        logger.log("⚠️ 未保存模型，使用内存中的模型继续")

    # ==================== 步骤7: 最终统计 ====================
    logger.log("\n" + "=" * 60)
    logger.log("步骤7: 最终统计")
    logger.log("=" * 60)

    # 统计剪枝后的 Attention 和 MLP 参数量
    final_attention_params, final_mlp_params = compute_component_param_counts(model, 0, num_layers)
    total_final_attention = sum(final_attention_params.values())
    total_final_mlp = sum(final_mlp_params.values())

    # 统计剪枝后总参数量
    final_parameters = sum(p.numel() for p in model.parameters())
    total_final_prunable = total_final_attention + total_final_mlp

    logger.log(f"\n参数量详细统计:")
    logger.log(f"\n{'组件':<15} {'剪枝前':<20} {'剪枝后':<20} {'减少量':<20} {'减少率':<10}")
    logger.log("-" * 85)

    # Attention 统计
    attn_reduced = total_original_attention - total_final_attention
    attn_reduction_rate = attn_reduced / total_original_attention if total_original_attention > 0 else 0
    logger.log(f"{'Attention':<15} {total_original_attention:<20,} {total_final_attention:<20,} {attn_reduced:<20,} {attn_reduction_rate:<10.2%}")

    # MLP 统计
    mlp_reduced = total_original_mlp - total_final_mlp
    mlp_reduction_rate = mlp_reduced / total_original_mlp if total_original_mlp > 0 else 0
    logger.log(f"{'MLP':<15} {total_original_mlp:<20,} {total_final_mlp:<20,} {mlp_reduced:<20,} {mlp_reduction_rate:<10.2%}")

    # Attention+MLP 合计
    prunable_reduced = (total_original_attention + total_original_mlp) - (total_final_attention + total_final_mlp)
    prunable_reduction_rate = prunable_reduced / total_original_prunable if total_original_prunable > 0 else 0
    logger.log("-" * 85)
    logger.log(f"{'Attn+MLP 合计':<15} {total_original_prunable:<20,} {total_final_prunable:<20,} {prunable_reduced:<20,} {prunable_reduction_rate:<10.2%}")

    # 其他参数（embedding, norm等）
    other_original = before_pruning_parameters - total_original_prunable
    other_final = final_parameters - total_final_prunable
    other_reduced = other_original - other_final
    logger.log(f"{'其他参数':<15} {other_original:<20,} {other_final:<20,} {other_reduced:<20,} {'-':<10}")

    # 总计
    total_reduced = before_pruning_parameters - final_parameters
    total_reduction_rate = total_reduced / before_pruning_parameters if before_pruning_parameters > 0 else 0
    logger.log("-" * 85)
    logger.log(f"{'模型总计':<15} {before_pruning_parameters:<20,} {final_parameters:<20,} {total_reduced:<20,} {total_reduction_rate:<10.2%}")

    logger.log(f"\n参数量占比分析:")
    logger.log(f"  剪枝前 Attention:MLP = {total_original_attention/total_original_mlp:.3f}:1")
    logger.log(f"  剪枝后 Attention:MLP = {total_final_attention/total_final_mlp:.3f}:1" if total_final_mlp > 0 else "  剪枝后 MLP已完全剪除")

    logger.log(f"\n剪枝量占比:")
    logger.log(f"  相对 Attention+MLP: {prunable_reduction_rate:.2%} (目标: {args.pruning_ratio:.2%})")
    logger.log(f"  相对模型总参数量: {total_reduction_rate:.2%}")

    # 计算物理大小（假设 float16，每个参数 2 bytes）
    before_size_gb = before_pruning_parameters * 2 / (1024**3)
    final_size_gb = final_parameters * 2 / (1024**3)
    logger.log(f"\n模型大小（FP16）:")
    logger.log(f"  剪枝前: {before_size_gb:.2f} GB")
    logger.log(f"  剪枝后: {final_size_gb:.2f} GB")
    logger.log(f"  减少: {before_size_gb - final_size_gb:.2f} GB ({(before_size_gb - final_size_gb)/before_size_gb:.2%})")

    # 验证所有层保持4:1 GQA比例
    logger.log(f"\nGQA比例验证:")
    gqa_ratios = []
    for idx, layer in enumerate(model.model.layers):
        # 检查是否有 num_heads 属性（剪枝后会设置）
        if hasattr(layer.self_attn, 'num_heads') and hasattr(layer.self_attn, 'num_key_value_heads'):
            num_q = layer.self_attn.num_heads
            num_kv = layer.self_attn.num_key_value_heads
        else:
            # 未剪枝的层，从权重维度计算
            # q_proj: (num_q_heads * head_dim, hidden_size)
            # k_proj: (num_kv_heads * head_dim, hidden_size)
            num_q = layer.self_attn.q_proj.out_features // args.head_dim
            num_kv = layer.self_attn.k_proj.out_features // args.head_dim

        ratio = num_q // num_kv if num_kv > 0 else 0
        gqa_ratios.append(ratio)

    all_4_to_1 = all(ratio == 4 for ratio in gqa_ratios)
    unique_ratios = set(gqa_ratios)

    if all_4_to_1:
        logger.log(f"  ✅ 所有层保持4:1比例")
    else:
        logger.log(f"  ❌ 存在不一致")
        logger.log(f"  发现的比例: {sorted(unique_ratios)}")
        # 显示不一致的层
        inconsistent_layers = [i for i, r in enumerate(gqa_ratios) if r != 4]
        if len(inconsistent_layers) <= 5:
            logger.log(f"  不一致的层: {inconsistent_layers}")
        else:
            logger.log(f"  不一致的层数量: {len(inconsistent_layers)}")

    # ==================== 步骤8: 评估剪枝后PPL ====================
    ppl_before_finetune = None
    if args.test_after_prune:
        logger.log("\n" + "=" * 60)
        logger.log("步骤8: 评估剪枝后困惑度")
        logger.log("=" * 60)

        model.to(args.device)
        model.eval()

        logger.log(f"使用数据集: wikitext2, seq_len={args.eval_seq_len}")
        ppl_before_finetune = PPLMetric(model, tokenizer, ['wikitext2'],
                       seq_len=args.eval_seq_len, device=args.device)
        logger.log(f"\n剪枝后 PPL: {ppl_before_finetune}")
    else:
        logger.log("\n⚠️ 未启用 --test_after_prune，跳过PPL评估")

    # ==================== 步骤9: 微调剪枝后的模型 ====================
    finetune_stats = None
    if args.finetune:
        logger.log("\n" + "=" * 60)
        logger.log("步骤9: 微调剪枝后的模型")
        logger.log("=" * 60)

        # 创建微调器
        use_lora = (args.finetune_method == 'lora')
        finetuner = FineTuner(
            model,
            tokenizer,
            device=args.device,
            logger=logger,
            use_lora=use_lora,
            lora_r=args.lora_r if use_lora else 8,
            lora_alpha=args.lora_alpha if use_lora else 16,
            lora_dropout=args.lora_dropout if use_lora else 0.05,
            lora_target_attention=args.lora_target_attention if use_lora else True,
            lora_target_mlp=args.lora_target_mlp if use_lora else True
        )

        # 执行微调
        finetune_stats = finetuner.finetune(
            dataset_name='wikitext',
            num_samples=args.finetune_samples,
            seq_len=args.finetune_seq_len,
            lr=args.finetune_lr,
            epochs=args.finetune_epochs,
            batch_size=args.finetune_batch_size,
            gradient_accumulation_steps=args.finetune_grad_accum,
            max_grad_norm=args.finetune_max_grad_norm,
            warmup_steps=args.finetune_warmup_steps,
            weight_decay=args.finetune_weight_decay,
            split='train'
        )

    else:
        logger.log("\n⚠️ 未启用 --finetune，跳过微调")

    # ==================== 步骤10: 保存微调后的模型 ====================
    if args.finetune and args.save_model:
        logger.log("\n" + "=" * 60)
        logger.log("步骤10: 保存微调后的模型")
        logger.log("=" * 60)

        finetuned_path = logger.best_checkpoint_path.replace('.bin', '_finetuned.bin')

        # 使用 FineTuner 的保存方法
        extra_info = args.__dict__.copy()
        extra_info.update({
            'attention_pruning_rates': attn_layer_pruning_rates,
            'mlp_pruning_rates': mlp_layer_pruning_rates,
            'pruning_distribution': args.pruning_distribution,
            'attention_pruned_params': attn_pruned_params,
            'mlp_pruned_params': mlp_pruned_params
        })

        finetuner.save_finetuned_model(
            save_path=finetuned_path,
            layer_pruning_rates={'attention': attn_layer_pruning_rates, 'mlp': mlp_layer_pruning_rates},
            layer_importance=layer_importance,
            finetune_stats=finetune_stats,
            extra_info=extra_info
        )

    # ==================== 步骤11: 评估微调后PPL ====================
    if args.finetune and args.test_after_prune:
        logger.log("\n" + "=" * 60)
        logger.log("步骤11: 评估微调后困惑度")
        logger.log("=" * 60)

        model.to(args.device)
        model.eval()

        logger.log(f"使用数据集: wikitext2, seq_len={args.eval_seq_len}")
        ppl_after_finetune = PPLMetric(model, tokenizer, ['wikitext2'],
                                       seq_len=args.eval_seq_len, device=args.device)
        logger.log(f"\n微调后 PPL: {ppl_after_finetune}")

        # 对比剪枝前后和微调前后的变化
        logger.log("\n" + "=" * 60)
        logger.log("性能对比总结")
        logger.log("=" * 60)

        # 显示原模型PPL（如果测量了）
        if ppl_original:
            logger.log(f"原始模型: {ppl_original}")

        # 显示剪枝后和微调后的对比
        if ppl_before_finetune:
            logger.log(f"剪枝后（微调前）: {ppl_before_finetune}")
            logger.log(f"微调后: {ppl_after_finetune}")

            # 计算改善百分比
            wikitext_key = 'wikitext2 (wikitext-2-raw-v1)'
            if wikitext_key in ppl_after_finetune:
                after_val = ppl_after_finetune[wikitext_key]

                # 相对于剪枝后的改善
                if wikitext_key in ppl_before_finetune:
                    before_val = ppl_before_finetune[wikitext_key]
                    improvement = (before_val - after_val) / before_val * 100
                    logger.log(f"微调改善（vs剪枝后）: {improvement:.2f}%")

                # 相对于原模型的退化（如果有原模型数据）
                if ppl_original and wikitext_key in ppl_original:
                    original_val = ppl_original[wikitext_key]
                    degradation = (after_val - original_val) / original_val * 100
                    logger.log(f"相对原模型退化: {degradation:+.2f}%")

    logger.log("\n" + "=" * 60)
    logger.log("✅ 完整流程完成！")
    logger.log("=" * 60)


if __name__ == "__main__":
    main()
