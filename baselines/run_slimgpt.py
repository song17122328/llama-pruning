#!/usr/bin/env python3
"""
SlimGPT Baseline
基于 Hessian 的最优脑损伤（Optimal Brain Surgeon）剪枝方法

论文：SlimGPT: Layer-wise Structured Pruning for Large Language Models
方法：通过 Hessian 矩阵计算剪枝误差，选择最小误差的神经元剪枝，并补偿其他权重

核心思想：
- 计算 Hessian 矩阵的逆（使用输入特征近似）
- 计算每列（神经元/head）的剪枝误差
- 剪枝时补偿其他列的权重，最小化输出变化

使用方法：
    python baselines/run_slimgpt.py \
        --base_model /newdata/LLMs/Llama-3-8B-Instruct \
        --pruning_ratio 0.2 \
        --output_name SlimGPT_2000
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

from baselines.slimgpt_utils import (
    collect_layer_inputs,
    compute_hessian_inv,
    prune_attention_heads,
    prune_ffn_channels,
    compute_layer_pruning_ratios
)
from core.datasets import DatasetManager
from core.utils.logger import LoggerWithDepth
from core.analysis import ModelAnalyzer, ModelComparator
from evaluation.run_evaluation import evaluate_single_model


def setup_output_directories(base_dir):
    """创建输出目录结构"""
    dirs = {
        'base': base_dir,
        'models': base_dir,
        'analysis': os.path.join(base_dir, 'analysis'),
        'evaluation': os.path.join(base_dir, 'evaluation'),
        'logs': os.path.join(base_dir, 'logs'),
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def create_dataloader(dataset_manager, num_samples, seq_len, batch_size=1):
    """创建数据加载器"""
    # 获取文本样本
    texts = dataset_manager.get_layer_importance_samples(
        num_samples=num_samples,
        seq_len=seq_len
    )

    # Tokenize
    tokenizer = dataset_manager.tokenizer
    encodings = []

    for text in texts:
        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=seq_len,
            padding='max_length'
        )
        encodings.append(tokens)

    # 创建简单的数据加载器
    class SimpleDataset:
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            return self.encodings[idx]

    dataset = SimpleDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def slimgpt_prune_model(
    model,
    dataloader,
    pruning_ratio: float,
    head_dim: int = 128,
    device: str = 'cuda',
    max_samples: int = 128,
    logger=None
):
    """
    执行 SlimGPT 剪枝

    Args:
        model: LLaMA 模型
        dataloader: 校准数据加载器
        pruning_ratio: 目标剪枝率
        head_dim: attention head 维度
        device: 设备
        max_samples: 最大样本数（用于 Hessian 计算）
        logger: 日志记录器

    Returns:
        pruning_info: 剪枝信息字典
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    model = model.to(device)
    model.eval()

    num_layers = len(model.model.layers)

    log(f"\n{'='*60}")
    log(f"SlimGPT 剪枝")
    log(f"{'='*60}")
    log(f"目标剪枝率: {pruning_ratio:.1%}")
    log(f"总层数: {num_layers}")

    # 计算各层的剪枝率（对数增长策略）
    layer_ratios = compute_layer_pruning_ratios(
        num_layers=num_layers,
        target_ratio=pruning_ratio,
        strategy='log'
    )

    log(f"\n各层剪枝率（对数增长策略）:")
    for i, r in enumerate(layer_ratios):
        log(f"  Layer {i:2d}: {r:.3f}")

    pruning_info = {
        'layer_ratios': layer_ratios.tolist(),
        'layers': {}
    }

    # 逐层剪枝
    for layer_idx in range(num_layers):
        log(f"\n{'='*60}")
        log(f"处理 Layer {layer_idx} (ratio={layer_ratios[layer_idx]:.3f})")
        log(f"{'='*60}")

        layer = model.model.layers[layer_idx]
        layer_info = {}

        # Step 1: 收集层输入
        log(f"  [1/4] 收集层输入...")
        X = collect_layer_inputs(
            model, dataloader, layer_idx, device, max_samples
        )
        log(f"    ✓ 收集到 {X.shape[0]} 个 tokens")

        # Step 2: 计算 Hessian 逆
        log(f"  [2/4] 计算 Hessian 逆...")
        H_inv = compute_hessian_inv(X, damping=1e-6)
        log(f"    ✓ Hessian 形状: {H_inv.shape}")

        # Step 3: 剪枝 Attention
        log(f"  [3/4] 剪枝 Attention heads...")

        # 获取 o_proj 权重
        W_o = layer.self_attn.o_proj.weight.data.clone()
        original_shape = W_o.shape

        # 获取 head 数量
        num_heads = W_o.shape[1] // head_dim

        # 剪枝
        W_o_pruned, pruned_heads = prune_attention_heads(
            W=W_o,
            H_inv=H_inv,
            num_heads=num_heads,
            pruning_ratio=layer_ratios[layer_idx],
            head_dim=head_dim,
            device=device
        )

        # 更新权重
        layer.self_attn.o_proj.weight.data = W_o_pruned

        log(f"    ✓ 剪枝 {len(pruned_heads)}/{num_heads} 个 heads")
        log(f"      剪枝的 heads: {pruned_heads}")

        layer_info['attention'] = {
            'num_heads': num_heads,
            'pruned_heads': pruned_heads,
            'num_pruned': len(pruned_heads)
        }

        # Step 4: 剪枝 FFN
        log(f"  [4/4] 剪枝 FFN 通道...")

        # 获取 down_proj 权重
        W_down = layer.mlp.down_proj.weight.data.clone()
        intermediate_size = W_down.shape[1]

        # 剪枝
        W_down_pruned, pruned_channels = prune_ffn_channels(
            W=W_down,
            H_inv=H_inv,
            pruning_ratio=layer_ratios[layer_idx]
        )

        # 更新权重
        layer.mlp.down_proj.weight.data = W_down_pruned

        log(f"    ✓ 剪枝 {len(pruned_channels)}/{intermediate_size} 个通道")

        layer_info['mlp'] = {
            'intermediate_size': intermediate_size,
            'num_pruned': len(pruned_channels)
        }

        pruning_info['layers'][layer_idx] = layer_info

        # 清理显存
        del X, H_inv, W_o, W_down
        torch.cuda.empty_cache()

    log(f"\n{'='*60}")
    log(f"✓ SlimGPT 剪枝完成")
    log(f"{'='*60}")

    return pruning_info


def main():
    parser = argparse.ArgumentParser(description='SlimGPT Baseline - 基于 Hessian 的最优脑损伤剪枝')

    # 必需参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--pruning_ratio', type=float, required=True,
                       help='目标剪枝率（例如: 0.2 表示20%）')
    parser.add_argument('--output_name', type=str, default=None,
                       help='输出目录名称（默认: SlimGPT_{pruning_ratio}）')

    # Hessian 计算参数
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'ptb', 'c4'],
                       help='数据集选择（默认: wikitext2）')
    parser.add_argument('--num_samples', type=int, default=128,
                       help='Hessian 计算样本数（默认: 128，与 H-GSP 一致）')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度（默认: 128）')
    parser.add_argument('--max_samples', type=int, default=128,
                       help='Hessian 最大 token 数（默认: 128，单位k）')

    # 模型参数
    parser.add_argument('--head_dim', type=int, default=128,
                       help='Attention head 维度（默认: 128）')

    # 评估参数
    parser.add_argument('--run_evaluation', action='store_true', default=True,
                       help='运行评估（默认: True）')
    parser.add_argument('--eval_metrics', type=str, default='ppl,zeroshot,speed,memory',
                       help='评估指标（默认: ppl,zeroshot,speed,memory）')

    # 微调参数
    parser.add_argument('--finetune', action='store_true',
                       help='剪枝后进行 LoRA 微调')

    # 其他
    from core.utils.get_best_gpu import get_best_gpu
    bestDevice = "cuda:" + str(get_best_gpu())
    parser.add_argument('--device', type=str, default=bestDevice,
                       help='设备（默认: 自动选择）')

    args = parser.parse_args()

    # 设置默认输出名称
    if args.output_name is None:
        ratio_percent = int(args.pruning_ratio * 10000)
        args.output_name = f"SlimGPT_{ratio_percent}"

    # 设置输出目录
    output_base_dir = os.path.join('results', args.output_name)
    output_dirs = setup_output_directories(output_base_dir)

    # 设置 logger
    logger = LoggerWithDepth(
        env_name='logs',
        config=args.__dict__,
        root_dir=output_base_dir
    )

    logger.log(f"\n{'='*80}")
    logger.log(f"SlimGPT Baseline - Optimal Brain Surgeon (OBS) 剪枝")
    logger.log(f"{'='*80}")
    logger.log(f"方法: 基于 Hessian 的最优脑损伤剪枝")
    logger.log(f"模型: {args.base_model}")
    logger.log(f"剪枝率: {args.pruning_ratio:.1%}")
    logger.log(f"输出: {output_base_dir}")
    logger.log(f"{'='*80}\n")

    # ========== Step 1: 加载模型 ==========
    logger.log(f"[Step 1] 加载模型...")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=args.device,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    total_params = sum(p.numel() for p in model.parameters())

    logger.log(f"✓ 模型加载完成")
    logger.log(f"  总参数量: {total_params:,}")

    # 分析原始模型
    logger.log(f"\n分析原始模型结构...")
    original_analyzer = ModelAnalyzer(model, "原始模型")
    original_analysis = original_analyzer.analyze()
    logger.log(f"  ✓ 原始模型分析完成")

    # ========== Step 2: 准备数据 ==========
    logger.log(f"\n[Step 2] 准备校准数据...")
    logger.log(f"  数据集: {args.dataset}")
    logger.log(f"  样本数: {args.num_samples}")
    logger.log(f"  序列长度: {args.seq_len}")

    # 创建数据集管理器
    dataset_manager = DatasetManager(dataset_name=args.dataset, tokenizer=tokenizer)

    # 创建数据加载器
    dataloader = create_dataloader(
        dataset_manager=dataset_manager,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        batch_size=1
    )

    logger.log(f"✓ 数据加载器创建完成")

    # ========== Step 3: SlimGPT 剪枝 ==========
    logger.log(f"\n[Step 3] 执行 SlimGPT 剪枝...")

    pruning_info = slimgpt_prune_model(
        model=model,
        dataloader=dataloader,
        pruning_ratio=args.pruning_ratio,
        head_dim=args.head_dim,
        device=args.device,
        max_samples=args.max_samples,
        logger=logger
    )

    # 保存剪枝信息
    pruning_info_path = os.path.join(output_dirs['analysis'], 'slimgpt_pruning_info.json')
    with open(pruning_info_path, 'w') as f:
        json.dump(pruning_info, f, indent=2)
    logger.log(f"✓ 剪枝信息已保存: {pruning_info_path}")

    # ========== Step 4: 统计剪枝结果 ==========
    logger.log(f"\n[Step 4] 统计剪枝结果...")

    after_params = sum(p.numel() for p in model.parameters())
    actual_ratio = (total_params - after_params) / total_params

    logger.log(f"\n{'='*60}")
    logger.log(f"剪枝统计")
    logger.log(f"{'='*60}")
    logger.log(f"参数统计:")
    logger.log(f"  剪枝前: {total_params:,}")
    logger.log(f"  剪枝后: {after_params:,}")
    logger.log(f"  实际剪枝率: {actual_ratio:.2%}")
    logger.log(f"{'='*60}")

    # 分析剪枝后的模型
    logger.log(f"\n分析剪枝后模型结构...")
    pruned_analyzer = ModelAnalyzer(model, "剪枝后模型")
    pruned_analysis = pruned_analyzer.analyze()
    logger.log(f"  ✓ 剪枝后模型分析完成")

    # 生成对比报告
    comparator = ModelComparator(
        original_analysis=original_analysis,
        pruned_analysis=pruned_analysis,
        original_name="原始模型",
        pruned_name="SlimGPT剪枝后"
    )
    comparison_result = comparator.compare()

    # 保存分析报告
    original_analysis_path = os.path.join(output_dirs['analysis'], 'original_model_analysis.json')
    with open(original_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(original_analysis, f, indent=2, ensure_ascii=False)

    pruned_analysis_path = os.path.join(output_dirs['analysis'], 'pruned_model_analysis.json')
    with open(pruned_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(pruned_analysis, f, indent=2, ensure_ascii=False)

    comparison_path = os.path.join(output_dirs['analysis'], 'model_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)

    logger.log(f"  ✓ 分析报告已保存: {output_dirs['analysis']}/")

    # ========== Step 5: 保存模型 ==========
    logger.log(f"\n[Step 5] 保存剪枝后的模型...")
    save_path = os.path.join(output_dirs['models'], 'pruned_model.bin')

    save_dict = {
        'model': model,
        'tokenizer': tokenizer,
        'method': 'SlimGPT',
        'pruning_ratio': args.pruning_ratio,
        'actual_ratio': actual_ratio,
        'pruning_info': pruning_info,
        'config': args.__dict__
    }

    torch.save(save_dict, save_path)
    logger.log(f"✓ 模型已保存: {save_path}")
    logger.log(f"  文件大小: {os.path.getsize(save_path) / (1024**3):.2f} GB")

    # ========== Step 6: 微调（可选）==========
    if args.finetune:
        logger.log(f"\n[Step 6] LoRA 微调恢复...")

        import subprocess

        finetune_cmd = [
            "python", "finetune_lora.py",
            "--pruned_model", save_path,
            "--data_path", "yahma/alpaca-cleaned",
            "--num_epochs", "2",
            "--learning_rate", "1e-4",
            "--batch_size", "64",
            "--micro_batch_size", "4",
            "--lora_r", "8",
            "--lora_alpha", "16",
            "--device", args.device
        ]

        logger.log(f"  启动 LoRA 微调...")
        try:
            subprocess.run(finetune_cmd, check=True, capture_output=False, text=True)
            logger.log(f"✓ LoRA 微调完成")
            finetuned_output_dir = os.path.join('results', f"{args.output_name}_finetuned")
            logger.log(f"  微调后的模型保存在: {finetuned_output_dir}")
        except subprocess.CalledProcessError as e:
            logger.log(f"⚠️ LoRA 微调失败: {e}")
    else:
        logger.log(f"\n[Step 6] 跳过微调（未指定 --finetune）")

    # ========== Step 7: 评估（可选）==========
    if args.run_evaluation:
        logger.log(f"\n[Step 7] 运行评估...")

        # 清理显存
        import gc
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.log(f"  ✓ 显存已清理")

        # 解析评估类型
        eval_types = [t.strip() for t in args.eval_metrics.split(',')]
        if 'all' in eval_types:
            eval_types = ['ppl', 'zeroshot', 'speed', 'memory']

        logger.log(f"  评估类型: {', '.join(eval_types)}")

        # 运行评估
        logger.log(f"\n  开始评估...")
        eval_results = evaluate_single_model(
            model_path=save_path,
            metrics=eval_types,
            device=args.device,
            ppl_datasets=['wikitext2', 'ptb'],
            zeroshot_tasks=['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa'],
            speed_samples=50,
            verbose=True,
            use_custom_zeroshot=True,
            zeroshot_batch_size=8
        )

        # 保存评估结果
        eval_result_path = os.path.join(output_dirs['evaluation'], 'evaluation_results.json')
        with open(eval_result_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.log(f"\n✓ 评估结果已保存: {eval_result_path}")

        # 打印简要评估摘要
        logger.log(f"\n{'='*60}")
        logger.log(f"评估结果摘要")
        logger.log(f"{'='*60}")
        if 'ppl' in eval_results.get('metrics', {}):
            logger.log(f"\nPPL 结果:")
            for dataset, ppl in eval_results['metrics']['ppl'].items():
                logger.log(f"  {dataset}: {ppl:.2f}" if ppl else f"  {dataset}: N/A")

        if 'avg_zeroshot_acc' in eval_results.get('metrics', {}):
            acc = eval_results['metrics']['avg_zeroshot_acc']
            logger.log(f"\nZero-shot 平均准确率: {acc*100:.2f}%")
    else:
        logger.log(f"\n[Step 7] 跳过评估（未指定 --run_evaluation）")

    logger.log(f"\n{'='*80}")
    logger.log(f"✓ SlimGPT baseline 完成！")
    logger.log(f"{'='*80}")
    logger.log(f"\n输出目录: {output_dirs['base']}")
    logger.log(f"  - 模型: {output_dirs['models']}")
    logger.log(f"  - 分析结果: {output_dirs['analysis']}")
    logger.log(f"  - 评估结果: {output_dirs['evaluation']}")
    logger.log(f"  - 日志: {output_dirs['logs']}")
    logger.log(f"{'='*80}\n")


if __name__ == '__main__':
    main()
