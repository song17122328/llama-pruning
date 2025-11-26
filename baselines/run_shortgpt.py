#!/usr/bin/env python3
"""
ShortGPT Baseline
基于 Block Influence (BI) 的层移除方法

论文：ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
方法：计算每层的 Block Influence（输入输出相似度），移除最不重要的层

核心思想：
- 如果一层的输入和输出非常相似，说明该层的变换作用很小
- 相似度高 → 重要性低 → 可以被剪枝

使用方法：
    python baselines/run_shortgpt.py \
        --base_model /newdata/LLMs/Llama-3-8B-Instruct \
        --n_remove_layers 8 \
        --output_name ShortGPT_remove_8
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.shortgpt_utils import (
    compute_layer_importances_bi,
    select_layers_to_remove,
    remove_layers_from_model
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


def main():
    parser = argparse.ArgumentParser(description='ShortGPT Baseline - 基于 Block Influence 的层移除')

    # 必需参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--n_remove_layers', type=int, required=True,
                       help='要移除的层数')
    parser.add_argument('--output_name', type=str, default=None,
                       help='输出目录名称（默认: ShortGPT_remove_{n_remove_layers}）')

    # BI 计算参数
    parser.add_argument('--dataset', type=str, default='wikitext2',
                       choices=['wikitext2', 'ptb', 'c4'],
                       help='数据集选择（默认: wikitext2）')
    parser.add_argument('--num_samples', type=int, default=128,
                       help='BI 计算样本数（默认: 128，与 H-GSP 一致）')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='序列长度（默认: 128，与 H-GSP 一致）')
    parser.add_argument('--stride', type=int, default=128,
                       help='滑动窗口步长（默认: 128，与 seq_len 一致）')

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
        args.output_name = f"ShortGPT_remove_{args.n_remove_layers}"

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
    logger.log(f"ShortGPT Baseline - Block Influence (BI) 层移除")
    logger.log(f"{'='*80}")
    logger.log(f"方法: 基于输入输出相似度的层重要性计算")
    logger.log(f"模型: {args.base_model}")
    logger.log(f"移除层数: {args.n_remove_layers}")
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

    original_num_layers = len(model.model.layers)
    total_params = sum(p.numel() for p in model.parameters())

    logger.log(f"✓ 模型加载完成")
    logger.log(f"  总层数: {original_num_layers}")
    logger.log(f"  总参数量: {total_params:,}")

    # 分析原始模型
    logger.log(f"\n分析原始模型结构...")
    original_analyzer = ModelAnalyzer(model, "原始模型")
    original_analysis = original_analyzer.analyze()
    logger.log(f"  ✓ 原始模型分析完成")

    # ========== Step 2: 计算 Block Influence 重要性 ==========
    logger.log(f"\n[Step 2] 计算 Block Influence (BI) 重要性...")
    logger.log(f"  数据集: {args.dataset}")
    logger.log(f"  样本数: {args.num_samples}")
    logger.log(f"  序列长度: {args.seq_len}")
    logger.log(f"  Stride: {args.stride}")

    # 创建数据集管理器
    dataset_manager = DatasetManager(dataset_name=args.dataset, tokenizer=tokenizer)

    # 加载文本样本
    texts = dataset_manager.get_layer_importance_samples(
        num_samples=args.num_samples,
        seq_len=args.seq_len
    )

    # 计算每层的 BI 重要性
    importances = compute_layer_importances_bi(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        stride=args.stride,
        device=args.device,
        verbose=True
    )

    # 保存重要性分数
    importance_dict = {i: float(imp) for i, imp in enumerate(importances)}
    importance_path = os.path.join(output_dirs['analysis'], 'layer_bi_importance.json')
    with open(importance_path, 'w') as f:
        json.dump(importance_dict, f, indent=2)
    logger.log(f"✓ BI 重要性已保存: {importance_path}")

    # 打印每层的重要性
    logger.log(f"\n{'='*60}")
    logger.log(f"Block Influence (BI) 重要性")
    logger.log(f"{'='*60}")
    for layer_idx, imp in enumerate(importances):
        logger.log(f"Layer {layer_idx:2d}   {imp:12.4f}")
    logger.log(f"{'='*60}")

    # ========== Step 3: 选择要移除的层 ==========
    logger.log(f"\n[Step 3] 选择要移除的层...")

    layers_to_remove = select_layers_to_remove(
        importances=importances,
        n_remove=args.n_remove_layers,
        verbose=True
    )

    # 保存移除列表
    remove_path = os.path.join(output_dirs['analysis'], 'layers_to_remove.json')
    with open(remove_path, 'w') as f:
        json.dump({
            'layers_to_remove': layers_to_remove,
            'n_remove': args.n_remove_layers,
            'original_num_layers': original_num_layers
        }, f, indent=2)
    logger.log(f"✓ 移除列表已保存: {remove_path}")

    # ========== Step 4: 移除层 ==========
    logger.log(f"\n[Step 4] 移除层...")

    original_num, new_num = remove_layers_from_model(
        model=model,
        layers_to_remove=layers_to_remove,
        verbose=True
    )

    # 统计参数变化
    after_params = sum(p.numel() for p in model.parameters())
    actual_ratio = (total_params - after_params) / total_params

    logger.log(f"\n{'='*60}")
    logger.log(f"剪枝统计")
    logger.log(f"{'='*60}")
    logger.log(f"层数统计:")
    logger.log(f"  剪枝前: {original_num}")
    logger.log(f"  剪枝后: {new_num}")
    logger.log(f"  移除率: {args.n_remove_layers / original_num:.1%}")
    logger.log(f"\n参数统计:")
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
        pruned_name="ShortGPT剪枝后"
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
        'method': 'ShortGPT',
        'layers_removed': layers_to_remove,
        'n_remove': args.n_remove_layers,
        'original_num_layers': original_num_layers,
        'new_num_layers': new_num,
        'actual_ratio': actual_ratio,
        'bi_importances': importances.tolist(),
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
    logger.log(f"✓ ShortGPT baseline 完成！")
    logger.log(f"{'='*80}")
    logger.log(f"\n输出目录: {output_dirs['base']}")
    logger.log(f"  - 模型: {output_dirs['models']}")
    logger.log(f"  - 分析结果: {output_dirs['analysis']}")
    logger.log(f"  - 评估结果: {output_dirs['evaluation']}")
    logger.log(f"  - 日志: {output_dirs['logs']}")
    logger.log(f"{'='*80}\n")


if __name__ == '__main__':
    main()
