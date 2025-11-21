#!/usr/bin/env python3
"""
性能指标评估

包括:
1. PPL (Perplexity) - 多数据集
2. Zero-shot准确率 - 常识推理、阅读理解等
3. Few-shot准确率 - MMLU等（可选）
"""

import sys
import os
import torch
from typing import Dict, List, Optional, Union

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def evaluate_ppl(
    model,
    tokenizer,
    datasets: List[str] = ['wikitext2', 'ptb', 'c4'],
    seq_len: int = 128,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    评估多个数据集上的PPL

    Args:
        model: 模型
        tokenizer: tokenizer
        datasets: 数据集列表，支持 'wikitext2', 'ptb', 'c4'
        seq_len: 序列长度
        device: 设备

    Returns:
        {dataset_name: ppl_value}
    """
    from evaluation.metrics.ppl import PPLMetric

    print(f"\n{'='*60}")
    print(f"评估 PPL (seq_len={seq_len})")
    print(f"{'='*60}")

    # 直接使用 PPLMetric 评估所有数据集（简化调用）
    ppl_metric = PPLMetric(
        model=model,
        tokenizer=tokenizer,
        datasets=datasets,
        seq_len=seq_len,
        device=device
    )

    return ppl_metric.results


def evaluate_zeroshot(
    model_path: str,
    tasks: List[str] = None,
    batch_size: int = 8,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    使用lm-evaluation-harness评估Zero-shot任务（在线加载数据集）

    Args:
        model_path: 模型路径（支持HF目录或.bin checkpoint）
        tasks: 任务列表，默认为常用的7个任务
        batch_size: 批次大小
        device: 设备

    Returns:
        评估结果字典

    注意：
        - 使用 lm-eval 标准任务，在线加载数据集
        - 需要安装 lm-eval: pip install lm-eval
        - .bin文件会自动使用自定义加载器
    """
    # 使用标准 lm-eval 任务名称（在线加载）
    if tasks is None:
        tasks = [
            'boolq',           # 是非问答
            'piqa',            # 物理常识
            'hellaswag',       # 常识推理
            'winogrande',      # 代词消歧
            'arc_easy',        # 科学问答（简单）
            'arc_challenge',   # 科学问答（困难）
            'openbookqa'       # 科学推理
        ]

    print(f"\n{'='*60}")
    print(f"评估 Zero-shot 任务 (lm-eval 在线模式)")
    print(f"{'='*60}")
    print(f"任务: {', '.join(tasks)}\n")

    try:
        # 导入 lm_eval
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        print("开始评估（从 HuggingFace 在线加载数据集）...")
        print("首次运行可能需要下载数据集，请耐心等待...\n")

        # 检查是否是checkpoint文件
        if model_path.endswith('.bin'):
            print("检测到.bin格式，使用自定义加载器...")
            from evaluation.utils.model_loader import load_model_and_tokenizer

            # 加载剪枝后的模型
            model, tokenizer = load_model_and_tokenizer(
                model_path,
                device=device,
                force_single_device=True
            )

            # 使用HFLM包装预加载的模型
            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size
            )

            # 使用包装后的模型进行评估
            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=tasks,
                log_samples=False,
                verbosity="INFO"  # 显示更多信息
            )
        else:
            # HF格式，直接使用路径
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_path},dtype=float16,device={device}",
                tasks=tasks,
                batch_size=batch_size,
                log_samples=False,
                verbosity="INFO"  # 显示更多信息
            )

        # 提取关键结果
        summary = {}
        for task in tasks:
            if task in results['results']:
                task_results = results['results'][task]
                # 提取准确率（不同任务的metric名称可能不同）
                # 新版lm-eval的key格式为 'acc_norm,none' 或 'acc,none'
                acc = None
                for key, value in task_results.items():
                    if 'acc_norm' in key:
                        acc = value
                        break
                    elif 'acc' in key and acc is None:
                        acc = value

                summary[task] = {
                    'accuracy': acc,
                    'full_results': task_results
                }

                print(f"  ✓ {task}: {acc*100:.2f}%" if acc is not None else f"  ✓ {task}: N/A")

        return summary

    except ImportError:
        print("✗ lm-eval未安装")
        print("请安装: pip install lm-eval")
        return None

    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_fewshot(
    model_path: str,
    tasks: List[str] = None,
    num_fewshot: int = 5,
    batch_size: int = 8,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    评估Few-shot任务（可选）

    Args:
        model_path: 模型路径（支持HF目录或.bin checkpoint）
        tasks: 任务列表，默认为MMLU
        num_fewshot: few-shot样本数
        batch_size: 批次大小
        device: 设备

    Returns:
        评估结果字典
    """
    if tasks is None:
        tasks = ['mmlu']  # MMLU是最常用的few-shot任务

    print(f"\n{'='*60}")
    print(f"评估 {num_fewshot}-shot 任务")
    print(f"{'='*60}")
    print(f"任务: {', '.join(tasks)}")
    print("⚠️  Few-shot评估较慢，建议只在最终对比时使用\n")

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM

        # 检查是否是checkpoint文件
        if model_path.endswith('.bin'):
            print("检测到.bin格式，使用自定义加载器...")
            from evaluation.utils.model_loader import load_model_and_tokenizer

            model, tokenizer = load_model_and_tokenizer(
                model_path,
                device=device,
                force_single_device=True
            )

            lm = HFLM(
                pretrained=model,
                tokenizer=tokenizer,
                batch_size=batch_size
            )

            results = lm_eval.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot=num_fewshot,
                log_samples=False
            )
        else:
            # HF格式，直接使用路径
            results = lm_eval.simple_evaluate(
                model="hf",
                model_args=f"pretrained={model_path},dtype=float16,device={device}",
                tasks=tasks,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                log_samples=False
            )

        summary = {}
        for task in tasks:
            if task in results['results']:
                task_results = results['results'][task]
                # 新版lm-eval的key格式为 'acc,none' 等
                acc = None
                for key, value in task_results.items():
                    if 'acc' in key:
                        acc = value
                        break

                summary[task] = {
                    'accuracy': acc,
                    'full_results': task_results
                }

                print(f"  ✓ {task}: {acc*100:.2f}%" if acc is not None else f"  ✓ {task}: N/A")

        return summary

    except ImportError:
        print("✗ lm-eval未安装")
        return None

    except Exception as e:
        print(f"✗ 评估失败: {e}")
        return None


def compute_average_accuracy(zeroshot_results: Dict[str, Dict]) -> float:
    """
    计算多个任务的平均准确率

    Args:
        zeroshot_results: evaluate_zeroshot返回的结果

    Returns:
        平均准确率 (0-1)
    """
    if not zeroshot_results:
        return 0.0

    accuracies = []
    for task, results in zeroshot_results.items():
        if results and 'accuracy' in results and results['accuracy'] is not None:
            accuracies.append(results['accuracy'])

    if not accuracies:
        return 0.0

    avg_acc = sum(accuracies) / len(accuracies)
    return avg_acc


if __name__ == '__main__':
    # 测试代码
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--metrics', type=str, default='ppl,zeroshot',
                       help='逗号分隔的指标: ppl, zeroshot, fewshot')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    metrics = args.metrics.split(',')

    # 加载模型
    from evaluation.utils.model_loader import load_model_and_tokenizer

    if 'ppl' in metrics:
        model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
        ppl_results = evaluate_ppl(model, tokenizer, device=args.device)
        print(f"\nPPL结果: {ppl_results}")

    if 'zeroshot' in metrics:
        zeroshot_results = evaluate_zeroshot(args.model_path, device=args.device)
        if zeroshot_results:
            avg_acc = compute_average_accuracy(zeroshot_results)
            print(f"\n平均准确率: {avg_acc*100:.2f}%")

    if 'fewshot' in metrics:
        fewshot_results = evaluate_fewshot(args.model_path, device=args.device)
