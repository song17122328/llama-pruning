#!/usr/bin/env python3
"""
统一评估脚本

用法:
    # 评估单个模型的所有指标
    python evaluation/run_evaluation.py \
        --model_path prune_log/ours_optimal/pytorch_model.bin \
        --metrics all \
        --output results/ours.json

        
     # 评估微调模型
    python evaluation/run_evaluation.py \
        --model_path results/taylor_only_2000_finetuned/pruned_model.bin \
        --metrics ppl,zeroshot,speed,memory \
        --output results/taylor_only_2000_finetuned/evaluation/evaluation_results.json 


    python evaluation/run_evaluation.py \
        --model_path results/LLM-Pruner_1937_finetuned/pruned_model.bin \
        --metrics ppl,zeroshot,speed,memory \
        --output results/LLM-Pruner_1937_finetuned/evaluation/evaluation_results.json
    
    python evaluation/run_evaluation.py \
        --model_path results/layerwise_only_2000_finetuned/pruned_model.bin \
        --metrics ppl,zeroshot,speed,memory \
        --output results/layerwise_only_2000_finetuned/evaluation/evaluation_results.json
    
    python evaluation/run_evaluation.py \
        --model_path results/blockwise_only_2000_finetuned/pruned_model.bin \
        --metrics ppl,zeroshot,speed,memory \
        --output results/blockwise_only_2000_finetuned/evaluation/evaluation_results.json 
   
    python evaluation/run_evaluation.py \
        --model_path results/HGSP_2000_finetuned/pruned_model.bin \
        --metrics ppl,zeroshot,speed,memory \
        --output results/HGSP_2000_finetuned/evaluation/evaluation_results.json
    
    python evaluation/run_evaluation.py \
        --model_path /newdata/LLMs/Llama-3-8B-Instruct \
        --metrics ppl,speed \
        --output results/original.json \
        --auto_select_gpu \
        --lm_eval

# 使用2040窗口 和512步长评估PPL
python evaluation/run_evaluation.py \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics ppl \
    --ppl_seq_len 2048 \
    --ppl_stride 512 \
    --output results/Llama-3-8B-Instruct/ppl_2048_512.json

python evaluation/run_evaluation.py \
    --model_path results/HGSP_2000/pruned_model.bin \
    --metrics ppl \
    --ppl_seq_len 2048 \
    --ppl_stride 512 \
    --output results/HGSP_2000/evaluation/ppl_2048_512.json
    
python evaluation/run_evaluation.py \
    --model_path results/HGSP_2000_finetuned/pruned_model.bin \
    --metrics ppl \
    --ppl_seq_len 2048 \
    --ppl_stride 512 \
    --output results/HGSP_2000_finetuned/evaluation/ppl_2048_512.json

    # 对比多个模型
    python evaluation/run_evaluation.py \
        --compare \
        --model_paths original.json,ours.json,baseline.json \
        --output comparison_table.md
"""

import argparse
import json
import os
import sys
from typing import Dict, List
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics.performance import (
    evaluate_ppl,
    evaluate_zeroshot,
    evaluate_fewshot,
    compute_average_accuracy
)
from evaluation.metrics.zeroshot import evaluate_zeroshot_custom
from evaluation.metrics.efficiency import evaluate_efficiency
from evaluation.utils.model_loader import (
    load_model_and_tokenizer,
    get_model_info,
    print_model_info,
    cleanup_model
)
from evaluation.utils.get_best_gpu import get_best_gpu


def evaluate_single_model(
    model_path: str,
    metrics: List[str],
    device: str = 'cuda',
    ppl_datasets: List[str] = None,
    ppl_seq_len: int = 128,
    ppl_stride: int = None,
    zeroshot_tasks: List[str] = None,
    speed_samples: int = 50,
    verbose: bool = True,
    use_custom_zeroshot: bool = True,
    zeroshot_batch_size: int = 8,
    slicegpt_base_model: str = None,
    slicegpt_sparsity: float = None
) -> Dict:
    """
    评估单个模型

    Args:
        model_path: 模型路径
        metrics: 要评估的指标列表 ['ppl', 'zeroshot', 'speed', 'memory']
        device: 设备
        ppl_datasets: PPL数据集列表
        ppl_seq_len: PPL评估窗口大小（默认128）
        ppl_stride: PPL评估步长（默认None，即等于seq_len）
        zeroshot_tasks: Zero-shot任务列表
        speed_samples: 速度测试样本数
        verbose: 是否打印详细信息
        use_custom_zeroshot: 是否使用自定义 zeroshot 评估器（不依赖 lm-eval）
        zeroshot_batch_size: Zero-shot 批处理大小（默认8，仅对自定义评估器有效）

    Returns:
        评估结果字典
    """
    results = {
        'model_path': model_path,
        'timestamp': datetime.now().isoformat(),
        'metrics': {}
    }

    print(f"\n{'='*80}")
    print(f"评估模型: {model_path}")
    print(f"{'='*80}")

    # 初始化模型变量
    model = None
    tokenizer = None

    # 根据需要的指标决定是否加载模型
    # 自定义 zeroshot 评估也需要模型
    need_model = any(m in metrics for m in ['ppl', 'speed', 'memory', 'efficiency'])
    if 'zeroshot' in metrics and use_custom_zeroshot:
        need_model = True

    # 特殊情况：SliceGPT 模型（.pt 文件）必须预加载
    # 因为 lm-eval 无法直接加载 .pt 文件
    if model_path.endswith('.pt') and 'zeroshot' in metrics:
        need_model = True

    if need_model:
        # 加载模型
        model, tokenizer = load_model_and_tokenizer(
            model_path,
            device=device,
            slicegpt_base_model=slicegpt_base_model,
            slicegpt_sparsity=slicegpt_sparsity
        )

        # 获取模型信息（所有评估都需要）
        model_info = get_model_info(model)
        results['metrics']['model_info'] = model_info

        if verbose:
            print_model_info(model_info, name="模型")

    # 1. PPL评估
    if 'ppl' in metrics:
        if ppl_datasets is None:
            ppl_datasets = ['wikitext2', 'ptb']

        # 检测是否使用默认配置（未显式指定参数）
        # 如果是默认值，自动测量两组：快速配置（128）和标准配置（2048+512）
        is_default_config = (ppl_seq_len == 128 and ppl_stride is None)

        if is_default_config:
            # 自动测量两组配置
            print(f"\n{'='*60}")
            print(f"检测到使用默认配置，将自动测量两组 PPL：")
            print(f"  1. 快速配置: seq_len=128, stride=128")
            print(f"  2. 标准配置: seq_len=2048, stride=512")
            print(f"{'='*60}")

            # 快速配置
            print(f"\n[1/2] 快速配置评估...")
            ppl_quick = evaluate_ppl(
                model, tokenizer,
                datasets=ppl_datasets,
                seq_len=128,
                stride=None,  # 等于 seq_len
                device=device
            )
            results['metrics']['ppl_quick_128'] = ppl_quick

            # 标准配置
            print(f"\n[2/2] 标准配置评估...")
            ppl_standard = evaluate_ppl(
                model, tokenizer,
                datasets=ppl_datasets,
                seq_len=2048,
                stride=512,
                device=device
            )
            results['metrics']['ppl_standard_2048_512'] = ppl_standard

            # 为了兼容性，也保存一份默认的 ppl（使用标准配置）
            results['metrics']['ppl'] = ppl_standard
        else:
            # 用户显式指定了参数，只运行一组
            print(f"\n检测到自定义配置: seq_len={ppl_seq_len}, stride={ppl_stride if ppl_stride is not None else ppl_seq_len}")
            ppl_results = evaluate_ppl(
                model, tokenizer,
                datasets=ppl_datasets,
                seq_len=ppl_seq_len,
                stride=ppl_stride,
                device=device
            )
            results['metrics']['ppl'] = ppl_results

    # 2. Zero-shot评估
    if 'zeroshot' in metrics:
        if use_custom_zeroshot:
            # 使用自定义评估器（不依赖 lm-eval，更稳定）
            # 需要已加载的模型
            if model is None:
                model, tokenizer = load_model_and_tokenizer(
                    model_path,
                    device=device,
                    force_single_device=True
                )

            # 转换任务名称（去掉 _local 后缀）
            tasks = [t.replace('_local', '') for t in zeroshot_tasks] if zeroshot_tasks else None

            zeroshot_results = evaluate_zeroshot_custom(
                model, tokenizer,
                tasks=tasks,
                device=device,
                batch_size=zeroshot_batch_size
            )
        else:
            # 使用 lm-eval，传递预加载的模型（避免重复加载）
            zeroshot_results = evaluate_zeroshot(
                model_path=model_path,
                tasks=zeroshot_tasks,
                device=device,
                model=model,
                tokenizer=tokenizer
            )

        results['metrics']['zeroshot'] = zeroshot_results

        if zeroshot_results:
            # 计算平均准确率
            accuracies = []
            for task, res in zeroshot_results.items():
                if isinstance(res, dict) and 'accuracy' in res:
                    accuracies.append(res['accuracy'])
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                results['metrics']['avg_zeroshot_acc'] = avg_acc
                print(f"\n平均Zero-shot准确率: {avg_acc*100:.2f}%")

    # 3. Few-shot评估（可选）
    if 'fewshot' in metrics:
        # evaluate_fewshot 支持 HF格式和 .bin checkpoint
        fewshot_results = evaluate_fewshot(model_path, device=device)
        results['metrics']['fewshot'] = fewshot_results

    # 4. 效率评估（速度+内存）
    if any(m in metrics for m in ['speed', 'memory', 'efficiency']):
        efficiency_results = evaluate_efficiency(
            model, tokenizer,
            device=device,
            num_samples=speed_samples,
            batch_sizes=[1, 4] if 'speed' in metrics else [1]
        )
        results['metrics']['efficiency'] = efficiency_results

    # 清理模型
    if need_model:
        cleanup_model(model, tokenizer)

    return results


def compare_models(result_files: List[str], output_path: str = None):
    """
    对比多个模型的评估结果

    Args:
        result_files: 结果文件路径列表
        output_path: 输出文件路径（markdown表格）
    """
    print(f"\n{'='*80}")
    print(f"对比模型评估结果")
    print(f"{'='*80}")

    # 加载所有结果
    all_results = {}
    for file_path in result_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
            model_name = os.path.basename(file_path).replace('.json', '')
            all_results[model_name] = results

    # 生成对比表格
    table = generate_comparison_table(all_results)

    print("\n" + table)

    # 保存到文件
    if output_path:
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"\n✓ 对比表格已保存到: {output_path}")


def generate_comparison_table(all_results: Dict[str, Dict]) -> str:
    """
    生成markdown格式的对比表格

    Args:
        all_results: {model_name: results}

    Returns:
        Markdown表格字符串
    """
    lines = []

    # 表头
    model_names = list(all_results.keys())
    lines.append("| Metric | " + " | ".join(model_names) + " |")
    lines.append("|" + "---|" * (len(model_names) + 1))

    # 参数量
    params_row = ["Parameters (B)"]
    for name in model_names:
        if 'model_info' in all_results[name]['metrics']:
            params_b = all_results[name]['metrics']['model_info']['total_params_B']
            params_row.append(f"{params_b:.2f}")
        else:
            params_row.append("N/A")
    lines.append("| " + " | ".join(params_row) + " |")

    # PPL (WikiText-2)
    if any('ppl' in r['metrics'] for r in all_results.values()):
        ppl_row = ["PPL (WikiText-2)"]
        for name in model_names:
            if 'ppl' in all_results[name]['metrics']:
                ppl_dict = all_results[name]['metrics']['ppl']
                # 查找wikitext2的结果
                wikitext_ppl = None
                for key, value in ppl_dict.items():
                    if 'wikitext' in key.lower() and value is not None:
                        wikitext_ppl = value
                        break
                ppl_row.append(f"{wikitext_ppl:.2f}" if wikitext_ppl else "N/A")
            else:
                ppl_row.append("N/A")
        lines.append("| " + " | ".join(ppl_row) + " |")

    # Zero-shot平均准确率
    if any('avg_zeroshot_acc' in r['metrics'] for r in all_results.values()):
        acc_row = ["Avg Zero-shot Acc (%)"]
        for name in model_names:
            if 'avg_zeroshot_acc' in all_results[name]['metrics']:
                acc = all_results[name]['metrics']['avg_zeroshot_acc']
                acc_row.append(f"{acc*100:.2f}")
            else:
                acc_row.append("N/A")
        lines.append("| " + " | ".join(acc_row) + " |")

    # 推理速度（batch=1）
    if any('efficiency' in r['metrics'] and 'speed' in r['metrics']['efficiency']
           for r in all_results.values()):
        speed_row = ["Throughput (tokens/s)"]
        for name in model_names:
            if ('efficiency' in all_results[name]['metrics'] and
                'speed' in all_results[name]['metrics']['efficiency'] and
                'batch_size_1' in all_results[name]['metrics']['efficiency']['speed']):
                throughput = all_results[name]['metrics']['efficiency']['speed']['batch_size_1']['throughput_tokens_per_sec']
                speed_row.append(f"{throughput:.1f}")
            else:
                speed_row.append("N/A")
        lines.append("| " + " | ".join(speed_row) + " |")

    # 显存占用
    if any('efficiency' in r['metrics'] and 'memory' in r['metrics']['efficiency']
           for r in all_results.values()):
        mem_row = ["GPU Memory (MB)"]
        for name in model_names:
            if ('efficiency' in all_results[name]['metrics'] and
                'memory' in all_results[name]['metrics']['efficiency']):
                mem = all_results[name]['metrics']['efficiency']['memory']['model_memory_mb']
                mem_row.append(f"{mem:.0f}")
            else:
                mem_row.append("N/A")
        lines.append("| " + " | ".join(mem_row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='统一评估脚本')

    # 模式选择
    parser.add_argument('--compare', action='store_true',
                       help='对比模式：对比多个已评估的模型')

    # 单模型评估参数
    parser.add_argument('--model_path', type=str,
                       help='模型路径（HF目录或checkpoint.bin）')
    parser.add_argument('--metrics', type=str, default='ppl,zeroshot,speed,memory',
                       help='评估指标（逗号分隔）: ppl, zeroshot, fewshot, speed, memory, efficiency, all')

    # 对比模式参数
    parser.add_argument('--model_paths', type=str,
                       help='多个结果文件路径（逗号分隔）')

    # 通用参数
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件路径（.json或.md）')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备: cuda/cpu 或 cuda:N 指定GPU')
    parser.add_argument('--auto_select_gpu', action='store_true',default=True,
                       help='自动选择剩余显存最多的GPU（会覆盖--device）')

    # 评估配置
    parser.add_argument('--ppl_datasets', type=str, default='wikitext2,ptb',
                       help='PPL数据集（逗号分隔）')
    parser.add_argument('--ppl_seq_len', type=int, default=128,
                       help='PPL评估窗口大小（默认: 128，标准配置: 2048）')
    parser.add_argument('--ppl_stride', type=int, default=None,
                       help='PPL评估步长（默认: None即等于seq_len，标准配置: 512）')
    parser.add_argument('--zeroshot_tasks', type=str,
                       default='boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa',
                       help='Zero-shot任务（逗号分隔）')
    parser.add_argument('--speed_samples', type=int, default=50,
                       help='速度测试样本数')
    parser.add_argument('--zeroshot_batch_size', type=int, default=8,
                       help='Zero-shot 批处理大小（默认8，仅对自定义评估器有效）')
    parser.add_argument('--use_lm_eval', action='store_true',default=True,
                       help='使用 lm-eval 在线模式（从 HuggingFace 加载数据）。默认使用自定义评估器（从本地加载数据）')

    # SliceGPT 专用参数
    parser.add_argument('--slicegpt_base_model', type=str, default=None,
                       help='SliceGPT 模型的基础模型路径（仅用于 .pt 模型，如未指定将自动推断）')
    parser.add_argument('--slicegpt_sparsity', type=float, default=None,
                       help='SliceGPT 模型的稀疏度（仅用于 .pt 模型，如未指定将从文件名推断）')

    args = parser.parse_args()

    # 自动选择GPU
    if args.auto_select_gpu:
        gpu_id = get_best_gpu()
        args.device = f'cuda:{gpu_id}'
        print(f"✓ 自动选择GPU: {args.device}\n")

    # 对比模式
    if args.compare:
        if not args.model_paths:
            print("错误: --compare模式需要提供--model_paths")
            return

        result_files = args.model_paths.split(',')
        compare_models(result_files, args.output)
        return

    # 单模型评估模式
    if not args.model_path:
        print("错误: 需要提供--model_path")
        return

    # 解析指标
    if args.metrics == 'all':
        metrics = ['ppl', 'zeroshot', 'speed', 'memory']
    else:
        metrics = args.metrics.split(',')

    # 运行评估
    results = evaluate_single_model(
        model_path=args.model_path,
        metrics=metrics,
        device=args.device,
        ppl_datasets=args.ppl_datasets.split(',') if args.ppl_datasets else None,
        ppl_seq_len=args.ppl_seq_len,
        ppl_stride=args.ppl_stride,
        zeroshot_tasks=args.zeroshot_tasks.split(',') if args.zeroshot_tasks else None,
        speed_samples=args.speed_samples,
        use_custom_zeroshot=not args.use_lm_eval,
        zeroshot_batch_size=args.zeroshot_batch_size,
        slicegpt_base_model=args.slicegpt_base_model,
        slicegpt_sparsity=args.slicegpt_sparsity
    )

    # 保存结果
    output_dir = os.path.dirname(args.output)
    if output_dir:  # 只有当有目录部分时才创建
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ 评估完成！结果已保存到: {args.output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
