#!/usr/bin/env python3
"""
参数网格搜索脚本（完整流程版本）

自动搜索不同的校准参数组合，对每个组合运行完整的流程：
剪枝 → 评估 → 微调 → 评估

搜索空间：
- taylor_seq_len: [32, 64, 128, 256]
- block_importance_seq_len = layer_importance_seq_len: [32, 64, 128, 256]
- taylor_num_samples = layer_importance_num_samples = block_importance_num_samples: 256 (固定)
- gradient_batch_size: 8 (固定)

使用方法：
    python run_parameter_search.py \
        --model Llama \
        --output_prefix grid_search \
        --gpu 0

参数说明：
    --model: 模型名称 (Llama, Llama-Instruct, Mistral, Mistral-Instruct, Qwen, Qwen-Instruct)
    --output_prefix: 输出目录前缀
    --gpu: 使用的 GPU ID（默认: 自动选择）
    --skip-completed: 跳过已完成的实验（检查微调后评估结果）
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys
import logging
from datetime import datetime
import json
import pandas as pd
from itertools import product

# 模型路径映射
MODEL_PATHS = {
    'Llama': '/newdata/LLMs/Llama-3-8B',
    'Llama-Instruct': '/newdata/LLMs/Llama-3-8B-Instruct',
    'Mistral': '/newdata/LLMs/Mistral-7B-v0.3',
    'Mistral-Instruct': '/newdata/LLMs/Mistral-7B-Instruct-v0.3',
    'Qwen': '/newdata/LLMs/Qwen2.5-7B',
    'Qwen-Instruct': '/newdata/LLMs/Qwen2.5-7B-Instruct'
}

# 参数搜索空间
TAYLOR_SEQ_LENS = [32, 64, 128, 256]
BLOCK_SEQ_LENS = [32, 64, 128, 256]
NUM_SAMPLES = 256  # 固定值
GRADIENT_BATCH_SIZE = 8  # 固定值


def setup_logging(log_file=None):
    """设置日志系统"""
    logger = logging.getLogger('parameter_search')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 终端输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_best_gpu():
    """获取剩余显存最大的GPU"""
    try:
        from core.utils.get_best_gpu import get_best_gpu as get_gpu
        return get_gpu()
    except:
        return 0


def run_complete_pipeline(model, taylor_seq_len, block_seq_len, output_prefix, gpu_id, logger):
    """
    运行完整流程：剪枝 → 评估 → 微调 → 评估

    Args:
        model: 模型名称
        taylor_seq_len: Taylor 重要性计算的序列长度
        block_seq_len: Block/Layer 重要性计算的序列长度
        output_prefix: 输出目录前缀
        gpu_id: GPU ID
        logger: 日志对象
    """
    log = logger.info

    log(f"[GPU {gpu_id}] 开始完整流程: taylor_seq_len={taylor_seq_len}, block_seq_len={block_seq_len}")

    # 设置环境变量限制可见 GPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 构建命令（调用 run_complete_pipeline.py）
    cmd = [
        'python', 'run_complete_pipeline.py',
        '--model', model,
        '--x1', str(NUM_SAMPLES),          # taylor_num_samples
        '--x2', str(taylor_seq_len),       # taylor_seq_len
        '--y1', str(NUM_SAMPLES),          # layer/block_importance_num_samples
        '--y2', str(block_seq_len),        # layer/block_importance_seq_len
        '--z', str(GRADIENT_BATCH_SIZE),   # gradient_batch_size
        '--output_prefix', output_prefix,
        '--skip-completed'
    ]

    try:
        log(f"[GPU {gpu_id}] 执行命令: {' '.join(cmd)}")
        log(f"[GPU {gpu_id}] {'='*60}")

        # 完整流程可能需要很长时间（剪枝 + 评估 + 微调 + 评估）
        # 估计时间: 剪枝2h + 评估1h + 微调4h + 评估1h = 8h
        # 不捕获输出，直接显示到终端以便实时查看进度
        result = subprocess.run(cmd, env=env, timeout=28800)  # 8小时超时

        log(f"[GPU {gpu_id}] {'='*60}")
        if result.returncode == 0:
            log(f"[GPU {gpu_id}] ✓ 完整流程完成")
            return True
        else:
            log(f"[GPU {gpu_id}] ✗ 流程失败（返回码: {result.returncode}）")
            return False

    except subprocess.TimeoutExpired:
        log(f"[GPU {gpu_id}] ✗ 流程超时（超过8小时）")
        return False
    except Exception as e:
        log(f"[GPU {gpu_id}] ✗ 流程异常: {e}")
        return False


def collect_results(output_prefix, model, logger):
    """
    收集所有实验结果并生成汇总报告

    读取每个配置的 Excel 汇总文件

    Args:
        output_prefix: 输出目录前缀
        model: 模型名称
        logger: 日志对象
    """
    log = logger.info
    log("="*80)
    log("收集实验结果")
    log("="*80)

    results = []

    for taylor_len in TAYLOR_SEQ_LENS:
        for block_len in BLOCK_SEQ_LENS:
            config_name = f"taylor_{taylor_len}_block_{block_len}"
            config_prefix = f"{output_prefix}_{config_name}"

            # 读取该配置的 Excel 汇总文件
            excel_file = Path('results') / config_prefix / model / f"{model}_results.xlsx"

            if excel_file.exists():
                try:
                    # 读取 Excel 文件
                    df = pd.read_excel(excel_file)

                    # 计算平均值
                    # Excel 文件包含所有方法的微调前后结果
                    config_results = {
                        'taylor_seq_len': taylor_len,
                        'block_seq_len': block_len,
                        'config': config_name
                    }

                    # 从 Excel 中提取微调后的 PPL 和 Accuracy
                    if 'ppl_after' in df.columns and 'acc_after' in df.columns:
                        avg_ppl_after = df['ppl_after'].mean()
                        avg_acc_after = df['acc_after'].mean()

                        config_results['avg_ppl'] = avg_ppl_after
                        config_results['avg_acc'] = avg_acc_after

                        # 同时记录每个方法的结果
                        for _, row in df.iterrows():
                            method = row.get('method', '')
                            config_results[f'{method}_ppl'] = row.get('ppl_after', float('inf'))
                            config_results[f'{method}_acc'] = row.get('acc_after', 0.0)

                        results.append(config_results)
                        log(f"✓ {config_name}: Avg PPL={avg_ppl_after:.2f}, Avg Acc={avg_acc_after:.4f}")
                    else:
                        log(f"⊗ Excel 格式错误 {config_name}")

                except Exception as e:
                    log(f"✗ 读取失败 {config_name}: {e}")
            else:
                log(f"⊗ 未完成 {config_name}")

    if not results:
        log("⚠ 没有找到任何结果")
        return

    # 创建 DataFrame
    df = pd.DataFrame(results)

    # 排序
    df_sorted_ppl = df.sort_values('avg_ppl')
    df_sorted_acc = df.sort_values('avg_acc', ascending=False)

    # 保存结果
    output_dir = Path('results') / f"{output_prefix}_summary" / model
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 CSV
    csv_file = output_dir / "grid_search_results.csv"
    df_sorted_ppl.to_csv(csv_file, index=False)
    log(f"\n✓ 结果已保存到: {csv_file}")

    # 保存最佳配置
    best_ppl = df_sorted_ppl.iloc[0]
    best_acc = df_sorted_acc.iloc[0]

    best_configs = {
        'best_ppl': {
            'config': best_ppl['config'],
            'taylor_seq_len': int(best_ppl['taylor_seq_len']),
            'block_seq_len': int(best_ppl['block_seq_len']),
            'avg_ppl': float(best_ppl['avg_ppl']),
            'avg_acc': float(best_ppl['avg_acc'])
        },
        'best_acc': {
            'config': best_acc['config'],
            'taylor_seq_len': int(best_acc['taylor_seq_len']),
            'block_seq_len': int(best_acc['block_seq_len']),
            'avg_ppl': float(best_acc['avg_ppl']),
            'avg_acc': float(best_acc['avg_acc'])
        }
    }

    json_file = output_dir / "best_configs.json"
    with open(json_file, 'w') as f:
        json.dump(best_configs, f, indent=2)
    log(f"✓ 最佳配置已保存到: {json_file}")

    # 打印摘要
    log("\n" + "="*80)
    log("搜索结果摘要")
    log("="*80)
    log(f"\n最佳 PPL 配置（跨所有方法平均）:")
    log(f"  - Config: {best_ppl['config']}")
    log(f"  - Taylor Seq Len: {best_ppl['taylor_seq_len']}")
    log(f"  - Block Seq Len: {best_ppl['block_seq_len']}")
    log(f"  - Avg PPL: {best_ppl['avg_ppl']:.2f}")
    log(f"  - Avg Acc: {best_ppl['avg_acc']:.4f}")

    log(f"\n最佳 Accuracy 配置（跨所有方法平均）:")
    log(f"  - Config: {best_acc['config']}")
    log(f"  - Taylor Seq Len: {best_acc['taylor_seq_len']}")
    log(f"  - Block Seq Len: {best_acc['block_seq_len']}")
    log(f"  - Avg PPL: {best_acc['avg_ppl']:.2f}")
    log(f"  - Avg Acc: {best_acc['avg_acc']:.4f}")

    log("\n所有配置（按平均 PPL 排序）:")
    log(df_sorted_ppl[['config', 'taylor_seq_len', 'block_seq_len', 'avg_ppl', 'avg_acc']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='参数网格搜索')

    parser.add_argument('--model', type=str, required=True,
                       choices=list(MODEL_PATHS.keys()),
                       help='模型名称')
    parser.add_argument('--output_prefix', type=str, default='grid_search',
                       help='输出目录前缀（默认: grid_search）')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID（默认: 自动选择）')
    parser.add_argument('--skip-completed', action='store_true',
                       help='跳过已完成的实验')

    args = parser.parse_args()

    # 设置 GPU
    if args.gpu is None:
        args.gpu = get_best_gpu()

    # 创建日志目录
    log_dir = Path('results') / f"{args.output_prefix}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    log_file = log_dir / f'grid_search_{timestamp}.log'
    logger = setup_logging(log_file)
    log = logger.info

    log("="*80)
    log("参数网格搜索（完整流程）")
    log("="*80)
    log(f"模型: {args.model}")
    log(f"GPU: {args.gpu}")
    log(f"输出前缀: {args.output_prefix}")
    log(f"日志文件: {log_file}")
    log(f"跳过已完成: {args.skip_completed}")
    log("")
    log("搜索空间:")
    log(f"  - taylor_seq_len: {TAYLOR_SEQ_LENS}")
    log(f"  - block_seq_len: {BLOCK_SEQ_LENS}")
    log(f"  - num_samples: {NUM_SAMPLES} (固定)")
    log(f"  - gradient_batch_size: {GRADIENT_BATCH_SIZE} (固定)")
    log(f"  - 总实验数: {len(TAYLOR_SEQ_LENS) * len(BLOCK_SEQ_LENS)}")
    log("")
    log("每个实验包含完整流程：")
    log("  1. 剪枝（6种方法）")
    log("  2. 评估剪枝后模型")
    log("  3. LoRA 微调")
    log("  4. 评估微调后模型")
    log("  5. 导出 Excel 汇总")
    log("="*80)

    # 生成所有参数组合
    param_combinations = list(product(TAYLOR_SEQ_LENS, BLOCK_SEQ_LENS))
    total = len(param_combinations)

    completed = 0
    failed = 0

    for idx, (taylor_len, block_len) in enumerate(param_combinations, 1):
        config_name = f"taylor_{taylor_len}_block_{block_len}"
        output_prefix = f"{args.output_prefix}_{config_name}"

        log(f"\n[{idx}/{total}] 实验: {config_name}")

        # 检查是否已完成（检查 Excel 输出文件）
        excel_file = Path('results') / output_prefix / args.model / f"{args.model}_results.xlsx"
        if args.skip_completed and excel_file.exists():
            log(f"⊗ 跳过（已完成）")
            completed += 1
            continue

        # 运行完整流程（剪枝 → 评估 → 微调 → 评估）
        success = run_complete_pipeline(
            model=args.model,
            taylor_seq_len=taylor_len,
            block_seq_len=block_len,
            output_prefix=output_prefix,
            gpu_id=args.gpu,
            logger=logger
        )

        if success:
            completed += 1
        else:
            failed += 1

    # 收集并分析结果
    log("\n" + "="*80)
    log(f"搜索完成: {completed} 成功, {failed} 失败")
    log("="*80)

    collect_results(args.output_prefix, args.model, logger)

    log("\n✓ 参数搜索全部完成！")
    log(f"汇总结果目录: results/{args.output_prefix}_summary/{args.model}/")
    log(f"日志文件: {log_file}")


if __name__ == '__main__':
    main()
