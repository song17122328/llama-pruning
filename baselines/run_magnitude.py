#!/usr/bin/env python3
"""
Magnitude Baseline
基于权重绝对值的剪枝方法

Magnitude 是最简单的剪枝方法：
- 不需要计算梯度
- 不需要收集激活值
- 只使用权重的绝对值作为重要性指标

使用方法：
python baselines/run_magnitude.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --pruning_ratio 0.2
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Magnitude Baseline - 基于权重绝对值的剪枝')

    # 必需参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--pruning_ratio', type=float, required=True,
                       help='目标剪枝率（例如: 0.2 表示20%）')
    parser.add_argument('--output_name', type=str, default=None,
                       help='输出目录名称（默认: Magnitude_{pruning_ratio}）')


    # 其他
    parser.add_argument('--device', type=str, default=None,
                       help='设备（默认: 自动选择）')

    args = parser.parse_args()

    # 设置默认输出名称
    if args.output_name is None:
        ratio_percent = int(args.pruning_ratio * 10000)
        args.output_name = f"Magnitude_{ratio_percent}"

    print(f"\n{'='*80}")
    print(f"Magnitude Baseline")
    print(f"{'='*80}")
    print(f"方法: 基于权重绝对值的剪枝")
    print(f"模型: {args.base_model}")
    print(f"剪枝率: {args.pruning_ratio:.1%}")
    print(f"输出: results/{args.output_name}/")
    print(f"{'='*80}\n")

    # 构建命令
    cmd = [
        "python", "run_global_pruning.py",
        "--base_model", args.base_model,
        "--output_name", args.output_name,
        "--pruning_ratio", str(args.pruning_ratio),
        "--importance_method", "magnitude",
    ]

    # 添加评估参数
    if args.run_evaluation:
        cmd.extend(["--run_evaluation", args.eval_metrics])

    # 添加微调参数
    if args.finetune:
        cmd.append("--finetune")

    # 添加设备参数
    if args.device:
        cmd.extend(["--device", args.device])

    # 打印命令
    print("执行命令:")
    print(" ".join(cmd))
    print()

    # 运行剪枝
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"✓ Magnitude baseline 完成！")
        print(f"  结果目录: results/{args.output_name}/")
        print(f"{'='*80}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 运行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
