#!/usr/bin/env python3
"""
Wanda (Weight and Activation) Baseline
结构化剪枝方法，结合权重和激活值

Wanda 方法特点：
- 使用 |W| × ||X||_2 作为重要性指标
- 需要收集校准数据集上的激活值（L2 Norm）
- 通过激活值识别重要的输入通道（考虑了 Outlier 的影响）
- 适用于结构化剪枝（GQA heads, FFN channels）

关键修正（本实现）：
1. 使用 L2 Norm 而非 Mean（符合 Wanda 论文）
2. 正确的 Hook 位置：down_proj 直接 Hook（包含 SwiGLU 作用）
3. 矩阵乘法优化：避免生成巨大中间矩阵

使用方法：
python baselines/run_wanda.py \
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
    parser = argparse.ArgumentParser(description='Wanda Baseline - 基于权重和激活值的结构化剪枝')

    # 必需参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--pruning_ratio', type=float, required=True,
                       help='目标剪枝率（例如: 0.2 表示20%）')
    parser.add_argument('--output_name', type=str, default=None,
                       help='输出目录名称（默认: Wanda_{pruning_ratio}）')

    args = parser.parse_args()

    # 设置默认输出名称
    if args.output_name is None:
        ratio_percent = int(args.pruning_ratio * 10000)
        args.output_name = f"Wanda_{ratio_percent}"

    print(f"\n{'='*80}")
    print(f"Wanda Baseline - Weight and Activation Structured Pruning")
    print(f"{'='*80}")
    print(f"方法: 结合权重和激活值的结构化剪枝")
    print(f"  - 重要性: |W| × ||X||_2 (L2 Norm)")
    print(f"  - Hook 位置: 正确捕获 SwiGLU 作用后的激活")
    print(f"  - 优化: 矩阵乘法，避免巨大中间矩阵")

    # 构建命令
    # 注意: run_global_pruning.py 内部固定使用 TAYLOR_NUM_SAMPLES=128
    # 不需要传递 --calibration_samples 参数
    cmd = [
        "python", "run_global_pruning.py",
        "--base_model", args.base_model,
        "--output_name", args.output_name,
        "--pruning_ratio", str(args.pruning_ratio),
        "--importance_method", "wanda",
        "--temperature", "0.0"
    ]

    # 打印命令
    print("执行命令:")
    print(" ".join(cmd))
    print()

    # 运行剪枝
    try:
        subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"✓ Wanda baseline 完成！")
        print(f"  结果目录: results/{args.output_name}/")
        print(f"{'='*80}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 运行失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
