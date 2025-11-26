#!/usr/bin/env python3
"""
测试模型参数统计功能

展示如何使用增强的 get_model_info() 和 print_model_info() 函数
来详细查看模型的参数分布和验证完整性。

使用方法：
    # 测试原始模型
    python evaluation/test_model_info.py /newdata/LLMs/Llama-3-8B-Instruct

    # 测试剪枝模型
    python evaluation/test_model_info.py results/Wanda_2000/pruned_model.bin

    # 测试 SliceGPT 模型
    python evaluation/test_model_info.py results/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
        --slicegpt-base-model /newdata/LLMs/Llama-3-8B-Instruct
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.utils.model_loader import load_model_and_tokenizer, get_model_info, print_model_info


def main():
    parser = argparse.ArgumentParser(description='测试模型参数统计功能')
    parser.add_argument('model_path', type=str, help='模型路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备（建议使用 cpu 避免占用 GPU）')
    parser.add_argument('--slicegpt-base-model', type=str, default=None,
                       help='SliceGPT 模型的基础模型路径（仅用于 .pt 模型）')
    parser.add_argument('--slicegpt-sparsity', type=float, default=None,
                       help='SliceGPT 模型的稀疏度（仅用于 .pt 模型）')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"测试模型参数统计功能")
    print(f"{'='*80}\n")

    # 加载模型
    print(f"正在加载模型: {args.model_path}")
    print(f"使用设备: {args.device}")
    print(f"")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        slicegpt_base_model=args.slicegpt_base_model,
        slicegpt_sparsity=args.slicegpt_sparsity
    )

    # 获取模型信息
    print(f"\n正在统计模型参数...")
    model_info = get_model_info(model)

    # 打印详细信息
    print_model_info(model_info, name=os.path.basename(args.model_path))

    # 打印原始 JSON（可以保存或用于其他用途）
    print(f"\n{'='*80}")
    print(f"原始统计数据 (可用于程序处理):")
    print(f"{'='*80}")

    import json
    # 只打印关键信息，避免太长
    summary = {
        'total_params': model_info['total_params'],
        'total_params_B': model_info['total_params_B'],
        'num_layers': model_info['num_layers'],
        'module_params': model_info.get('module_params'),
        'model_size_gb': model_info['model_size_gb']
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n✓ 测试完成！")


if __name__ == '__main__':
    main()
