#!/usr/bin/env python3
"""
SliceGPT 模型转换器

将 SliceGPT 的 .pt 格式转换为我们评估框架支持的 .bin 格式。

使用场景：
1. 在 slicegpt 环境中运行此脚本进行转换
2. 在 base 环境中使用转换后的模型进行评估

使用方法：
    # 在 slicegpt 环境中运行
    conda activate slicegpt
    python evaluation/convert_slicegpt_model.py \
        --slicegpt_model results/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
        --base_model /newdata/LLMs/Llama-3-8B-Instruct \
        --output results/SliceGPT_2000/pruned_model.bin

    # 然后在 base 环境中评估
    conda activate base
    python evaluation/run_evaluation.py \
        --model_path results/SliceGPT_2000/pruned_model.bin \
        --metrics all \
        --output results/SliceGPT_2000/evaluation/evaluation_results.json
"""

import argparse
import torch
import os
import sys
import pathlib

def main():
    parser = argparse.ArgumentParser(description='将 SliceGPT 模型转换为标准格式')
    parser.add_argument('--slicegpt_model', type=str, required=True,
                       help='SliceGPT 模型路径 (.pt 文件)')
    parser.add_argument('--base_model', type=str, default=None,
                       help='基础模型路径（如不指定，自动从文件名推断）')
    parser.add_argument('--sparsity', type=float, default=None,
                       help='稀疏度（如不指定，自动从文件名推断）')
    parser.add_argument('--output', type=str, required=True,
                       help='输出路径 (.bin 文件)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='加载设备（建议使用 cpu 以节省显存）')

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"SliceGPT 模型转换器")
    print(f"{'='*80}\n")

    # 1. 检查 slicegpt 是否已安装
    try:
        from slicegpt import hf_utils as slicegpt_hf_utils
        from slicegpt.config import config as slicegpt_config
    except ImportError:
        print("❌ 错误: 未安装 slicegpt 包")
        print("\n请在 slicegpt 环境中运行此脚本：")
        print("  conda activate slicegpt")
        print("  python evaluation/convert_slicegpt_model.py ...")
        print("\n如果还没有安装 slicegpt，请先安装：")
        print("  git clone https://github.com/microsoft/TransformerCompression")
        print("  cd TransformerCompression")
        print("  pip install -e .")
        sys.exit(1)

    from transformers import AutoTokenizer

    # 2. 推断参数
    pt_path = pathlib.Path(args.slicegpt_model)
    if not pt_path.exists():
        print(f"❌ 错误: 找不到模型文件 {args.slicegpt_model}")
        sys.exit(1)

    sliced_model_dir = pt_path.parent

    # 推断 sparsity
    sparsity = args.sparsity
    if sparsity is None:
        try:
            sparsity = float(pt_path.stem.split('_')[-1])
            print(f"✓ 从文件名推断 sparsity = {sparsity}")
        except:
            print(f"❌ 错误: 无法从文件名推断 sparsity，请使用 --sparsity 参数指定")
            sys.exit(1)

    # 推断 base_model
    base_model = args.base_model
    if base_model is None:
        model_name = pt_path.stem.rsplit('_', 1)[0]

        # 尝试常见路径
        possible_paths = [
            str(sliced_model_dir),  # 同目录
            f"/newdata/LLMs/{model_name}",
            f"/data/models/{model_name}",
        ]

        found = False
        for path in possible_paths:
            if pathlib.Path(path).exists() and (pathlib.Path(path) / "config.json").exists():
                base_model = path
                print(f"✓ 找到基础模型: {base_model}")
                found = True
                break

        if not found:
            print(f"❌ 错误: 无法找到基础模型，请使用 --base_model 参数指定")
            print(f"   从文件名推断的模型名: {model_name}")
            sys.exit(1)

    # 3. 加载 SliceGPT 模型
    print(f"\n[1/3] 加载 SliceGPT 模型...")
    print(f"  模型路径: {args.slicegpt_model}")
    print(f"  基础模型: {base_model}")
    print(f"  稀疏度: {sparsity}")
    print(f"  设备: {args.device}")

    # 设置 SliceGPT config
    slicegpt_config.device = torch.device(args.device)
    slicegpt_config.dtype = torch.float16

    model_adapter, tokenizer = slicegpt_hf_utils.load_sliced_model(
        base_model,
        str(sliced_model_dir),
        sparsity=sparsity,
        round_interval=8,
        token=None
    )

    model = model_adapter.model
    model.eval()

    # 移动到 CPU（为了保存）
    if args.device != 'cpu':
        print(f"  移动模型到 CPU 以便保存...")
        model = model.to('cpu')

    print(f"✓ 模型加载完成")

    # 4. 统计模型信息
    print(f"\n[2/3] 统计模型参数...")
    total_params = sum(p.numel() for p in model.parameters())
    total_params_b = total_params / 1e9

    print(f"  总参数量: {total_params:,} ({total_params_b:.2f}B)")

    # 5. 保存为标准格式
    print(f"\n[3/3] 保存为标准格式...")
    print(f"  输出路径: {args.output}")

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存完整信息（与我们的剪枝方法格式一致）
    save_dict = {
        'model': model,
        'tokenizer': tokenizer,
        'method': 'SliceGPT',
        'pruning_ratio': sparsity,
        'actual_ratio': sparsity,  # SliceGPT 的实际剪枝率就是 sparsity
        'config': {
            'base_model': base_model,
            'slicegpt_model': str(args.slicegpt_model),
            'sparsity': sparsity,
            'round_interval': 8,
        }
    }

    torch.save(save_dict, args.output)

    # 检查文件大小
    file_size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"✓ 保存完成")
    print(f"  文件大小: {file_size_gb:.2f} GB")

    # 6. 完成提示
    print(f"\n{'='*80}")
    print(f"✓ 转换完成！")
    print(f"{'='*80}")
    print(f"\n现在可以在 base 环境中评估此模型：")
    print(f"\n  conda activate base")
    print(f"  python evaluation/run_evaluation.py \\")
    print(f"      --model_path {args.output} \\")
    print(f"      --metrics all \\")
    print(f"      --output {output_dir}/evaluation/evaluation_results.json")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
