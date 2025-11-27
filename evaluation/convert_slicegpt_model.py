#!/usr/bin/env python3
"""
SliceGPT 模型转换器

将 SliceGPT 的 .pt 格式保存为我们评估框架支持的 .bin 格式。

工作原理：
- 在 slicegpt 环境中使用 SliceGPT 官方加载器加载模型
- 使用 dill 序列化保存模型对象（支持 SliceGPT 的动态类）
- 保留 SliceGPT 的所有特殊组件（旋转矩阵、融合的 LayerNorm 等）
- 生成的 .bin 文件可以在 base 环境中使用 dill 加载

注意：
- 需要安装 dill 包：pip install dill
- 不会将 SliceGPT 模型转换为标准 Llama 结构（无法实现）
- 保存的模型包含 SliceGPT 特有的结构修改
- 生成的 .bin 文件可能比其他剪枝方法的文件略大

使用场景：
1. 在 slicegpt 环境中运行此脚本保存模型
2. 在 base 环境中使用保存的模型进行评估（需要 dill，但无需 slicegpt 包）

使用方法：
    # 步骤 0: 安装 dill
    conda activate slicegpt
    pip install dill

    # 步骤 1: 在 slicegpt 环境中转换
    python evaluation/convert_slicegpt_model.py \
        --slicegpt_model results/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
        --base_model /newdata/LLMs/Llama-3-8B-Instruct \
        --output results/SliceGPT_2000/pruned_model.bin

    # 步骤 2: 在 base 环境中评估（也需要 dill）
    conda activate base
    pip install dill
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# 尝试导入 dill（用于序列化动态创建的类）
try:
    import dill
    HAS_DILL = True
except ImportError:
    HAS_DILL = False
    print("⚠️  警告: 未安装 dill 包，将使用标准 pickle")
    print("  如果遇到序列化错误，请安装: pip install dill")

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
    parser.add_argument('--device', type=str, default='cuda',
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

    # 检查是否真的被剪枝了
    from transformers import AutoConfig
    try:
        original_config = AutoConfig.from_pretrained(base_model)
        model_config = model.config

        if hasattr(model_config, 'hidden_size') and hasattr(original_config, 'hidden_size'):
            if model_config.hidden_size != original_config.hidden_size:
                print(f"  ✓ 检测到结构化剪枝:")
                print(f"    原始 hidden_size: {original_config.hidden_size}")
                print(f"    剪枝后 hidden_size: {model_config.hidden_size}")
                print(f"    实际剪枝率: {(1 - model_config.hidden_size / original_config.hidden_size) * 100:.2f}%")
            else:
                print(f"  ⚠️  注意: config 中的 hidden_size 未改变")
                print(f"  （SliceGPT 使用旋转矩阵和切片，可能不更新 config）")
    except Exception as e:
        print(f"  ⚠️  无法比较配置: {str(e)}")

    # 5. 直接保存 SliceGPT 模型（无需转换）
    print(f"\n[3/3] 保存 SliceGPT 模型...")
    print(f"  注意: SliceGPT 模型包含特殊结构（旋转矩阵、融合的 LayerNorm）")
    print(f"  将直接保存完整模型，保留所有 SliceGPT 组件")

    # 移动到 CPU（为了保存）
    if args.device != 'cpu':
        print(f"  移动模型到 CPU 以便保存...")
        model = model.to('cpu')

    print(f"  输出路径: {args.output}")

    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 直接保存 SliceGPT 加载的模型（与我们的剪枝方法格式一致）
    save_dict = {
        'model': model,  # 使用 SliceGPT 加载的原始模型
        'tokenizer': tokenizer,
        'method': 'SliceGPT',
        'pruning_ratio': sparsity,
        'actual_ratio': sparsity,  # SliceGPT 的实际剪枝率就是 sparsity
        'config': {
            'base_model': base_model,
            'slicegpt_model': str(args.slicegpt_model),
            'sparsity': sparsity,
            'round_interval': 8,
            'note': 'Model contains SliceGPT-specific components (rotation matrices, fused LayerNorms)',
        }
    }

    # 尝试使用 dill 保存（支持动态创建的类）
    if HAS_DILL:
        print(f"  使用 dill 序列化（支持 SliceGPT 的动态类）...")
        try:
            with open(args.output, 'wb') as f:
                dill.dump(save_dict, f)
            print(f"  ✓ 使用 dill 保存成功")
        except Exception as e:
            print(f"  ✗ dill 序列化失败: {str(e)[:200]}")
            print(f"  回退到标准 pickle...")
            torch.save(save_dict, args.output)
    else:
        print(f"  使用标准 pickle 序列化...")
        torch.save(save_dict, args.output)

    # 检查文件大小
    file_size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"✓ 保存完成")
    print(f"  文件大小: {file_size_gb:.2f} GB")

    # 6. 完成提示
    print(f"\n{'='*80}")
    print(f"✓ 保存完成！")
    print(f"{'='*80}")
    print(f"\n现在可以在 base 环境中加载此模型进行评估：")
    print(f"\n  conda activate base")
    print(f"  python evaluation/run_evaluation.py \\")
    print(f"      --model_path {args.output} \\")
    print(f"      --metrics all \\")
    print(f"      --output {output_dir}/evaluation/evaluation_results.json")
    print(f"\n注意: 此 .bin 文件包含完整的 SliceGPT 模型（含旋转矩阵和特殊结构）")
    print(f"      可以在不安装 slicegpt 包的环境中直接使用 torch.load() 加载")
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
