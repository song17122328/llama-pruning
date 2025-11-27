#!/usr/bin/env python3
"""
批量模型结构分析工具

功能：
1. 扫描指定目录下的所有模型
2. 自动识别模型类型（标准模型 vs SliceGPT）
3. 批量分析标准模型的结构
4. 为 SliceGPT 模型生成单独的分析脚本

用法：
    # 第一步：分析标准模型
    python evaluation/batch_model_analysis.py \
        --models_dir baselines/ \
        --base_model /newdata/LLMs/Llama-3-8B-Instruct \
        --output_dir baselines_analysis/

    # 第二步：运行生成的 SliceGPT 脚本（如果有）
    bash baselines_analysis/analyze_slicegpt.sh

    # 第三步：汇总所有结果
    python evaluation/batch_model_analysis.py \
        --models_dir baselines/ \
        --merge_results \
        --output_dir baselines_analysis/
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.model_analysis import ModelAnalyzer, ModelComparator, analyze_model_from_checkpoint
from evaluation.utils.model_loader import load_model_and_tokenizer


def detect_model_type(model_dir: Path) -> str:
    """
    检测模型类型

    Args:
        model_dir: 模型目录路径

    Returns:
        模型类型: 'slicegpt', 'huggingface', 'bin', 或 'no_model'
    """
    # SliceGPT: 有 .json 配置文件（但不是 config.json）
    json_files = list(model_dir.glob("*.json"))
    if json_files:
        for json_file in json_files:
            if json_file.name != "config.json":
                # 验证是否是 SliceGPT 配置
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if 'embedding_dimensions' in data or 'attention_input_dimensions' in data:
                            return 'slicegpt'
                except:
                    pass

    # HuggingFace: 有 config.json
    if (model_dir / "config.json").exists():
        return 'huggingface'

    # .bin 格式
    if (model_dir / "pruned_model.bin").exists():
        return 'bin'

    # 检查是否有 pruned_model 子目录
    if (model_dir / "pruned_model").is_dir():
        return 'huggingface'

    # 只有评估结果，无模型文件
    return 'no_model'


def scan_models(models_dir: Path) -> Dict[str, List[Path]]:
    """
    扫描目录下的所有模型并分类

    Args:
        models_dir: 模型目录

    Returns:
        分类后的模型字典
    """
    models = {
        'standard': [],   # 标准模型（HF或.bin）
        'slicegpt': [],   # SliceGPT 模型
        'no_model': []    # 只有评估结果
    }

    print(f"\n{'='*80}")
    print(f"扫描模型目录: {models_dir}")
    print(f"{'='*80}\n")

    for item in sorted(models_dir.iterdir()):
        if not item.is_dir():
            continue

        model_type = detect_model_type(item)

        if model_type == 'slicegpt':
            models['slicegpt'].append(item)
            print(f"✓ SliceGPT 模型: {item.name}")
        elif model_type in ['huggingface', 'bin']:
            models['standard'].append(item)
            print(f"✓ 标准模型: {item.name} (类型: {model_type})")
        else:
            models['no_model'].append(item)
            print(f"⊘ 跳过: {item.name} (无模型文件)")

    print(f"\n汇总:")
    print(f"  - 标准模型: {len(models['standard'])} 个")
    print(f"  - SliceGPT 模型: {len(models['slicegpt'])} 个")
    print(f"  - 无模型文件: {len(models['no_model'])} 个")

    return models


def get_model_path(model_dir: Path) -> str:
    """
    获取模型的实际路径

    Args:
        model_dir: 模型目录

    Returns:
        模型路径（可能是目录或.bin文件）
    """
    # 检查 .bin 文件
    bin_file = model_dir / "pruned_model.bin"
    if bin_file.exists():
        return str(bin_file)

    # 检查 pruned_model 子目录
    pruned_dir = model_dir / "pruned_model"
    if pruned_dir.is_dir():
        return str(pruned_dir)

    # 如果是 HuggingFace 目录，直接返回
    if (model_dir / "config.json").exists():
        return str(model_dir)

    return None


def analyze_standard_models(
    models: List[Path],
    base_model_path: str,
    device: str = 'cuda'
) -> Dict[str, Dict]:
    """
    批量分析标准模型

    Args:
        models: 模型目录列表
        base_model_path: 原始基础模型路径
        device: 设备

    Returns:
        分析结果字典
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"分析标准模型 ({len(models)} 个)")
    print(f"{'='*80}\n")

    # 先加载原始模型
    print("正在加载原始模型...")
    original_model, _ = load_model_and_tokenizer(
        model_path=base_model_path,
        device=device,
        torch_dtype=torch.float16,
        force_single_device=True
    )
    print("✓ 原始模型加载完成\n")

    # 分析原始模型
    original_analyzer = ModelAnalyzer(original_model, "Original Model")
    original_result = original_analyzer.analyze()

    # 清理原始模型以释放内存
    del original_model
    torch.cuda.empty_cache()

    # 分析每个剪枝模型
    for i, model_dir in enumerate(models, 1):
        model_name = model_dir.name
        print(f"\n{'='*80}")
        print(f"[{i}/{len(models)}] 分析模型: {model_name}")
        print(f"{'='*80}")

        try:
            # 获取模型路径
            model_path = get_model_path(model_dir)
            if not model_path:
                print(f"✗ 错误: 无法找到模型文件")
                continue

            print(f"模型路径: {model_path}")

            # 加载并分析模型
            pruned_result = analyze_model_from_checkpoint(
                checkpoint_path=model_path,
                device=device,
                verbose=False
            )

            # 对比分析
            comparator = ModelComparator(
                original_analysis=original_result,
                pruned_analysis=pruned_result,
                original_name="Llama-3-8B-Instruct",
                pruned_name=model_name
            )
            comparison_result = comparator.compare()

            # 创建输出目录
            analysis_dir = model_dir / "analysis"
            analysis_dir.mkdir(exist_ok=True)

            # 保存分析结果
            # 1. 保存模型结构
            structure_path = analysis_dir / "model_structure.json"
            with open(structure_path, 'w', encoding='utf-8') as f:
                json.dump(pruned_result, f, indent=2, ensure_ascii=False)
            print(f"✓ 结构分析已保存: {structure_path}")

            # 2. 保存对比结果（JSON）
            comparison_json_path = analysis_dir / "model_comparison.json"
            with open(comparison_json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            print(f"✓ 对比结果已保存: {comparison_json_path}")

            # 2.1 同时保存为 pruning_comparison.json（兼容可视化工具）
            pruning_comparison_path = analysis_dir / "pruning_comparison.json"
            with open(pruning_comparison_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_result, f, indent=2, ensure_ascii=False)
            print(f"✓ 剪枝对比已保存: {pruning_comparison_path}")

            # 3. 保存摘要（TXT）
            summary_path = analysis_dir / "structure_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"模型结构分析摘要\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"模型名称: {model_name}\n")
                f.write(f"模型路径: {model_path}\n\n")
                f.write(f"总参数量: {pruned_result['total_params']:,}\n")
                f.write(f"  - Embedding: {pruned_result['embedding_params']:,}\n")
                f.write(f"  - LM Head: {pruned_result['lm_head_params']:,}\n")
                f.write(f"  - Decoder Layers: {pruned_result['layer_summary']['total_layer_params']:,}\n")
                f.write(f"  - 层数: {pruned_result['layer_summary']['num_layers']}\n\n")

                f.write(f"剪枝统计:\n")
                total_comp = comparison_result['total_params']
                f.write(f"  - 原始参数: {total_comp['original']:,}\n")
                f.write(f"  - 剪枝后参数: {total_comp['pruned']:,}\n")
                f.write(f"  - 减少参数: {total_comp['reduced']:,}\n")
                f.write(f"  - 剪枝比例: {total_comp['reduction_ratio']*100:.2f}%\n")
            print(f"✓ 摘要已保存: {summary_path}")

            results[model_name] = {
                'structure': pruned_result,
                'comparison': comparison_result,
                'status': 'success'
            }

            # 清理模型以释放内存
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ 错误: {str(e)}")
            results[model_name] = {
                'status': 'error',
                'error': str(e)
            }
            continue

    return results


def generate_slicegpt_script(
    slicegpt_models: List[Path],
    base_model_path: str,
    output_path: Path
):
    """
    生成 SliceGPT 模型分析脚本

    Args:
        slicegpt_models: SliceGPT 模型目录列表
        base_model_path: 原始基础模型路径
        output_path: 输出脚本路径
    """
    if not slicegpt_models:
        print("\n⊘ 没有 SliceGPT 模型需要处理")
        return

    print(f"\n{'='*80}")
    print(f"生成 SliceGPT 分析脚本")
    print(f"{'='*80}\n")

    script_lines = [
        "#!/bin/bash",
        "#",
        "# SliceGPT 模型结构分析脚本",
        "# 自动生成于: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "#",
        "# 使用方法：",
        "#   conda activate slicegpt",
        "#   bash " + str(output_path),
        "#",
        "",
        "echo \"=======================================\"",
        "echo \"SliceGPT 模型结构分析\"",
        "echo \"=======================================\"",
        "echo \"\"",
        "",
        "# 检查是否在 slicegpt 环境中",
        "if [[ \"$CONDA_DEFAULT_ENV\" != \"slicegpt\" ]]; then",
        "    echo \"错误: 请先激活 slicegpt 环境\"",
        "    echo \"运行: conda activate slicegpt\"",
        "    exit 1",
        "fi",
        "",
        "# 确保安装了 dill",
        "echo \"检查依赖...\"",
        "pip show dill > /dev/null 2>&1 || pip install dill",
        "echo \"\"",
        ""
    ]

    for i, model_dir in enumerate(slicegpt_models, 1):
        # 查找 .json 配置文件
        json_files = [f for f in model_dir.glob("*.json") if f.name != "config.json"]
        if not json_files:
            continue

        json_file = json_files[0]
        analysis_dir = model_dir / "analysis"

        script_lines.extend([
            f"echo \"=======================================\"",
            f"echo \"[{i}/{len(slicegpt_models)}] 分析模型: {model_dir.name}\"",
            f"echo \"=======================================\"",
            f"echo \"模型配置: {json_file}\"",
            f"echo \"\"",
            "",
            f"# 创建输出目录",
            f"mkdir -p {analysis_dir}",
            "",
            f"# 使用 SliceGPT 的特殊加载方式分析模型",
            f"# 注意：这里需要实现 SliceGPT 的分析逻辑",
            f"# 暂时跳过，因为 SliceGPT 需要特殊的加载方式",
            f"echo \"⚠️  警告: SliceGPT 模型需要特殊处理，请手动分析\"",
            f"echo \"  配置文件: {json_file}\"",
            f"echo \"  输出目录: {analysis_dir}\"",
            f"echo \"\"",
            ""
        ])

    script_lines.extend([
        "echo \"=======================================\"",
        "echo \"完成！\"",
        "echo \"=======================================\""
    ])

    # 写入脚本
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(script_lines))

    # 设置可执行权限
    os.chmod(output_path, 0o755)

    print(f"✓ SliceGPT 分析脚本已生成: {output_path}")
    print(f"\n使用方法:")
    print(f"  conda activate slicegpt")
    print(f"  bash {output_path}")
    print(f"\n注意: SliceGPT 模型需要在 slicegpt 环境中手动处理")


def merge_results(models_dir: Path, output_dir: Path):
    """
    汇总所有模型的分析结果

    Args:
        models_dir: 模型目录
        output_dir: 输出目录
    """
    print(f"\n{'='*80}")
    print(f"汇总分析结果")
    print(f"{'='*80}\n")

    summary = {
        'timestamp': datetime.now().isoformat(),
        'models_dir': str(models_dir),
        'models': []
    }

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # 检查是否有分析结果
        structure_file = model_dir / "analysis" / "model_structure.json"
        comparison_file = model_dir / "analysis" / "model_comparison.json"

        if not structure_file.exists():
            continue

        try:
            with open(structure_file, 'r') as f:
                structure = json.load(f)

            model_summary = {
                'name': model_dir.name,
                'total_params': structure['total_params'],
                'num_layers': structure['layer_summary']['num_layers'],
                'layer_params': structure['layer_summary']['total_layer_params'],
                'embedding_params': structure['embedding_params'],
                'lm_head_params': structure['lm_head_params']
            }

            # 如果有对比结果，添加剪枝统计
            if comparison_file.exists():
                with open(comparison_file, 'r') as f:
                    comparison = json.load(f)
                model_summary['pruning'] = {
                    'original_params': comparison['total_params']['original'],
                    'pruned_params': comparison['total_params']['pruned'],
                    'reduction_ratio': comparison['total_params']['reduction_ratio']
                }

            summary['models'].append(model_summary)
            print(f"✓ 加载: {model_dir.name}")

        except Exception as e:
            print(f"✗ 错误 - {model_dir.name}: {str(e)}")
            continue

    # 保存汇总结果
    output_file = output_dir / "models_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 汇总结果已保存: {output_file}")

    # 打印统计
    print(f"\n汇总统计:")
    print(f"  - 总模型数: {len(summary['models'])}")

    if summary['models']:
        total_params = [m['total_params'] for m in summary['models']]
        print(f"  - 参数量范围: {min(total_params):,} ~ {max(total_params):,}")

        pruned_models = [m for m in summary['models'] if 'pruning' in m]
        if pruned_models:
            pruning_ratios = [m['pruning']['reduction_ratio'] for m in pruned_models]
            print(f"  - 剪枝比例范围: {min(pruning_ratios)*100:.2f}% ~ {max(pruning_ratios)*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='批量模型结构分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 分析标准模型（第一步）:
   python evaluation/batch_model_analysis.py \\
       --models_dir baselines/ \\
       --base_model /newdata/LLMs/Llama-3-8B-Instruct \\
       --output_dir baselines_analysis/

2. 运行 SliceGPT 脚本（第二步，如果有）:
   conda activate slicegpt
   bash baselines_analysis/analyze_slicegpt.sh

3. 汇总所有结果（第三步）:
   python evaluation/batch_model_analysis.py \\
       --models_dir baselines/ \\
       --merge_results \\
       --output_dir baselines_analysis/
        """
    )

    parser.add_argument(
        '--models_dir',
        type=str,
        required=True,
        help='模型目录路径（例如: baselines/）'
    )

    parser.add_argument(
        '--base_model',
        type=str,
        default='/newdata/LLMs/Llama-3-8B-Instruct',
        help='原始基础模型路径'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='baselines_analysis',
        help='输出目录'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='设备 (cuda/cpu)'
    )

    parser.add_argument(
        '--merge_results',
        action='store_true',
        help='仅汇总已有的分析结果'
    )

    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)

    # 创建输出目录
    output_dir.mkdir(exist_ok=True)

    # 如果是汇总模式
    if args.merge_results:
        merge_results(models_dir, output_dir)
        return

    # 扫描模型
    models = scan_models(models_dir)

    # 分析标准模型
    if models['standard']:
        results = analyze_standard_models(
            models=models['standard'],
            base_model_path=args.base_model,
            device=args.device
        )

        # 保存批量分析结果
        batch_results_file = output_dir / "batch_analysis_results.json"
        with open(batch_results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'base_model': args.base_model,
                'device': args.device,
                'results': {k: {'status': v['status']} for k, v in results.items()}
            }, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"✓ 标准模型分析完成")
        print(f"✓ 批量分析结果已保存: {batch_results_file}")
        print(f"{'='*80}")

    # 生成 SliceGPT 脚本
    if models['slicegpt']:
        slicegpt_script_path = output_dir / "analyze_slicegpt.sh"
        generate_slicegpt_script(
            slicegpt_models=models['slicegpt'],
            base_model_path=args.base_model,
            output_path=slicegpt_script_path
        )

    # 汇总结果
    print(f"\n{'='*80}")
    print(f"下一步操作:")
    print(f"{'='*80}")

    if models['slicegpt']:
        print(f"\n1. 运行 SliceGPT 分析脚本:")
        print(f"   conda activate slicegpt")
        print(f"   bash {output_dir}/analyze_slicegpt.sh")
        print(f"\n2. 汇总所有结果:")
        print(f"   python evaluation/batch_model_analysis.py \\")
        print(f"       --models_dir {args.models_dir} \\")
        print(f"       --merge_results \\")
        print(f"       --output_dir {args.output_dir}")
    else:
        print(f"\n汇总所有结果:")
        print(f"   python evaluation/batch_model_analysis.py \\")
        print(f"       --models_dir {args.models_dir} \\")
        print(f"       --merge_results \\")
        print(f"       --output_dir {args.output_dir}")

    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
