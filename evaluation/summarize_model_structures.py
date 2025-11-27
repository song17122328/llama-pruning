#!/usr/bin/env python3
"""
汇总所有模型的结构分析结果

从已有的 analysis/ 目录读取分析结果，生成综合报告
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def load_analysis_results(model_dir: Path) -> Dict:
    """
    加载模型的分析结果

    Args:
        model_dir: 模型目录

    Returns:
        分析结果字典，如果加载失败则返回 None
    """
    analysis_dir = model_dir / "analysis"
    if not analysis_dir.exists():
        return None

    result = {
        'model_name': model_dir.name,
        'has_comparison': False,
        'has_structure': False
    }

    # 加载模型对比结果
    comparison_file = analysis_dir / "model_comparison.json"
    if comparison_file.exists():
        try:
            with open(comparison_file, 'r') as f:
                result['comparison'] = json.load(f)
                result['has_comparison'] = True
        except Exception as e:
            print(f"  ⚠ 无法加载 {model_dir.name}/model_comparison.json: {e}")

    # 加载剪枝后模型结构
    pruned_structure_file = analysis_dir / "pruned_model_analysis.json"
    if pruned_structure_file.exists():
        try:
            with open(pruned_structure_file, 'r') as f:
                result['pruned_structure'] = json.load(f)
                result['has_structure'] = True
        except Exception as e:
            print(f"  ⚠ 无法加载 {model_dir.name}/pruned_model_analysis.json: {e}")

    # 加载原始模型结构
    original_structure_file = analysis_dir / "original_model_analysis.json"
    if original_structure_file.exists():
        try:
            with open(original_structure_file, 'r') as f:
                result['original_structure'] = json.load(f)
        except Exception as e:
            pass

    return result if (result['has_comparison'] or result['has_structure']) else None


def scan_all_models(directories: List[Path]) -> Dict[str, Dict]:
    """
    扫描所有目录下的模型分析结果

    Args:
        directories: 要扫描的目录列表

    Returns:
        所有模型的分析结果
    """
    all_results = {}

    for base_dir in directories:
        if not base_dir.exists():
            print(f"⊘ 跳过不存在的目录: {base_dir}")
            continue

        print(f"\n扫描目录: {base_dir}")
        print(f"{'-'*80}")

        for model_dir in sorted(base_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            result = load_analysis_results(model_dir)
            if result:
                all_results[f"{base_dir.name}/{model_dir.name}"] = result
                status = []
                if result['has_comparison']:
                    status.append("对比✓")
                if result['has_structure']:
                    status.append("结构✓")
                print(f"  ✓ {model_dir.name:30s} [{', '.join(status)}]")
            else:
                print(f"  ⊘ {model_dir.name:30s} [无分析结果]")

    return all_results


def generate_summary_table(all_results: Dict[str, Dict]) -> str:
    """
    生成模型结构汇总表格

    Args:
        all_results: 所有模型的分析结果

    Returns:
        表格字符串
    """
    lines = []
    lines.append("=" * 120)
    lines.append("模型结构分析汇总表")
    lines.append("=" * 120)
    lines.append("")

    # 表头
    header = f"{'模型名称':<40} {'总参数':>15} {'剪枝比例':>10} {'层数':>6} {'完全剪空的层':>12}"
    lines.append(header)
    lines.append("-" * 120)

    # 按参数量排序
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1].get('pruned_structure', {}).get('total_params', 0),
        reverse=True
    )

    for model_key, result in sorted_models:
        model_name = model_key

        # 获取参数信息
        if result['has_structure']:
            structure = result['pruned_structure']
            total_params = structure.get('total_params', 0)
            num_layers = structure['layer_summary'].get('num_layers', 0)
        elif result['has_comparison']:
            comparison = result['comparison']
            total_params = comparison['total_params'].get('pruned', 0)
            num_layers = len(comparison.get('layers', []))
        else:
            continue

        # 获取剪枝比例
        if result['has_comparison']:
            pruning_ratio = result['comparison']['total_params'].get('reduction_ratio', 0)
        else:
            pruning_ratio = 0

        # 统计完全剪空的层
        zero_layers = 0
        if result['has_structure']:
            zero_layers = sum(
                1 for layer in structure.get('layers', [])
                if layer.get('is_zero_layer', False)
            )

        # 格式化输出
        line = f"{model_name:<40} {total_params:>15,} {pruning_ratio*100:>9.2f}% {num_layers:>6} {zero_layers:>12}"
        lines.append(line)

    lines.append("")
    lines.append("=" * 120)

    return '\n'.join(lines)


def generate_detailed_report(all_results: Dict[str, Dict], output_file: Path):
    """
    生成详细的 JSON 报告

    Args:
        all_results: 所有模型的分析结果
        output_file: 输出文件路径
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(all_results),
        'models': []
    }

    for model_key, result in sorted(all_results.items()):
        model_summary = {
            'name': model_key,
            'has_comparison': result['has_comparison'],
            'has_structure': result['has_structure']
        }

        # 添加基本信息
        if result['has_structure']:
            structure = result['pruned_structure']
            model_summary['params'] = {
                'total': structure.get('total_params', 0),
                'embedding': structure.get('embedding_params', 0),
                'lm_head': structure.get('lm_head_params', 0),
                'layers': structure['layer_summary'].get('total_layer_params', 0),
                'num_layers': structure['layer_summary'].get('num_layers', 0)
            }

            # 统计特殊层
            zero_layers = [
                layer['layer_idx'] for layer in structure.get('layers', [])
                if layer.get('is_zero_layer', False)
            ]
            model_summary['zero_layers'] = zero_layers

        # 添加剪枝统计
        if result['has_comparison']:
            comparison = result['comparison']
            model_summary['pruning'] = {
                'original_params': comparison['total_params'].get('original', 0),
                'pruned_params': comparison['total_params'].get('pruned', 0),
                'reduction_ratio': comparison['total_params'].get('reduction_ratio', 0),
                'layer_reduction_ratio': comparison['layer_params'].get('reduction_ratio', 0)
            }

            # 每层的剪枝比例
            layer_pruning = []
            for layer in comparison.get('layers', []):
                layer_pruning.append({
                    'layer_idx': layer['layer_idx'],
                    'reduction_ratio': layer['total'].get('reduction_ratio', 0),
                    'is_zero': layer.get('is_zero_layer', False)
                })
            model_summary['layer_pruning'] = layer_pruning

        report['models'].append(model_summary)

    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 详细报告已保存: {output_file}")


def generate_statistics(all_results: Dict[str, Dict]) -> str:
    """
    生成统计信息

    Args:
        all_results: 所有模型的分析结果

    Returns:
        统计信息字符串
    """
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("统计信息")
    lines.append("=" * 80)

    # 基本统计
    lines.append(f"\n总模型数: {len(all_results)}")

    # 参数量统计
    params_list = []
    for result in all_results.values():
        if result['has_structure']:
            params_list.append(result['pruned_structure'].get('total_params', 0))
        elif result['has_comparison']:
            params_list.append(result['comparison']['total_params'].get('pruned', 0))

    if params_list:
        lines.append(f"\n参数量统计:")
        lines.append(f"  最小: {min(params_list):,}")
        lines.append(f"  最大: {max(params_list):,}")
        lines.append(f"  平均: {sum(params_list) // len(params_list):,}")

    # 剪枝比例统计
    pruning_ratios = []
    for result in all_results.values():
        if result['has_comparison']:
            pruning_ratios.append(result['comparison']['total_params'].get('reduction_ratio', 0))

    if pruning_ratios:
        lines.append(f"\n剪枝比例统计:")
        lines.append(f"  最小: {min(pruning_ratios)*100:.2f}%")
        lines.append(f"  最大: {max(pruning_ratios)*100:.2f}%")
        lines.append(f"  平均: {sum(pruning_ratios)/len(pruning_ratios)*100:.2f}%")

    # 完全剪空的层统计
    total_zero_layers = 0
    models_with_zero_layers = 0
    for result in all_results.values():
        if result['has_structure']:
            zero_count = sum(
                1 for layer in result['pruned_structure'].get('layers', [])
                if layer.get('is_zero_layer', False)
            )
            if zero_count > 0:
                total_zero_layers += zero_count
                models_with_zero_layers += 1

    if models_with_zero_layers > 0:
        lines.append(f"\n完全剪空的层统计:")
        lines.append(f"  有剪空层的模型数: {models_with_zero_layers}")
        lines.append(f"  总剪空层数: {total_zero_layers}")

    lines.append("\n" + "=" * 80)

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='汇总所有模型的结构分析结果',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--dirs',
        type=str,
        nargs='+',
        default=['baselines', 'results'],
        help='要扫描的目录（默认: baselines results）'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models_structure_summary',
        help='输出文件前缀（默认: models_structure_summary）'
    )

    args = parser.parse_args()

    # 扫描目录
    directories = [Path(d) for d in args.dirs]

    print(f"\n{'='*80}")
    print(f"汇总模型结构分析结果")
    print(f"{'='*80}")

    all_results = scan_all_models(directories)

    if not all_results:
        print("\n✗ 未找到任何分析结果")
        return

    print(f"\n✓ 成功加载 {len(all_results)} 个模型的分析结果")

    # 生成汇总表格
    summary_table = generate_summary_table(all_results)
    print(f"\n{summary_table}")

    # 生成统计信息
    statistics = generate_statistics(all_results)
    print(statistics)

    # 保存结果
    output_prefix = Path(args.output)

    # 保存表格（TXT）
    txt_file = output_prefix.parent / f"{output_prefix.name}.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(summary_table)
        f.write("\n\n")
        f.write(statistics)
    print(f"\n✓ 汇总表格已保存: {txt_file}")

    # 保存详细报告（JSON）
    json_file = output_prefix.parent / f"{output_prefix.name}.json"
    generate_detailed_report(all_results, json_file)

    print(f"\n{'='*80}")
    print(f"完成！")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
