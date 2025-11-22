#!/usr/bin/env python3
"""
汇总和可视化所有剪枝模型的评估结果和剪枝统计

用法:
    python generate_results_table.py --result_dir results --output summary_table.csv
    python generate_results_table.py --result_dir results --output summary_table.md --format markdown
    python generate_results_table.py --result_dir results --output summary_table.html --format html
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_evaluation_results(result_dir: str) -> Dict[str, Dict]:
    """
    遍历result目录下所有文件夹，加载evaluation_results.json

    Args:
        result_dir: 结果目录路径

    Returns:
        {model_name: evaluation_results}
    """
    all_results = {}
    result_path = Path(result_dir)

    if not result_path.exists():
        print(f"警告: 结果目录 '{result_dir}' 不存在")
        return all_results

    # 遍历所有子文件夹
    for model_dir in sorted(result_path.iterdir()):
        if not model_dir.is_dir():
            continue

        # 查找 evaluation_results.json
        eval_file = model_dir / "evaluation" / "evaluation_results.json"

        if not eval_file.exists():
            # 尝试直接在模型目录下查找
            eval_file = model_dir / "evaluation_results.json"

        if eval_file.exists():
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    all_results[model_dir.name] = results
                    print(f"✓ 加载评估结果: {model_dir.name}")
            except Exception as e:
                print(f"✗ 错误 - 无法加载 {eval_file}: {e}")
        else:
            print(f"⊘ 跳过 - {model_dir.name} (未找到 evaluation_results.json)")

    return all_results


def load_pruning_comparison(result_dir: str, model_name: str) -> Dict:
    """
    加载剪枝对比数据 (pruning_comparison.json 或 .txt)

    Args:
        result_dir: 结果目录路径
        model_name: 模型名称

    Returns:
        剪枝对比数据字典，如果未找到则返回 None
    """
    model_path = Path(result_dir) / model_name

    # 优先尝试 JSON 格式
    json_file = model_path / "analysis" / "pruning_comparison.json"
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"  ✓ 加载剪枝对比: {model_name} (JSON)")
                return data
        except Exception as e:
            print(f"  ✗ 错误 - 无法加载 {json_file}: {e}")

    # 尝试解析 TXT 格式（备用）
    txt_file = model_path / "analysis" / "pruning_comparison.txt"
    if txt_file.exists():
        try:
            data = parse_pruning_comparison_txt(txt_file)
            print(f"  ✓ 加载剪枝对比: {model_name} (TXT)")
            return data
        except Exception as e:
            print(f"  ✗ 错误 - 无法解析 {txt_file}: {e}")

    print(f"  ⊘ 未找到剪枝对比数据: {model_name}")
    return None


def parse_pruning_comparison_txt(txt_file: Path) -> Dict:
    """
    解析 pruning_comparison.txt 文件

    Args:
        txt_file: TXT 文件路径

    Returns:
        解析后的数据字典
    """
    # 这是一个简化版本，实际使用时建议优先使用 JSON 格式
    # 这里只是作为备用方案
    data = {
        'layers': []
    }

    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 查找表格数据部分
    in_table = False
    for line in lines:
        line = line.strip()

        # 跳过表头
        if line.startswith('Layer') and '总参数' in line:
            in_table = True
            continue

        if in_table and line.startswith('-'):
            continue

        if in_table and line:
            # 解析每一行：Layer idx total_orig total_pruned retention% ...
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                layer_idx = int(parts[0])
                # 从保留比例% 计算 reduction_ratio
                try:
                    retention_pct = float(parts[3].rstrip('%'))
                    reduction_ratio = (100.0 - retention_pct) / 100.0

                    layer_data = {
                        'layer_idx': layer_idx,
                        'total': {
                            'reduction_ratio': reduction_ratio
                        }
                    }
                    data['layers'].append(layer_data)
                except:
                    pass

        # 如果遇到空行或新的分隔线，退出表格解析
        if in_table and (line.startswith('=') or not line):
            break

    return data


def calculate_pruning_statistics(pruning_data: Dict) -> Dict:
    """
    计算剪枝统计信息：标准差、方差、不同保留比例阈值下的层

    Args:
        pruning_data: pruning_comparison 数据

    Returns:
        统计信息字典
    """
    if not pruning_data or 'layers' not in pruning_data:
        return {
            'std': None,
            'var': None,
            'layers_below_5pct': [],
            'layers_below_10pct': [],
            'layers_below_15pct': [],
            'layers_below_20pct': []
        }

    layers = pruning_data['layers']

    # 提取每层的保留比例（1 - reduction_ratio）
    retention_ratios = []
    layer_indices = []

    for layer in layers:
        if 'total' in layer and 'reduction_ratio' in layer['total']:
            reduction_ratio = layer['total']['reduction_ratio']
            retention_ratio = 1.0 - reduction_ratio  # 保留比例
            retention_ratios.append(retention_ratio)
            layer_indices.append(layer['layer_idx'])

    if not retention_ratios:
        return {
            'std': None,
            'var': None,
            'layers_below_5pct': [],
            'layers_below_10pct': [],
            'layers_below_15pct': [],
            'layers_below_20pct': []
        }

    # 转换为 numpy 数组
    retention_ratios = np.array(retention_ratios)

    # 计算标准差和方差
    std = np.std(retention_ratios, ddof=1)  # 样本标准差
    var = np.var(retention_ratios, ddof=1)  # 样本方差

    # 找出保留比例低于不同阈值的层
    layers_below_5pct = [layer_indices[i] for i, r in enumerate(retention_ratios) if r < 0.05]
    layers_below_10pct = [layer_indices[i] for i, r in enumerate(retention_ratios) if r < 0.10]
    layers_below_15pct = [layer_indices[i] for i, r in enumerate(retention_ratios) if r < 0.15]
    layers_below_20pct = [layer_indices[i] for i, r in enumerate(retention_ratios) if r < 0.20]

    return {
        'std': std,
        'var': var,
        'layers_below_5pct': layers_below_5pct,
        'layers_below_10pct': layers_below_10pct,
        'layers_below_15pct': layers_below_15pct,
        'layers_below_20pct': layers_below_20pct
    }


def extract_metrics(all_results: Dict[str, Dict], result_dir: str) -> pd.DataFrame:
    """
    从评估结果中提取所有指标并构建DataFrame

    Args:
        all_results: {model_name: evaluation_results}
        result_dir: 结果目录路径（用于加载剪枝对比数据）

    Returns:
        包含所有指标的DataFrame
    """
    rows = []

    for model_name, results in all_results.items():
        row = {'模型名称': model_name}

        metrics = results.get('metrics', {})

        # 1. 模型大小 (GB)
        model_info = metrics.get('model_info', {})
        row['模型大小 (GB)'] = model_info.get('model_size_gb', None)
        row['参数量 (B)'] = model_info.get('total_params_B', None)

        # 2. PPL 指标
        ppl_results = metrics.get('ppl', {})

        # 处理 wikitext2
        wikitext2_ppl = None
        for key, value in ppl_results.items():
            if 'wikitext2' in key.lower() or 'wikitext-2' in key.lower():
                wikitext2_ppl = value
                break
        row['PPL (WikiText-2)'] = wikitext2_ppl

        # 处理 ptb
        ptb_ppl = None
        for key, value in ppl_results.items():
            if 'ptb' in key.lower() or 'penn' in key.lower():
                ptb_ppl = value
                break
        row['PPL (PTB)'] = ptb_ppl

        # 3. Zero-shot 准确率 (7个任务)
        zeroshot_results = metrics.get('zeroshot', {})

        task_mapping = {
            'boolq': 'BoolQ',
            'piqa': 'PIQA',
            'hellaswag': 'HellaSwag',
            'winogrande': 'WinoGrande',
            'arc_easy': 'ARC-e',
            'arc_challenge': 'ARC-c',
            'openbookqa': 'OBQA'
        }

        for task_key, task_name in task_mapping.items():
            # 查找匹配的任务结果
            task_acc = None
            for key, value in zeroshot_results.items():
                if task_key in key.lower():
                    if isinstance(value, dict) and 'accuracy' in value:
                        task_acc = value['accuracy'] * 100  # 转换为百分比
                    elif isinstance(value, (int, float)):
                        task_acc = value * 100
                    break
            row[f'ZS-{task_name} (%)'] = task_acc

        # 4. 平均 Zero-shot 准确率
        avg_acc = metrics.get('avg_zeroshot_acc', None)
        if avg_acc is not None:
            row['ZS-平均 (%)'] = avg_acc * 100
        else:
            # 如果没有预计算，尝试从各个任务中计算
            accuracies = [row[f'ZS-{name} (%)'] for name in task_mapping.values()
                         if row.get(f'ZS-{name} (%)') is not None]
            if accuracies:
                row['ZS-平均 (%)'] = sum(accuracies) / len(accuracies)

        # 5. 效率指标 (吞吐量和延迟)
        efficiency = metrics.get('efficiency', {})
        speed_results = efficiency.get('speed', {})

        # Batch size = 1
        batch1 = speed_results.get('batch_size_1', {})
        row['吞吐量-BS1 (tokens/s)'] = batch1.get('throughput_tokens_per_sec', None)
        row['延迟-BS1 (ms/token)'] = batch1.get('latency_ms_per_token', None)

        # Batch size = 4
        batch4 = speed_results.get('batch_size_4', {})
        row['吞吐量-BS4 (tokens/s)'] = batch4.get('throughput_tokens_per_sec', None)
        row['延迟-BS4 (ms/token)'] = batch4.get('latency_ms_per_token', None)

        # 6. 显存占用 (可选)
        memory_results = efficiency.get('memory', {})
        row['显存占用 (MB)'] = memory_results.get('model_memory_mb', None)

        # 7. 剪枝统计分析
        pruning_data = load_pruning_comparison(result_dir, model_name)
        if pruning_data:
            pruning_stats = calculate_pruning_statistics(pruning_data)

            row['剪枝标准差'] = pruning_stats['std']
            row['剪枝方差'] = pruning_stats['var']
            row['保留<5%的层'] = str(pruning_stats['layers_below_5pct']) if pruning_stats['layers_below_5pct'] else '[]'
            row['保留<10%的层'] = str(pruning_stats['layers_below_10pct']) if pruning_stats['layers_below_10pct'] else '[]'
            row['保留<15%的层'] = str(pruning_stats['layers_below_15pct']) if pruning_stats['layers_below_15pct'] else '[]'
            row['保留<20%的层'] = str(pruning_stats['layers_below_20pct']) if pruning_stats['layers_below_20pct'] else '[]'
        else:
            # 如果没有剪枝数据，填充 None
            row['剪枝标准差'] = None
            row['剪枝方差'] = None
            row['保留<5%的层'] = None
            row['保留<10%的层'] = None
            row['保留<15%的层'] = None
            row['保留<20%的层'] = None

        rows.append(row)

    # 创建DataFrame
    df = pd.DataFrame(rows)

    # 按模型大小降序排序（最大的在上面）
    df = df.sort_values('模型大小 (GB)', ascending=False).reset_index(drop=True)

    return df


def format_dataframe(df: pd.DataFrame, decimal_places: int = 2) -> pd.DataFrame:
    """
    格式化DataFrame中的数值，保留指定小数位

    Args:
        df: 原始DataFrame
        decimal_places: 小数位数

    Returns:
        格式化后的DataFrame
    """
    formatted_df = df.copy()

    # 对数值列进行格式化
    for col in formatted_df.columns:
        if col == '模型名称' or col.startswith('保留<'):
            continue

        # 检查是否是数值列
        if formatted_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # 根据列名决定保留的小数位
            if 'GB' in col or 'B)' in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                )
            elif '(%)' in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                )
            elif 'tokens/s' in col or 'MB' in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
                )
            elif 'ms' in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
                )
            elif '标准差' in col or '方差' in col:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )
            else:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{x:.{decimal_places}f}" if pd.notna(x) else "N/A"
                )

    return formatted_df


def save_table(df: pd.DataFrame, output_path: str, format: str = 'auto'):
    """
    保存表格到文件

    Args:
        df: DataFrame
        output_path: 输出文件路径
        format: 输出格式 ('auto', 'csv', 'markdown', 'html', 'excel', 'latex')
    """
    # 自动推断格式
    if format == 'auto':
        ext = Path(output_path).suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.md': 'markdown',
            '.html': 'html',
            '.xlsx': 'excel',
            '.tex': 'latex'
        }
        format = format_map.get(ext, 'csv')

    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 保存文件
    if format == 'csv':
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    elif format == 'markdown':
        # 格式化后再保存为markdown
        formatted_df = format_dataframe(df)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_df.to_markdown(index=False))
    elif format == 'html':
        formatted_df = format_dataframe(df)
        with open(output_path, 'w', encoding='utf-8') as f:
            html = formatted_df.to_html(index=False, border=1, justify='center', escape=False)
            # 添加CSS样式美化表格
            html_with_style = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>模型评估结果汇总</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 12px;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            text-align: center;
            font-weight: bold;
        }}
        td {{
            padding: 8px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>模型评估结果汇总</h1>
    {html}
</body>
</html>
"""
            f.write(html_with_style)
    elif format == 'excel':
        df.to_excel(output_path, index=False, engine='openpyxl')
    elif format == 'latex':
        formatted_df = format_dataframe(df)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_df.to_latex(index=False))
    else:
        raise ValueError(f"不支持的格式: {format}")

    print(f"\n✓ 表格已保存到: {output_path} (格式: {format})")


def print_summary(df: pd.DataFrame):
    """打印汇总统计信息"""
    print(f"\n{'='*80}")
    print(f"汇总统计")
    print(f"{'='*80}")
    print(f"总模型数: {len(df)}")
    print(f"总列数: {len(df.columns)}")

    # 打印前几行预览
    print(f"\n预览 (前3行, 部分列):")
    preview_cols = ['模型名称', 'PPL (WikiText-2)', 'ZS-平均 (%)', '剪枝标准差', '保留<5%的层']
    preview_cols = [col for col in preview_cols if col in df.columns]
    if preview_cols:
        print(df[preview_cols].head(3).to_string(index=False))

    # 统计信息
    if 'PPL (WikiText-2)' in df.columns:
        ppl_values = df['PPL (WikiText-2)'].dropna()
        if len(ppl_values) > 0:
            print(f"\nPPL (WikiText-2) 统计:")
            print(f"  - 最小: {ppl_values.min():.2f}")
            print(f"  - 最大: {ppl_values.max():.2f}")
            print(f"  - 平均: {ppl_values.mean():.2f}")

    if 'ZS-平均 (%)' in df.columns:
        acc_values = df['ZS-平均 (%)'].dropna()
        if len(acc_values) > 0:
            print(f"\nZero-shot 平均准确率 统计:")
            print(f"  - 最小: {acc_values.min():.2f}%")
            print(f"  - 最大: {acc_values.max():.2f}%")
            print(f"  - 平均: {acc_values.mean():.2f}%")

    if '剪枝标准差' in df.columns:
        std_values = df['剪枝标准差'].dropna()
        if len(std_values) > 0:
            print(f"\n剪枝标准差 统计:")
            print(f"  - 最小: {std_values.min():.4f}")
            print(f"  - 最大: {std_values.max():.4f}")
            print(f"  - 平均: {std_values.mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='汇总和可视化所有剪枝模型的评估结果和剪枝统计',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default='results',
        help='结果目录路径 (默认: results)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出文件路径 (支持 .csv, .md, .html, .xlsx, .tex)'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='auto',
        choices=['auto', 'csv', 'markdown', 'html', 'excel', 'latex'],
        help='输出格式 (默认: auto，根据文件扩展名自动推断)'
    )

    parser.add_argument(
        '--decimal_places',
        type=int,
        default=2,
        help='数值保留的小数位数 (默认: 2)'
    )

    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='不打印汇总统计信息'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"汇总剪枝模型评估结果")
    print(f"{'='*80}")
    print(f"结果目录: {args.result_dir}")
    print(f"输出文件: {args.output}")
    print(f"输出格式: {args.format}")
    print(f"{'='*80}\n")

    # 1. 加载所有评估结果
    all_results = load_evaluation_results(args.result_dir)

    if not all_results:
        print("\n错误: 未找到任何评估结果")
        return

    print(f"\n✓ 成功加载 {len(all_results)} 个模型的评估结果\n")

    # 2. 提取指标并构建DataFrame（包括剪枝统计）
    df = extract_metrics(all_results, args.result_dir)

    # 3. 打印汇总统计
    if not args.no_summary:
        print_summary(df)

    # 4. 保存表格
    save_table(df, args.output, args.format)

    print(f"\n{'='*80}")
    print(f"✓ 完成!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
