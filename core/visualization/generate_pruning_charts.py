#!/usr/bin/env python3
"""
为每个剪枝模型生成剪枝比例和保留比例的直方图

用法:
    python generate_pruning_charts.py --result_dir results
    python generate_pruning_charts.py --result_dir results --output_dir charts
"""

import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 配置中文字体
def setup_chinese_font():
    """配置 matplotlib 以支持中文显示"""
    # 尝试使用系统中可用的中文字体
    chinese_fonts = [
        'SimHei',           # 黑体 (Windows)
        'Microsoft YaHei',  # 微软雅黑 (Windows)
        'SimSun',           # 宋体 (Windows)
        'STSong',           # 华文宋体 (Mac)
        'STHeiti',          # 华文黑体 (Mac)
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux)
        'WenQuanYi Zen Hei',    # 文泉驿正黑 (Linux)
        'Noto Sans CJK SC',     # 思源黑体 (Linux)
        'Noto Sans CJK JP',     # 思源黑体日文
        'DejaVu Sans',          # 默认字体（英文回退）
    ]

    # 获取系统可用字体
    available_fonts = set([f.name for f in matplotlib.font_manager.fontManager.ttflist])

    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            return font

    # 如果没有找到中文字体，使用默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


def load_pruning_comparison(model_path: Path) -> Dict:
    """
    加载剪枝对比数据

    Args:
        model_path: 模型目录路径

    Returns:
        剪枝对比数据字典，如果未找到则返回 None
    """
    # 优先尝试 JSON 格式
    json_file = model_path / "analysis" / "pruning_comparison.json"
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ✗ 错误 - 无法加载 {json_file}: {e}")
            return None

    print(f"  ⊘ 未找到剪枝对比数据: {model_path.name}")
    return None


def extract_layer_ratios(pruning_data: Dict) -> Tuple[List[float], List[float]]:
    """
    从剪枝数据中提取每层的剪枝率和保留率

    Args:
        pruning_data: pruning_comparison 数据

    Returns:
        (pruning_ratios, retention_ratios) - 每层的剪枝率和保留率列表
    """
    if not pruning_data or 'layers' not in pruning_data:
        return [], []

    layers = pruning_data['layers']
    pruning_ratios = []
    retention_ratios = []

    for layer in layers:
        if 'total' in layer and 'reduction_ratio' in layer['total']:
            pruning_ratio = layer['total']['reduction_ratio']
            retention_ratio = 1.0 - pruning_ratio

            pruning_ratios.append(pruning_ratio * 100)  # 转换为百分比
            retention_ratios.append(retention_ratio * 100)

    return pruning_ratios, retention_ratios


def plot_pruning_chart(
    layer_indices: List[int],
    ratios: List[float],
    model_name: str,
    chart_type: str,
    output_path: str
):
    """
    绘制剪枝或保留比例的直方图

    Args:
        layer_indices: 层索引列表
        ratios: 比例列表（百分比）
        model_name: 模型名称
        chart_type: 'pruning' 或 'retention'
        output_path: 输出文件路径
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 计算模型平均比例
    avg_ratio = np.mean(ratios)

    # 颜色方案
    if chart_type == 'pruning':
        colors = ['#e74c3c' if r > 80 else '#e67e22' if r > 50 else '#3498db' for r in ratios]
        title = f'{model_name} - 各层剪枝比例'
        ylabel = '剪枝比例 (%)'
    else:  # retention
        colors = ['#27ae60' if r > 80 else '#f39c12' if r > 50 else '#e74c3c' for r in ratios]
        title = f'{model_name} - 各层保留比例'
        ylabel = '保留比例 (%)'

    # 绘制直方图
    bars = ax.bar(layer_indices, ratios, color=colors, edgecolor='black', linewidth=0.5)

    # 添加数值标签 - 修改：小于5%时不显示
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        height = bar.get_height()
        if ratio >= 5:  # 只在比例 >= 5% 时显示标签
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.1f}%',
                   ha='center', va='bottom', fontsize=8, rotation=0)

    # 设置坐标轴
    ax.set_xlabel('层索引', fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax.set_ylim(0, 105)

    # 添加网格线 - 20%, 40%, 60%, 80%, 100% (浅色虚线)
    for y in [20, 40, 60, 80, 100]:
        ax.axhline(y=y, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    # 添加参考线
    if chart_type == 'pruning':
        # 50% 参考线（较深虚线）
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='50% 剪枝', zorder=2)
        # 80% 参考线（醒目红线）
        ax.axhline(y=80, color='darkred', linestyle='-', linewidth=2, alpha=0.8,
                  label='80% 剪枝', zorder=2)
        # 模型平均剪枝比例（醒目蓝线）
        ax.axhline(y=avg_ratio, color='#2980b9', linestyle='-', linewidth=2.5, alpha=0.9,
                  label=f'平均剪枝比例: {avg_ratio:.1f}%', zorder=3)
    else:
        # 50% 参考线（较深虚线）
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                  label='50% 保留', zorder=2)
        # 80% 参考线（醒目红线）
        ax.axhline(y=80, color='darkred', linestyle='-', linewidth=2, alpha=0.8,
                  label='80% 保留', zorder=2)
        # 模型平均保留比例（醒目绿线）
        ax.axhline(y=avg_ratio, color='#16a085', linestyle='-', linewidth=2.5, alpha=0.9,
                  label=f'平均保留比例: {avg_ratio:.1f}%', zorder=3)

    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 已生成: {output_path}")


def generate_charts_for_model(
    model_name: str,
    model_path: Path,
    output_dir: Path
):
    """
    为单个模型生成剪枝和保留比例图

    Args:
        model_name: 模型名称
        model_path: 模型目录路径
        output_dir: 输出目录路径
    """
    print(f"\n处理模型: {model_name}")

    # 加载剪枝数据
    pruning_data = load_pruning_comparison(model_path)
    if not pruning_data:
        return

    # 提取比例数据
    pruning_ratios, retention_ratios = extract_layer_ratios(pruning_data)
    if not pruning_ratios:
        print(f"  ✗ 未找到有效的层数据")
        return

    # 层索引
    layer_indices = list(range(len(pruning_ratios)))

    # 创建模型专属输出目录
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # 生成剪枝比例图
    pruning_chart_path = model_output_dir / f"{model_name}_pruning_ratio.png"
    plot_pruning_chart(
        layer_indices,
        pruning_ratios,
        model_name,
        'pruning',
        str(pruning_chart_path)
    )

    # 生成保留比例图
    retention_chart_path = model_output_dir / f"{model_name}_retention_ratio.png"
    plot_pruning_chart(
        layer_indices,
        retention_ratios,
        model_name,
        'retention',
        str(retention_chart_path)
    )


def main():
    parser = argparse.ArgumentParser(
        description='为每个剪枝模型生成剪枝和保留比例的直方图',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--result_dir',
        type=str,
        default='results',
        help='结果目录路径 (默认: results)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='pruning_charts',
        help='图表输出目录 (默认: pruning_charts)'
    )

    parser.add_argument(
        '--models',
        type=str,
        default=None,
        help='指定要处理的模型（逗号分隔），不指定则处理所有模型'
    )

    args = parser.parse_args()

    # 配置中文字体
    font_used = setup_chinese_font()

    print(f"\n{'='*80}")
    print(f"生成剪枝比例图表")
    print(f"{'='*80}")
    print(f"结果目录: {args.result_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"使用字体: {font_used}")
    print(f"{'='*80}")

    result_path = Path(args.result_dir)
    output_path = Path(args.output_dir)

    if not result_path.exists():
        print(f"\n错误: 结果目录 '{args.result_dir}' 不存在")
        return

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 确定要处理的模型
    if args.models:
        model_names = [m.strip() for m in args.models.split(',')]
        model_dirs = [result_path / name for name in model_names if (result_path / name).is_dir()]
    else:
        # 处理所有模型
        model_dirs = sorted([d for d in result_path.iterdir() if d.is_dir()])

    if not model_dirs:
        print("\n未找到任何模型目录")
        return

    print(f"\n找到 {len(model_dirs)} 个模型")

    # 为每个模型生成图表
    success_count = 0
    for model_dir in model_dirs:
        try:
            generate_charts_for_model(model_dir.name, model_dir, output_path)
            success_count += 1
        except Exception as e:
            print(f"  ✗ 错误: {e}")

    print(f"\n{'='*80}")
    print(f"✓ 完成! 成功为 {success_count}/{len(model_dirs)} 个模型生成图表")
    print(f"图表保存在: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
