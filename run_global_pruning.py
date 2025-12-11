#!/usr/bin/env python3
"""
基于全局性价比排序的混合结构化剪枝

核心思想：
- 将剪枝问题建模为分数背包问题
- Score = Importance / Cost
- 全局排序，优先剪除"性价比"最低的 groups
- 自动实现深度剪枝（层移除）+ 宽度剪枝（神经元剪除）的混合策略
"""

import os
import gc
import torch
import argparse
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from core.methods.global_pruning import (
    build_global_group_table,
    select_groups_to_prune
)
from core.methods.gqa_aware import prune_attention_by_gqa_groups
from core.datasets import DatasetManager
from core.models import IdentityDecoderLayer, ZeroAttention, ZeroMLP
from evaluation.metrics.ppl import PPLMetric
from core.utils.logger import LoggerWithDepth
from core.analysis import ModelAnalyzer, ModelComparator
from core.analysis.gradient_analysis import GradientAnalyzer

import sys
# 导入 evaluation 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'evaluation'))
from evaluation.run_evaluation import evaluate_single_model

def setup_chinese_font():
    """配置 matplotlib 以支持中文显示"""
    import matplotlib.font_manager as fm
    
    chinese_fonts = [
        'SimHei', 'Microsoft YaHei', 'SimSun',  # Windows
        'STSong', 'STHeiti',  # Mac
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',  # Linux
        'Noto Sans CJK SC', 'Noto Sans CJK',  # 通用
    ]
    
    # 获取系统所有可用字体
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    # 查找第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    
    # 如果没有找到中文字体，尝试查找包含 CJK 的字体
    for font_obj in fm.fontManager.ttflist:
        if 'CJK' in font_obj.name or 'Chinese' in font_obj.name:
            selected_font = font_obj.name
            plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return selected_font
    
    # 实在找不到中文字体，返回 None 表示不支持中文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None


def generate_pruning_charts(pruning_data, model_name, output_dir, use_english=False):
    """
    生成剪枝直方图（剪枝率和保留率）

    Args:
        pruning_data: pruning_comparison 数据
        model_name: 模型名称
        output_dir: 输出目录路径
        use_english: 是否使用英文标签（当中文字体不可用时）
    """
    if not pruning_data or 'layers' not in pruning_data:
        return

    layers = pruning_data['layers']
    if not layers:
        return

    # 提取每层的剪枝率和保留率
    pruning_ratios = []
    retention_ratios = []
    layer_indices = []

    # 提取每层的 MLP 和 Attention 剪枝比重（占总参数的比例）
    mlp_pruning_ratios = []
    attention_pruning_ratios = []

    # 提取每层的 Attention 和 MLP 各自的剪枝率
    attention_reduction_ratios = []  # Attention剪枝参数 / 原始Attention参数
    mlp_reduction_ratios = []        # MLP剪枝参数 / 原始MLP参数

    for layer in layers:
        if 'total' in layer and 'reduction_ratio' in layer['total']:
            layer_indices.append(layer['layer_idx'])
            pruning_ratio = layer['total']['reduction_ratio']
            pruning_ratios.append(pruning_ratio * 100)
            retention_ratios.append((1.0 - pruning_ratio) * 100)

            # 计算 MLP 和 Attention 的剪枝比重（剪枝参数数 / 原始层总参数）
            total_original = layer['total']['original']

            # MLP 剪枝比重 = MLP剪枝参数 / 原始层总参数
            if total_original > 0 and 'mlp' in layer:
                mlp_reduced = layer['mlp'].get('reduced', 0)
                mlp_pruning_ratios.append(mlp_reduced / total_original * 100)
            else:
                mlp_pruning_ratios.append(0)

            # Attention 剪枝比重 = Attention剪枝参数 / 原始层总参数
            if total_original > 0 and 'attention' in layer:
                attention_reduced = layer['attention'].get('reduced', 0)
                attention_pruning_ratios.append(attention_reduced / total_original * 100)
            else:
                attention_pruning_ratios.append(0)

            # Attention 自身的剪枝率 = Attention剪枝参数 / 原始Attention总参数
            if 'attention' in layer and layer['attention'].get('original', 0) > 0:
                attention_original = layer['attention']['original']
                attention_reduced = layer['attention'].get('reduced', 0)
                attention_reduction_ratios.append(attention_reduced / attention_original * 100)
            else:
                attention_reduction_ratios.append(0)

            # MLP 自身的剪枝率 = MLP剪枝参数 / 原始MLP总参数
            if 'mlp' in layer and layer['mlp'].get('original', 0) > 0:
                mlp_original = layer['mlp']['original']
                mlp_reduced = layer['mlp'].get('reduced', 0)
                mlp_reduction_ratios.append(mlp_reduced / mlp_original * 100)
            else:
                mlp_reduction_ratios.append(0)

    if not pruning_ratios:
        return

    # 获取整体和层目标比例
    total_ratio = None
    layer_target_ratio = None

    if 'total_params' in pruning_data and 'reduction_ratio' in pruning_data['total_params']:
        total_reduction = pruning_data['total_params']['reduction_ratio'] * 100
        total_ratio = total_reduction

    if 'layer_params' in pruning_data and 'reduction_ratio' in pruning_data['layer_params']:
        layer_reduction = pruning_data['layer_params']['reduction_ratio'] * 100
        layer_target_ratio = layer_reduction

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 定义标签（中文或英文）
    if use_english:
        labels = {
            'pruning_ylabel': 'Pruning Ratio (%)',
            'retention_ylabel': 'Retention Ratio (%)',
            'xlabel': 'Layer Index',
            'pruning_title': f'{model_name} - Pruning Ratio per Layer',
            'retention_title': f'{model_name} - Retention Ratio per Layer',
            'pruning_title_full': f'{model_name} - Pruning Ratio (Overall: {{0:.1f}}%, Layer Target: {{1:.1f}}%)',
            'retention_title_full': f'{model_name} - Retention Ratio (Overall: {{0:.1f}}%, Layer Target: {{1:.1f}}%)',
            'pruning_legend': 'Layer Target Pruning: {0:.1f}%',
            'retention_legend': 'Layer Target Retention: {0:.1f}%',
        }
    else:
        labels = {
            'pruning_ylabel': '剪枝比例 (%)',
            'retention_ylabel': '保留比例 (%)',
            'xlabel': '层索引',
            'pruning_title': f'{model_name} - 各层剪枝比例',
            'retention_title': f'{model_name} - 各层保留比例',
            'pruning_title_full': f'{model_name} - 各层剪枝比例 (模型整体: {{0:.1f}}%, 层目标: {{1:.1f}}%)',
            'retention_title_full': f'{model_name} - 各层保留比例 (模型整体: {{0:.1f}}%, 层目标: {{1:.1f}}%)',
            'pruning_legend': '层目标剪枝: {0:.1f}%',
            'retention_legend': '层目标保留: {0:.1f}%',
        }

    # 生成两个图表：剪枝率和保留率
    for chart_type, ratios in [('pruning', pruning_ratios), ('retention', retention_ratios)]:
        fig, ax = plt.subplots(figsize=(14, 6))

        # 计算当前图表对应的目标比例
        if chart_type == 'pruning':
            target = layer_target_ratio
            colors = ['#e74c3c' if r >= target else '#3498db' for r in ratios] if target else ['#3498db'] * len(ratios)
            ylabel = labels['pruning_ylabel']
            if total_ratio and target:
                title = labels['pruning_title_full'].format(total_ratio, target)
            else:
                title = labels['pruning_title']
            line_color = '#ff8c00'
            line_label = labels['pruning_legend'].format(target) if target else None
        else:  # retention
            target = (100 - layer_target_ratio) if layer_target_ratio else None
            colors = ['#27ae60' if r >= target else '#e67e22' for r in ratios] if target else ['#27ae60'] * len(ratios)
            ylabel = labels['retention_ylabel']
            total_ret = (100 - total_ratio) if total_ratio else None
            if total_ret and target:
                title = labels['retention_title_full'].format(total_ret, target)
            else:
                title = labels['retention_title']
            line_color = '#27ae60'
            line_label = labels['retention_legend'].format(target) if target else None

        # 绘制直方图
        bars = ax.bar(layer_indices, ratios, color=colors, edgecolor='black', linewidth=0.5)

        # 添加数值标签
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.1f}%',
                   ha='center', va='bottom', fontsize=8)

        # 设置坐标轴
        ax.set_xlabel(labels['xlabel'], fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(layer_indices)
        ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
        ax.set_ylim(0, 105)

        # 添加网格线
        for y in [20, 40, 60, 80, 100]:
            ax.axhline(y=y, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

        # 添加目标线
        if target is not None and line_label:
            ax.axhline(y=target, color=line_color, linestyle='--', linewidth=2.5, alpha=0.9,
                      label=line_label, zorder=3)
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        plt.tight_layout()

        # 保存图表
        chart_path = output_path / f"{chart_type}_ratio.png"
        plt.savefig(str(chart_path), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 已生成: {chart_path}")

    # ========== 生成剪枝比重细分图表（MLP vs Attention）==========
    fig, ax = plt.subplots(figsize=(14, 6))

    # 创建堆叠柱状图
    width = 0.8
    x_pos = range(len(layer_indices))

    # 绘制 MLP 剪枝比重（底部，蓝色）
    bars_mlp = ax.bar(x_pos, mlp_pruning_ratios, width,
                      label='MLP' if use_english else 'MLP 剪枝比重',
                      color='#3498db', edgecolor='black', linewidth=0.5)

    # 绘制 Attention 剪枝比重（堆叠在 MLP 上方，红色）
    bars_attn = ax.bar(x_pos, attention_pruning_ratios, width,
                       bottom=mlp_pruning_ratios,
                       label='Attention' if use_english else 'Attention 剪枝比重',
                       color='#e74c3c', edgecolor='black', linewidth=0.5)

    # 添加数值标签（显示 MLP 和 Attention 的比重）
    for i, (mlp_ratio, attn_ratio) in enumerate(zip(mlp_pruning_ratios, attention_pruning_ratios)):
        # MLP 标签（在 MLP 柱子中间）
        if mlp_ratio > 2:  # 只有当比重足够大时才显示
            ax.text(i, mlp_ratio / 2, f'{mlp_ratio:.1f}%',
                   ha='center', va='center', fontsize=7, color='white', fontweight='bold')

        # Attention 标签（在 Attention 柱子中间）
        if attn_ratio > 2:  # 只有当比重足够大时才显示
            ax.text(i, mlp_ratio + attn_ratio / 2, f'{attn_ratio:.1f}%',
                   ha='center', va='center', fontsize=7, color='white', fontweight='bold')

        # 总剪枝率标签（在柱子顶部）
        total_ratio = mlp_ratio + attn_ratio
        ax.text(i, total_ratio + 1, f'{total_ratio:.1f}%',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 设置坐标轴
    if use_english:
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pruning Ratio (% of Total Layer Params)', fontsize=12, fontweight='bold')
        title = f'{model_name} - Pruning Breakdown: MLP vs Attention'
        if layer_target_ratio:
            title += f' (Target: {layer_target_ratio:.1f}%)'
    else:
        ax.set_xlabel('层索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('剪枝比重 (占总参数的百分比)', fontsize=12, fontweight='bold')
        title = f'{model_name} - 剪枝比重细分：MLP vs Attention'
        if layer_target_ratio:
            title += f' (目标: {layer_target_ratio:.1f}%)'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax.set_ylim(0, 105)

    # 添加网格线
    for y in [20, 40, 60, 80, 100]:
        ax.axhline(y=y, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    # 添加目标线（总剪枝目标）
    if layer_target_ratio is not None:
        ax.axhline(y=layer_target_ratio, color='#ff8c00', linestyle='--', linewidth=2.5, alpha=0.9,
                  label=f'{"Target" if use_english else "目标剪枝率"}: {layer_target_ratio:.1f}%', zorder=3)

    # 添加图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()

    # 保存图表
    breakdown_chart_path = output_path / "pruning_ratio_breakdown.png"
    plt.savefig(str(breakdown_chart_path), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 已生成: {breakdown_chart_path}")

    # ========== 生成 Attention 自身剪枝率图表 ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    # 设置颜色（红色系）
    colors = ['#e74c3c' if r >= 50 else '#3498db' for r in attention_reduction_ratios]

    # 绘制柱状图
    bars = ax.bar(layer_indices, attention_reduction_ratios, color=colors,
                  edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for bar, ratio in zip(bars, attention_reduction_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.1f}%',
               ha='center', va='bottom', fontsize=8)

    # 设置坐标轴
    if use_english:
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Pruning Ratio (%)', fontsize=12, fontweight='bold')
        title = f'{model_name} - Attention Module Pruning Ratio per Layer'
    else:
        ax.set_xlabel('层索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention 剪枝率 (%)', fontsize=12, fontweight='bold')
        title = f'{model_name} - 各层 Attention 模块剪枝率'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax.set_ylim(0, 105)

    # 添加网格线
    for y in [20, 40, 60, 80, 100]:
        ax.axhline(y=y, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    plt.tight_layout()

    # 保存图表
    attention_chart_path = output_path / "attention_pruning_ratio.png"
    plt.savefig(str(attention_chart_path), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 已生成: {attention_chart_path}")

    # ========== 生成 MLP 自身剪枝率图表 ==========
    fig, ax = plt.subplots(figsize=(14, 6))

    # 设置颜色（蓝色系）
    colors = ['#e74c3c' if r >= 50 else '#3498db' for r in mlp_reduction_ratios]

    # 绘制柱状图
    bars = ax.bar(layer_indices, mlp_reduction_ratios, color=colors,
                  edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for bar, ratio in zip(bars, mlp_reduction_ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ratio:.1f}%',
               ha='center', va='bottom', fontsize=8)

    # 设置坐标轴
    if use_english:
        ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('MLP Pruning Ratio (%)', fontsize=12, fontweight='bold')
        title = f'{model_name} - MLP Module Pruning Ratio per Layer'
    else:
        ax.set_xlabel('层索引', fontsize=12, fontweight='bold')
        ax.set_ylabel('MLP 剪枝率 (%)', fontsize=12, fontweight='bold')
        title = f'{model_name} - 各层 MLP 模块剪枝率'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(layer_indices)
    ax.set_xticklabels([str(i) for i in layer_indices], fontsize=9)
    ax.set_ylim(0, 105)

    # 添加网格线
    for y in [20, 40, 60, 80, 100]:
        ax.axhline(y=y, color='lightgray', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)

    plt.tight_layout()

    # 保存图表
    mlp_chart_path = output_path / "mlp_pruning_ratio.png"
    plt.savefig(str(mlp_chart_path), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 已生成: {mlp_chart_path}")


def get_model_gqa_config(model):
    """
    自动检测模型的 GQA 配置

    支持 LLaMA、Mistral、Qwen 等模型的自动配置检测

    Args:
        model: HuggingFace model

    Returns:
        tuple: (num_attention_heads, num_key_value_heads, gqa_ratio, head_dim)
    """
    config = model.config

    # 获取 attention heads 数量
    num_attention_heads = config.num_attention_heads
    # 有些模型可能没有 num_key_value_heads 字段（如旧的 MHA 模型）
    num_key_value_heads = getattr(config, 'num_key_value_heads', num_attention_heads)

    # 计算 GQA ratio
    gqa_ratio = num_attention_heads // num_key_value_heads

    # 计算 head_dim
    hidden_size = config.hidden_size
    head_dim = hidden_size // num_attention_heads

    return num_attention_heads, num_key_value_heads, gqa_ratio, head_dim


def setup_output_directories(base_dir):
    """
    创建统一的输出目录结构

    Args:
        base_dir: 基础输出目录（例如: prune_log/my_experiment）

    Returns:
        dict: 包含各个子目录路径的字典
    """
    dirs = {
        'base': base_dir,
        # 'models': os.path.join(base_dir, 'models'),           # 保存模型
        'models': base_dir,   
        'analysis': os.path.join(base_dir, 'analysis'),       # 保存中间分析结果
        'evaluation': os.path.join(base_dir, 'evaluation'),   # 保存评估结果
        'logs': os.path.join(base_dir, 'logs'),               # 保存日志
        'visualization': os.path.join(base_dir, 'visualization')  # 保存剪枝可视化结果
    }

    # 创建所有目录
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def collect_layer_activations(model, input_ids, device='cuda'):
    """
    收集每层的激活值用于 Wanda 方法 (修正版：L2 Norm + 正确的 Hook 位置)

    关键修正：
    1. 使用 L2 Norm 而非 Mean (符合 Wanda 论文)
    2. 直接 Hook down_proj 获取包含 SwiGLU 作用后的真实输入

    Returns:
        activations: Dict[layer_idx -> Dict[name -> Tensor]]
                    每个 Tensor 是 [hidden_dim] 的 L2 Norm
    """
    activations = {}
    hooks = []

    def get_activation_hook(layer_idx, name):
        def hook(module, input, output):
            if layer_idx not in activations:
                activations[layer_idx] = {}

            # 提取输入激活值
            if isinstance(input, tuple):
                x = input[0].detach()
            else:
                x = input.detach()

            # 展平 Batch 和 Seq 维度 -> [Total_Tokens, Hidden]
            x = x.reshape(-1, x.shape[-1])

            # Wanda 标准：计算每个 Input Channel 的 L2 Norm
            # L2 Norm = sqrt(sum(x^2)) over all tokens
            norm = x.pow(2).sum(dim=0).sqrt().cpu()

            activations[layer_idx][name] = norm
        return hook

    # 为每层的关键模块注册 hooks
    for layer_idx, layer in enumerate(model.model.layers):
        # Attention 的输入激活
        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'q_proj')))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'k_proj')))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'v_proj')))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'o_proj')))

        # MLP 输入激活
        hooks.append(layer.mlp.gate_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'gate_proj')))
        hooks.append(layer.mlp.up_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'up_proj')))

        # 【关键修正】直接 Hook down_proj，获取包含 Gate 作用后的真实输入
        # down_proj 的输入是 SiLU(gate_proj(x)) * up_proj(x)
        hooks.append(layer.mlp.down_proj.register_forward_hook(
            get_activation_hook(layer_idx, 'down_proj')))

    # 执行前向传播
    with torch.no_grad():
        model(input_ids)

    # 移除所有 hooks
    for hook in hooks:
        hook.remove()

    return activations


def apply_global_pruning(model, groups_to_prune_df, head_dim=128, gqa_ratio=4, logger=None):
    """
    根据全局分析表执行实际剪枝

    Args:
        model: 模型
        groups_to_prune_df: 要剪枝的 groups DataFrame
        head_dim: attention head 维度
        gqa_ratio: Q:KV 比例
        logger: 日志记录器

    Returns:
        pruned_layers: 被完全剪空的层列表
        pruning_stats: 剪枝统计信息
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    log("\n" + "="*60)
    log("执行全局剪枝")
    log("="*60)

    num_layers = len(model.model.layers)
    pruning_stats = {
        'attention': {},  # {layer_idx: (old_kv, new_kv)}
        'mlp': {},        # {layer_idx: (old_channels, new_channels)}
        'empty_layers': []
    }

    # 按层组织要剪枝的 groups
    layer_prune_info = {}
    for layer_idx in range(num_layers):
        layer_data = groups_to_prune_df[groups_to_prune_df['layer_idx'] == layer_idx]

        attn_groups = layer_data[layer_data['group_type'] == 'attention']['group_idx'].tolist()
        mlp_groups = layer_data[layer_data['group_type'] == 'mlp']['group_idx'].tolist()

        layer_prune_info[layer_idx] = {
            'attention': attn_groups,
            'mlp': mlp_groups
        }

    # 执行剪枝
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        prune_info = layer_prune_info[layer_idx]

        log(f"\n处理 Layer {layer_idx}:")

        # ========== Attention 剪枝 ==========
        attn_prune_indices = prune_info['attention']

        if len(attn_prune_indices) > 0:
            # 获取当前 KV heads 数量（从权重形状推断）
            k_proj_out_features = layer.self_attn.k_proj.out_features
            num_kv_heads = k_proj_out_features // head_dim

            # 计算保留的 indices
            all_kv_indices = set(range(num_kv_heads))
            keep_kv_indices = sorted(list(all_kv_indices - set(attn_prune_indices)))

            # 从权重形状获取 Q heads 数量
            q_proj_out_features = layer.self_attn.q_proj.out_features
            old_q = q_proj_out_features // head_dim
            old_kv = num_kv_heads

            if len(keep_kv_indices) == 0:
                # 所有heads都被剪枝，替换为 ZeroAttention
                # 利用残差连接：hidden = hidden + 0 = hidden（跳过Attention）
                log(f"  ⚠️ Attention 被完全剪空（{old_q}Q:{old_kv}KV → 0），替换为 ZeroAttention")
                # 传入模型类型以确保返回值格式正确（Mistral 等模型有特殊格式）
                layer.self_attn = ZeroAttention(model_type=model.config.model_type)
                pruning_stats['attention'][layer_idx] = (old_kv, 0)
            else:
                # 执行部分剪枝
                new_q, new_kv = prune_attention_by_gqa_groups(
                    layer,
                    keep_kv_indices,
                    head_dim=head_dim,
                    gqa_ratio=gqa_ratio
                )
                log(f"  Attention: {old_q}Q:{old_kv}KV → {new_q}Q:{new_kv}KV")
                pruning_stats['attention'][layer_idx] = (old_kv, new_kv)

        # ========== MLP 剪枝 ==========
        mlp_prune_indices = prune_info['mlp']

        if len(mlp_prune_indices) > 0:
            intermediate_size = layer.mlp.gate_proj.out_features

            # 计算保留的 indices
            all_mlp_indices = set(range(intermediate_size))
            keep_mlp_indices = sorted(list(all_mlp_indices - set(mlp_prune_indices)))

            # 最小维度阈值：小于等于此值时替换为 ZeroMLP
            # 原因：intermediate_size=1 时存在数值不稳定和内存布局问题
            MIN_MLP_DIM = 1

            if len(keep_mlp_indices) <= MIN_MLP_DIM:
                # 维度过小，替换为 ZeroMLP
                # 利用残差连接：hidden = hidden + 0 = hidden（跳过MLP）
                log(f"  ⚠️ MLP 维度过小（{intermediate_size} → {len(keep_mlp_indices)} channels），替换为 ZeroMLP")
                layer.mlp = ZeroMLP()
                pruning_stats['mlp'][layer_idx] = (intermediate_size, 0)
            else:
                # 执行部分 MLP 剪枝
                keep_mlp_indices_tensor = torch.tensor(keep_mlp_indices, device=layer.mlp.gate_proj.weight.device)

                # 剪枝 gate_proj 和 up_proj（保留对应的行）
                # 重要：使用 .contiguous() 确保内存连续，避免 SDPA 等操作报错
                layer.mlp.gate_proj.weight = torch.nn.Parameter(
                    layer.mlp.gate_proj.weight[keep_mlp_indices_tensor, :].contiguous()
                )
                # 剪枝 gate_proj bias（如果存在，用于 Qwen2.5 等模型）
                if layer.mlp.gate_proj.bias is not None:
                    layer.mlp.gate_proj.bias = torch.nn.Parameter(
                        layer.mlp.gate_proj.bias[keep_mlp_indices_tensor].contiguous()
                    )


                layer.mlp.up_proj.weight = torch.nn.Parameter(
                    layer.mlp.up_proj.weight[keep_mlp_indices_tensor, :].contiguous()
                )
                # 剪枝 up_proj bias（如果存在）
                if layer.mlp.up_proj.bias is not None:
                    layer.mlp.up_proj.bias = torch.nn.Parameter(
                        layer.mlp.up_proj.bias[keep_mlp_indices_tensor].contiguous()
                    )

                # 剪枝 down_proj（保留对应的列）
                layer.mlp.down_proj.weight = torch.nn.Parameter(
                    layer.mlp.down_proj.weight[:, keep_mlp_indices_tensor].contiguous()
                )
                # down_proj bias 不需要剪枝（只剪了输入维度，输出维度不变）
                # if layer.mlp.down_proj.bias is not None:
                #     # down_proj.bias 不需要剪枝

                # 更新 intermediate_size
                new_intermediate_size = len(keep_mlp_indices)
                layer.mlp.gate_proj.out_features = new_intermediate_size
                layer.mlp.up_proj.out_features = new_intermediate_size
                layer.mlp.down_proj.in_features = new_intermediate_size

                log(f"  MLP: {intermediate_size} → {new_intermediate_size} channels")
                pruning_stats['mlp'][layer_idx] = (intermediate_size, new_intermediate_size)

        # 检查是否整层被剪空
        attn_empty = (layer_idx in pruning_stats['attention'] and
                     pruning_stats['attention'][layer_idx][1] == 0)
        mlp_empty = (layer_idx in pruning_stats['mlp'] and
                    pruning_stats['mlp'][layer_idx][1] == 0)

        if attn_empty and mlp_empty:
            log(f"  🔴 Layer {layer_idx} 被完全剪空（自动深度剪枝）")
            pruning_stats['empty_layers'].append(layer_idx)

    return pruning_stats


def remove_empty_layers(model, empty_layers, logger=None):
    """
    "移除"被完全剪空的层 - 通过替换为 identity 层

    注意：由于 HuggingFace Transformers 的内部实现可能在多处假设层数固定，
    完全删除层可能导致 "list index out of range" 错误。
    因此我们采用更安全的策略：将空层替换为简单的 pass-through 层。

    Args:
        model: 模型
        empty_layers: 要移除的层索引列表
        logger: 日志记录器
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    if len(empty_layers) == 0:
        log("\n✓ 没有层被完全剪空，跳过层移除")
        return

    log(f"\n{'='*60}")
    log(f"移除完全剪空的层")
    log(f"{'='*60}")
    log(f"要移除的层: {empty_layers}")
    log(f"策略: 替换为 Identity 层（保持模型结构完整）")

    # 为了避免 HuggingFace Transformers 内部的各种假设被打破
    # 我们不删除层，而是将它们替换为全局定义的 IdentityDecoderLayer
    num_layers = len(model.model.layers)

    # 替换空层为 identity 层
    for layer_idx in empty_layers:
        if layer_idx < num_layers:
            log(f"  替换 Layer {layer_idx} 为 Identity 层")
            # 获取原始层和配置，以便复制必要的属性（如 Qwen2 的 attention_type）
            original_layer = model.model.layers[layer_idx]
            model.model.layers[layer_idx] = IdentityDecoderLayer(
                original_layer=original_layer,
                config=model.config,
                layer_idx=layer_idx
            )

    log(f"✓ 已替换 {len(empty_layers)} 层为 Identity 层")
    log(f"  物理层数: {num_layers} (保持不变)")
    log(f"  有效层数: {num_layers - len(empty_layers)}")

    # 确保模型在正确的设备上并处于eval模式
    device = next(model.parameters()).device
    model.eval()

    log(f"✓ 模型状态已刷新")

    # 验证模型是否可以正常forward（使用一个小的dummy输入）
    try:
        with torch.no_grad():
            dummy_input = torch.randint(0, 1000, (1, 10)).to(device)
            _ = model(dummy_input)
        log(f"✓ 模型forward验证通过")
    except Exception as e:
        log(f"⚠️  模型forward验证失败: {e}")
        log(f"   这可能会导致后续PPL计算出错")
        import traceback
        log(f"   错误详情:\n{traceback.format_exc()}")


def auto_collapse(model, pruning_stats, collapse_threshold=0.15, logger=None):
    """
    自动坍缩：检测稀疏层并强制移除整层

    H-GSP 核心思想：避免"留 10% 不如不留"的情况
    当某层的参数保留率低于阈值时，直接移除整层

    Args:
        model: 模型
        pruning_stats: 剪枝统计信息
            {'attention': {layer_idx: (old, new)}, 'mlp': {layer_idx: (old, new)}, 'empty_layers': []}
        collapse_threshold: 坍缩阈值（默认 0.15 = 15%）
        logger: 日志记录器

    Returns:
        additional_empty_layers: 需要额外移除的层列表
    """
    def log(msg):
        if logger:
            logger.log(msg)
        else:
            print(msg)

    log(f"\n{'='*60}")
    log(f"自动坍缩检测 (Auto-Collapse)")
    log(f"{'='*60}")
    log(f"坍缩阈值: {collapse_threshold:.1%}")
    log(f"检测逻辑: 当层参数保留率 < {collapse_threshold:.1%} 时，强制移除整层")

    num_layers = len(model.model.layers)
    additional_empty_layers = []

    for layer_idx in range(num_layers):
        # 跳过已经被剪空的层
        if layer_idx in pruning_stats.get('empty_layers', []):
            continue

        # 计算该层的参数保留率
        attn_info = pruning_stats['attention'].get(layer_idx)
        mlp_info = pruning_stats['mlp'].get(layer_idx)

        # 计算 Attention 保留率
        if attn_info:
            old_kv, new_kv = attn_info
            attn_retain_rate = new_kv / old_kv if old_kv > 0 else 1.0
        else:
            attn_retain_rate = 1.0

        # 计算 MLP 保留率
        if mlp_info:
            old_channels, new_channels = mlp_info
            mlp_retain_rate = new_channels / old_channels if old_channels > 0 else 1.0
        else:
            mlp_retain_rate = 1.0

        # 计算综合保留率（取两者的平均）
        avg_retain_rate = (attn_retain_rate + mlp_retain_rate) / 2.0

        # 判断是否触发坍缩
        if avg_retain_rate < collapse_threshold:
            log(f"  🔻 Layer {layer_idx} 触发坍缩:")
            log(f"     Attn 保留率: {attn_retain_rate:.1%}, MLP 保留率: {mlp_retain_rate:.1%}")
            log(f"     平均保留率: {avg_retain_rate:.1%} < {collapse_threshold:.1%}")
            log(f"     决策: 强制移除整层")
            additional_empty_layers.append(layer_idx)

    if len(additional_empty_layers) == 0:
        log(f"\n✓ 没有层触发坍缩阈值")
    else:
        log(f"\n✓ 检测到 {len(additional_empty_layers)} 层需要坍缩: {additional_empty_layers}")
        log(f"  这些层将被强制移除（利用残差悖论）")

    return additional_empty_layers


def main():
    parser = argparse.ArgumentParser(description='H-GSP: Hybrid Global Structural Pruning for LLaMA')

    # 核心参数
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径')
    parser.add_argument('--output_name', type=str, required=True,
                       help='输出目录名称，所有结果保存在 results/{output_name}/ 下')

    # 剪枝参数
    parser.add_argument('--pruning_ratio', type=float, default=0.2,
                       help='目标剪枝率（默认: 0.2）')
    parser.add_argument('--importance_method', type=str, default='taylor',
                       choices=['taylor', 'wanda', 'taylor_2nd', 'magnitude'],
                       help='重要性计算方法（默认: taylor）')
    parser.add_argument('--dataset', type=str, default='c4',
                       choices=['wikitext2', 'ptb', 'c4', 'wikitext_zh', 'c4_zh'],
                       help='校准数据集选择（默认: c4\n'
                            '  英文: wikitext2, ptb, c4\n'
                            '  中文: wikitext_zh, c4_zh (推荐用于 Qwen/ChatGLM 等中文模型)')
    parser.add_argument('--gradient_batch_size', type=int, default=8,
                       help='梯度计算批次大小（默认: 8）')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                       help='使用梯度检查点节省显存')

    # H-GSP 核心参数
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='H-GSP 温度参数 T,当temperature为0时表示只用全局Taylor（默认: 1.0，推荐范围: 0.5-2.0）')
    parser.add_argument('--tau', type=float, default=-100,
                       help='H-GSP 门控阈值 τ（默认: None 自动计算25分位数）\n'
                            '  - tau=-100: 纯 Block-wise 模式（只用块级重要性）\n'
                            '  - tau=inf: 纯 Layer-wise 模式（只用层级重要性）\n'
                            '  - tau=None: 自动模式（推荐，根据数据自适应）')
    parser.add_argument('--epsilon', type=float, default=0,
                       help='H-GSP 坍缩阈值 ε（默认: 0）')

    # 层冻结参数
    parser.add_argument('--freeze_first_n_layers', type=int, default=0,
                       help='冻结前N层不剪枝（默认: 0）')
    parser.add_argument('--freeze_last_n_layers', type=int, default=0,
                       help='冻结后N层不剪枝（默认: 0）')

    # H-GSP 内部参数（用于调试和优化）
    parser.add_argument('--taylor_num_samples', type=int, default=128,
                       help='Taylor 重要性计算的样本数（默认: 128）')
    parser.add_argument('--taylor_seq_len', type=int, default=128,
                       help='Taylor 重要性计算的序列长度（默认: 128）')
    parser.add_argument('--layer_importance_num_samples', type=int, default=128,
                       help='层重要性分析的样本数（默认: 128）')
    parser.add_argument('--layer_importance_seq_len', type=int, default=128,
                       help='层重要性分析的序列长度（默认: 128）')
    parser.add_argument('--block_importance_num_samples', type=int, default=128,
                       help='块重要性分析的样本数（默认: 128）')
    parser.add_argument('--block_importance_seq_len', type=int, default=128,
                       help='块重要性分析的序列长度（默认: 128）')

    # GQA 配置
    parser.add_argument('--head_dim', type=int, default=128,
                       help='Attention head 维度（默认: 128）')
    parser.add_argument('--gqa_ratio', type=int, default=4,
                       help='Q:KV 比例（默认: 4）')

    # 评估参数
    parser.add_argument('--run_evaluation', type=str, default="ppl, zeroshot",
                       help='评估类型: ppl, zeroshot, efficiency, all（多个用逗号分隔）')
    parser.add_argument('--eval_ppl_datasets', type=str, default='wikitext2,ptb',
                       help='PPL评估数据集（默认: wikitext2,ptb）')
    parser.add_argument('--eval_ppl_seq_len', type=int, default=128,
                       help='PPL评估窗口大小（默认: 128，标准配置: 2048）')
    parser.add_argument('--eval_ppl_stride', type=int, default=None,
                       help='PPL评估步长（默认: None即等于seq_len，标准配置: 512）')
    parser.add_argument('--eval_zeroshot_tasks', type=str, default='boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa',
                       help='Zero-shot评估任务')
    parser.add_argument('--eval_use_custom_zeroshot', action='store_true',
                       help='使用自定义zero-shot评估器（默认False，使用在线评估）')
    # 微调参数（LoRA）
    parser.add_argument('--finetune', action='store_true',
                       help='剪枝后进行 LoRA 微调恢复')
    parser.add_argument('--finetune_data_path', type=str, default='yahma/alpaca-cleaned',
                       help='微调数据集路径（默认: yahma/alpaca-cleaned）')
    parser.add_argument('--finetune_epochs', type=int, default=2,
                       help='微调轮数（默认: 2）')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                       help='微调学习率（默认: 1e-4）')
    parser.add_argument('--finetune_batch_size', type=int, default=64,
                       help='微调 batch size（默认: 64）')
    parser.add_argument('--finetune_micro_batch_size', type=int, default=4,
                       help='微调 micro batch size（默认: 4）')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank（默认: 8）')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha（默认: 16）')
    parser.add_argument('--skip_finetune_evaluation', action='store_true',
                       help='跳过微调后的自动评估')

    # 其他
    from core.utils.get_best_gpu import get_best_gpu
    bestDevice = "cuda:"+str(get_best_gpu())
    parser.add_argument('--device', type=str, default=bestDevice,
                       help='设备')
    parser.add_argument('--layer_start', type=int, default=0,
                       help='起始层（debug用）')
    parser.add_argument('--layer_end', type=int, default=None,
                       help='结束层（debug用）')

    args = parser.parse_args()

    # 设置输出目录为 results/{output_name}
    output_base_dir = os.path.join('results', args.output_name)

    # 创建输出目录结构（先创建，再初始化 logger）
    output_dirs = setup_output_directories(output_base_dir)

    # 设置 logger，日志保存在 logs 子目录下
    logger = LoggerWithDepth(
        env_name='logs',  # 在 logs 子目录下创建时间戳文件夹
        config=args.__dict__,
        root_dir=output_base_dir  # 基础目录是 results/{output_name}
    )
    logger.log(f"\n✓ 输出目录结构已创建:")
    logger.log(f"  基础目录: {output_dirs['base']}")
    logger.log(f"  模型保存: {output_dirs['models']}")
    logger.log(f"  分析结果: {output_dirs['analysis']}")
    logger.log(f"  评估结果: {output_dirs['evaluation']}")
    logger.log(f"  剪枝可视化结果: {output_dirs['visualization']}")
    logger.log(f"  日志文件: {output_dirs['logs']}")

    logger.log("\n" + "="*60)
    logger.log("基于全局性价比的混合结构化剪枝 (H-GSP)")
    logger.log("="*60)
    logger.log(f"模型: {args.base_model}")
    logger.log(f"剪枝率: {args.pruning_ratio:.1%}")
    logger.log(f"重要性方法: {args.importance_method}")
    logger.log(f"数据集: {args.dataset}")
    logger.log(f"\nH-GSP 参数:")
    logger.log(f"  温度 T: {args.temperature}")
    logger.log(f"  阈值 τ: {'自动计算' if args.tau is None else args.tau}")
    logger.log(f"  坍缩阈值 ε: {args.epsilon}")

    # ========== Step 1: 加载模型 ==========
    logger.log("\n[Step 1] 加载模型...")

    # 根据设备选择加载方式
    if 'cpu' in args.device.lower():
        device_map = args.device
    else:
        # 单GPU：直接指定设备，避免多GPU分布
        device_map = args.device

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )

    # 加载 tokenizer，处理 sentencepiece 兼容性问题
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    except (ValueError, OSError) as e:
        if "sentencepiece" in str(e).lower():
            logger.log("  ⚠️  Fast tokenizer 需要 sentencepiece，尝试使用 slow tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
            except Exception as e2:
                logger.log(f"  ❌ Slow tokenizer 也失败，请安装: pip install sentencepiece")
                raise e2
        else:
            raise e

    # 启用梯度检查点（节省显存）
    if args.use_gradient_checkpointing:
        logger.log("  启用梯度检查点（Gradient Checkpointing）...")
        model.gradient_checkpointing_enable()

    # 获取实际使用的设备
    if hasattr(model, 'hf_device_map'):
        logger.log(f"  模型分布: {model.hf_device_map}")
        # 获取第一个模块的设备（输入数据应该发送到这里）
        first_device = next(iter(model.hf_device_map.values()))
        args.device = f'cuda:{first_device}' if isinstance(first_device, int) else first_device
        logger.log(f"  输入设备: {args.device}")
    else:
        args.device = next(model.parameters()).device

    # 自动检测模型的 GQA 配置
    num_q_heads, num_kv_heads, detected_gqa_ratio, detected_head_dim = get_model_gqa_config(model)

    logger.log(f"\n检测到的模型配置:")
    logger.log(f"  模型类型: {model.config.model_type}")
    logger.log(f"  Q Heads: {num_q_heads}")
    logger.log(f"  KV Heads: {num_kv_heads}")
    logger.log(f"  GQA Ratio: {detected_gqa_ratio}:1")
    logger.log(f"  Head Dim: {detected_head_dim}")

    # 自动更新配置（如果与命令行参数不同，使用检测到的值）
    if args.gqa_ratio != detected_gqa_ratio:
        logger.log(f"\n⚠️  命令行指定 gqa_ratio={args.gqa_ratio}，但检测到 {detected_gqa_ratio}")
        logger.log(f"  将使用检测到的值: {detected_gqa_ratio}")
        args.gqa_ratio = detected_gqa_ratio

    if args.head_dim != detected_head_dim:
        logger.log(f"\n⚠️  命令行指定 head_dim={args.head_dim}，但检测到 {detected_head_dim}")
        logger.log(f"  将使用检测到的值: {detected_head_dim}")
        args.head_dim = detected_head_dim

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"\n✓ 模型加载完成")
    logger.log(f"  总参数量: {total_params:,}")

    # 显示GPU显存使用情况
    if torch.cuda.is_available() and 'cuda' in str(args.device).lower():
        device_str = str(args.device)
        gpu_id = int(device_str.split(':')[-1]) if ':' in device_str else 0
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.log(f"  GPU 显存: {allocated:.2f}GB / {total_mem:.2f}GB (已分配)")
        logger.log(f"  GPU 显存: {reserved:.2f}GB / {total_mem:.2f}GB (已预留)")

    # 分析原始模型（剪枝前）
    logger.log(f"\n分析原始模型结构...")
    original_analyzer = ModelAnalyzer(model, "原始模型")
    original_analysis = original_analyzer.analyze()
    logger.log(f"  ✓ 原始模型分析完成")

    # 创建数据集管理器（统一管理所有数据集加载）
    logger.log(f"\n✓ 初始化数据集管理器: {args.dataset}")
    dataset_manager = DatasetManager(dataset_name=args.dataset, tokenizer=tokenizer)

    # ========== Step 3: 计算重要性（梯度或激活） ==========
    activations = None
    hessian_diag = None

    # H-GSP 内部参数（从命令行参数获取）
    TAYLOR_NUM_SAMPLES = args.taylor_num_samples
    TAYLOR_SEQ_LEN = args.taylor_seq_len
    LAYER_IMPORTANCE_NUM_SAMPLES = args.layer_importance_num_samples
    LAYER_IMPORTANCE_SEQ_LEN = args.layer_importance_seq_len
    BLOCK_IMPORTANCE_NUM_SAMPLES = args.block_importance_num_samples
    BLOCK_IMPORTANCE_SEQ_LEN = args.block_importance_seq_len

    if args.importance_method in ['taylor', 'taylor_2nd']:
        logger.log(f"\n[Step 3] 计算梯度（{'一阶' if args.importance_method == 'taylor' else '二阶'} Taylor importance）...")
        logger.log(f"  样本数: {TAYLOR_NUM_SAMPLES}, 序列长度: {TAYLOR_SEQ_LEN}")

        # 初始化梯度分析器
        gradient_analyzer = GradientAnalyzer(model, logger)
        logger.log(f"  ✓ 梯度分析器已初始化（将收集梯度统计用于诊断）")

        # 分批计算梯度以节省内存
        batch_size = args.gradient_batch_size
        num_batches = (TAYLOR_NUM_SAMPLES + batch_size - 1) // batch_size
        logger.log(f"  批次大小: {batch_size}, 总批次数: {num_batches}")

        model.zero_grad()
        total_loss = 0.0
        start_time = time.time()

        # 如果在 CPU 上运行，给出提示
        if 'cpu' in str(args.device).lower():
            logger.log(f"  ⚠️ 在 CPU 上运行，速度会非常慢！")
            logger.log(f"  预计每个批次需要 5-10 分钟（取决于 CPU 性能）")
            logger.log(f"  总预计时间: {num_batches * 7:.0f} 分钟左右")
            logger.log("")

        # 二阶泰勒需要累积 Hessian 对角线近似
        # ⚠️ 存储在CPU上以避免GPU OOM
        if args.importance_method == 'taylor_2nd':
            hessian_diag = {}
            logger.log("  初始化 Hessian 对角线存储（在CPU上以节省GPU显存）...")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 存储在CPU上，避免占用GPU显存
                    hessian_diag[name] = torch.zeros_like(param.data, device='cpu')

        # ✅ 修复：一次性加载所有样本，避免每批次重复获取相同样本
        logger.log(f"  加载 {TAYLOR_NUM_SAMPLES} 个样本用于梯度计算...")
        all_gradient_samples = dataset_manager.get_gradient_samples(
            num_samples=TAYLOR_NUM_SAMPLES,
            seq_len=TAYLOR_SEQ_LEN
        )
        logger.log(f"  ✓ 样本加载完成，shape: {all_gradient_samples.shape}")

        # 使用 tqdm 显示进度条
        pbar = tqdm(range(num_batches), desc="计算梯度", ncols=100)

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, TAYLOR_NUM_SAMPLES)
            current_batch_size = end_idx - start_idx

            batch_start_time = time.time()

            # ⚠️ 关键：对于二阶泰勒，每个批次清零梯度以获得独立的梯度
            # 对于一阶泰勒，不清零梯度，让梯度累积
            if args.importance_method == 'taylor_2nd':
                model.zero_grad()

            # ✅ 修复：从预加载的样本中切片获取当前批次
            input_ids = all_gradient_samples[start_idx:end_idx].to(args.device)

            # 前向传播
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss  # ✅ 修复：不除以 num_batches

            # 反向传播
            loss.backward()

            # 🔍 诊断：打印第一个batch的梯度分布（帮助诊断序列长度问题）
            if batch_idx == 0:
                sample_layers = [0, 2, 10, 20, 31]
                logger.log(f"  梯度分布诊断（序列长度 {TAYLOR_SEQ_LEN}）：")
                for layer_idx in sample_layers:
                    layer_name = f'model.layers.{layer_idx}.mlp.gate_proj.weight'
                    for name, param in model.named_parameters():
                        if name == layer_name and param.grad is not None:
                            grad_mean = param.grad.abs().mean().item()
                            grad_std = param.grad.abs().std().item()
                            logger.log(f"    Layer {layer_idx:2d}: grad_mean={grad_mean:.6e}, grad_std={grad_std:.6e}")
                            break

            # 📊 收集梯度统计（用于后续诊断和可视化）
            gradient_analyzer.accumulate_gradient_stats(layer_prefix='model.layers')

            # 二阶泰勒：累积 Hessian 对角线（使用梯度平方近似）
            if args.importance_method == 'taylor_2nd':
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # ✅ 修复：累加每个批次独立的梯度平方
                        # 注意：这里累加的是梯度平方，不是平方和除以批次数
                        hessian_diag[name] += (param.grad ** 2).cpu()

            # 累加 loss（用于报告平均值）
            batch_time = time.time() - batch_start_time
            total_loss += loss.item()

            # 更新进度条信息
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'time': f'{batch_time:.2f}s'
            })

            # 清理内存
            del input_ids, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        # ✅ 修复：在所有批次完成后，将 Hessian 对角线除以批次数得到平均值
        if args.importance_method == 'taylor_2nd':
            for name in hessian_diag:
                hessian_diag[name] /= num_batches

        total_time = time.time() - start_time
        logger.log(f"✓ 梯度计算完成")
        logger.log(f"  平均 loss: {total_loss/num_batches:.4f}")
        logger.log(f"  总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.log(f"  平均每批次: {total_time/num_batches:.2f}s")

        if args.importance_method == 'taylor_2nd':
            logger.log(f"  ✓ Hessian 对角线近似计算完成")
            logger.log(f"  Hessian 字典包含 {len(hessian_diag)} 个参数")

            # 打印一些示例键名，用于调试
            sample_keys = list(hessian_diag.keys())[:10]
            logger.log(f"  示例 Hessian 键名（前10个）：")
            for key in sample_keys:
                logger.log(f"    - {key}")

            # 检查是否包含预期的键名
            layer_0_keys = [k for k in hessian_diag.keys() if 'layers.0.' in k]
            if layer_0_keys:
                logger.log(f"  Layer 0 的参数示例：")
                for key in layer_0_keys[:5]:
                    logger.log(f"    - {key}")

    elif args.importance_method == 'magnitude':
        logger.log(f"\n[Step 3] 使用 Magnitude importance (权重绝对值)...")
        logger.log(f"  ✓ Magnitude 方法不需要计算梯度或激活值")
        logger.log(f"  直接使用模型权重进行剪枝")

    elif args.importance_method == 'wanda':
        logger.log(f"\n[Step 3] 收集激活值（Wanda importance）...")
        logger.log(f"  样本数: {TAYLOR_NUM_SAMPLES}, 序列长度: {TAYLOR_SEQ_LEN}")

        # 分批收集激活
        batch_size = args.gradient_batch_size
        num_batches = (TAYLOR_NUM_SAMPLES + batch_size - 1) // batch_size
        logger.log(f"  批次大小: {batch_size}, 总批次数: {num_batches}")

        # ✅ 修复：一次性加载所有样本，避免每批次重复获取相同样本
        logger.log(f"  加载 {TAYLOR_NUM_SAMPLES} 个样本用于激活收集...")
        all_gradient_samples = dataset_manager.get_gradient_samples(
            num_samples=TAYLOR_NUM_SAMPLES,
            seq_len=TAYLOR_SEQ_LEN
        )
        logger.log(f"  ✓ 样本加载完成，shape: {all_gradient_samples.shape}")

        all_activations = {}
        start_time = time.time()

        pbar = tqdm(range(num_batches), desc="收集激活", ncols=100)

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, TAYLOR_NUM_SAMPLES)
            current_batch_size = end_idx - start_idx

            batch_start_time = time.time()

            # ✅ 修复：从预加载的样本中切片获取当前批次
            input_ids = all_gradient_samples[start_idx:end_idx].to(args.device)

            # 收集激活
            batch_activations = collect_layer_activations(model, input_ids, args.device)

            # 累加激活值
            for layer_idx, layer_acts in batch_activations.items():
                if layer_idx not in all_activations:
                    all_activations[layer_idx] = {}
                for name, act in layer_acts.items():
                    if name not in all_activations[layer_idx]:
                        all_activations[layer_idx][name] = act.to(args.device)
                    else:
                        all_activations[layer_idx][name] += act.to(args.device)

            batch_time = time.time() - batch_start_time

            # 更新进度条信息
            pbar.set_postfix({'time': f'{batch_time:.2f}s'})

            del input_ids, batch_activations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        # 平均激活值
        for layer_idx in all_activations:
            for name in all_activations[layer_idx]:
                all_activations[layer_idx][name] /= num_batches

        activations = all_activations

        total_time = time.time() - start_time
        logger.log(f"✓ 激活值收集完成")
        logger.log(f"  总耗时: {total_time:.2f}s ({total_time/60:.2f}min)")
        logger.log(f"  平均每批次: {total_time/num_batches:.2f}s")

    # ========== Step 3.5: 计算层移除困惑度（H-GSP 必需）==========
    # ========== Step 3.5: 计算层移除困惑度（H-GSP Layer-wise 重要性）==========
    # 当温度 T=0 时，只使用全局 Taylor 重要性，跳过层级和块级重要性测试
    if args.temperature == 0.0:
        logger.log(f"\n[Step 3.5-3.6] 跳过层级和块级重要性测试")
        logger.log(f"  原因: temperature=0，只使用全局 Taylor 重要性（推荐配置）")
        logger.log(f"  ✓ 避免模型兼容性问题，聚焦核心方法")

        # 设置为空，后续构建全局分析表时会自动处理
        layer_removal_ppl = {}
        block_removal_ppl = {'attention': {}, 'mlp': {}}

    else:
        logger.log(f"\n[Step 3.5] 计算层重要性（H-GSP Layer-wise 重要性）...")
        logger.log(f"  样本数: {LAYER_IMPORTANCE_NUM_SAMPLES}, 序列长度: {LAYER_IMPORTANCE_SEQ_LEN}")

        from core.importance.layer_analyzer import LayerImportanceAnalyzer

        # 加载用于层重要性分析的样本（文本格式）
        layer_texts_list = dataset_manager.get_layer_importance_samples(
            num_samples=LAYER_IMPORTANCE_NUM_SAMPLES,
            seq_len=LAYER_IMPORTANCE_SEQ_LEN
        )

        # 创建分析器
        analyzer = LayerImportanceAnalyzer(model, tokenizer, device=args.device)

        # 计算每层的重要性（使用loss增加值方法）
        num_layers = len(model.model.layers)
        layer_removal_ppl = analyzer.measure_layer_importance_by_removal(
            texts=layer_texts_list,
            num_layers=num_layers
        )

        logger.log(f"✓ 层重要性计算完成（loss增加值方法）")
        print("\n" + "="*60)
        print("层级重要度（移除层后的loss增加值）")
        print("="*60)
        for layer_idx in range(num_layers):
            importance = layer_removal_ppl.get(layer_idx, 0.0)
            print(f"Layer {layer_idx:2d}   {importance:10.4f}")

        # 保存层重要性到分析目录
        import json
        layer_importance_path = os.path.join(output_dirs['analysis'], 'layer_importance_loss.json')
        with open(layer_importance_path, 'w') as f:
            json.dump(layer_removal_ppl, f, indent=2)
        logger.log(f"✓ 层重要性已保存: {layer_importance_path}")

        # ========== Step 3.6: 计算块重要性（H-GSP Block-wise 重要性）==========
        logger.log(f"\n[Step 3.6] 计算块重要性（H-GSP Block-wise 重要性）...")
        logger.log(f"  方法: 基于loss增加值（移除块后的loss变化）")
        logger.log(f"  样本数: {BLOCK_IMPORTANCE_NUM_SAMPLES}, 序列长度: {BLOCK_IMPORTANCE_SEQ_LEN}")

        # 加载用于块重要性分析的样本（文本格式）
        block_texts_list = dataset_manager.get_layer_importance_samples(
            num_samples=BLOCK_IMPORTANCE_NUM_SAMPLES,
            seq_len=BLOCK_IMPORTANCE_SEQ_LEN
        )

        # 计算每层的 Attention 和 MLP 块重要性（使用loss增加值方法）
        block_removal_ppl = analyzer.measure_block_importance_by_removal(
            texts=block_texts_list,
            num_layers=num_layers
        )

        logger.log(f"✓ 块重要性计算完成（loss增加值方法）")
        logger.log(f"  示例 - Layer 0 Attention: {block_removal_ppl['attention'][0]:.4f}, MLP: {block_removal_ppl['mlp'][0]:.4f}")
        logger.log(f"  示例 - Layer {num_layers-1} Attention: {block_removal_ppl['attention'][num_layers-1]:.4f}, MLP: {block_removal_ppl['mlp'][num_layers-1]:.4f}")

        # 保存块重要性到分析目录
        block_importance_path = os.path.join(output_dirs['analysis'], 'block_importance_loss.json')
        with open(block_importance_path, 'w') as f:
            json.dump(block_removal_ppl, f, indent=2)
        logger.log(f"✓ 块重要性已保存: {block_importance_path}")

    # ========== Step 3.7: 梯度诊断和可视化（仅在使用 Taylor 方法时）==========
    if args.importance_method in ['taylor', 'taylor_2nd']:
        logger.log(f"\n[Step 3.7] 梯度诊断和可视化...")

        num_layers = len(model.model.layers)

        # 保存梯度统计到文件
        gradient_stats_path = os.path.join(output_dirs['analysis'], 'gradient_statistics.json')
        gradient_analyzer.save_gradient_stats(gradient_stats_path)

        # 注意：此时还没有计算重要性得分和剪枝率，所以先不生成完整的可视化
        # 完整的可视化将在剪枝完成后生成
        logger.log(f"  ✓ 梯度统计已收集并保存")
        logger.log(f"  ℹ️  完整的梯度可视化将在剪枝完成后生成")

    # ========== Step 4: 构建全局分析表 ==========
    logger.log("\n[Step 4] 构建全局 Group 分析表...")

    layer_end = args.layer_end if args.layer_end else len(model.model.layers)

    # 传递重要性信息
    importance_info = {}
    if args.importance_method in ['taylor', 'taylor_2nd']:
        importance_info['gradients'] = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        if args.importance_method == 'taylor_2nd':
            importance_info['hessian_diag'] = hessian_diag
    elif args.importance_method == 'wanda':
        importance_info['activations'] = activations

    df = build_global_group_table(
        model=model,
        importance_method=args.importance_method,
        importance_info=importance_info,
        layer_start=args.layer_start,
        layer_end=layer_end,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
        device=args.device,
        layer_removal_ppl=layer_removal_ppl,    # H-GSP: 层级重要性
        block_removal_ppl=block_removal_ppl,    # H-GSP: 块级重要性
        temperature=args.temperature,           # H-GSP: 温度参数 T
        tau=args.tau,                          # H-GSP: 门控阈值 τ
        freeze_first_n_layers=args.freeze_first_n_layers,  # 冻结前N层
        freeze_last_n_layers=args.freeze_last_n_layers     # 冻结后N层
    )

    logger.log(f"✓ 分析表构建完成")

    # ========== Step 5: 选择要剪枝的 groups ==========
    logger.log(f"\n[Step 5] 根据剪枝率选择要剪枝的 groups...")

    groups_to_prune = select_groups_to_prune(
        df=df,
        pruning_ratio=args.pruning_ratio,
        total_params=total_params
    )

    logger.log(f"✓ 选中 {len(groups_to_prune)} 个 groups 进行剪枝")

    # 保存分析表到 analysis 目录（按score排序）
    table_path = os.path.join(output_dirs['analysis'], 'global_group_table.csv')
    df.to_csv(table_path, index=False)
    logger.log(f"✓ 分析表已保存（按score排序）: {table_path}")

    prune_table_path = os.path.join(output_dirs['analysis'], 'groups_to_prune.csv')
    groups_to_prune.to_csv(prune_table_path, index=False)
    logger.log(f"✓ 剪枝列表已保存（按score排序）: {prune_table_path}")

    # 保存按层排序的分析表
    df_by_layer = df.sort_values(['layer_idx', 'group_type', 'group_idx']).reset_index(drop=True)
    table_by_layer_path = os.path.join(output_dirs['analysis'], 'global_group_table_by_layer.csv')
    df_by_layer.to_csv(table_by_layer_path, index=False)
    logger.log(f"✓ 分析表已保存（按层排序）: {table_by_layer_path}")

    # 保存按层排序的剪枝列表
    prune_by_layer = groups_to_prune.sort_values(['layer_idx', 'group_type', 'group_idx']).reset_index(drop=True)
    prune_by_layer_path = os.path.join(output_dirs['analysis'], 'groups_to_prune_by_layer.csv')
    prune_by_layer.to_csv(prune_by_layer_path, index=False)
    logger.log(f"✓ 剪枝列表已保存（按层排序）: {prune_by_layer_path}")

    # 生成层级统计摘要
    summary_lines = []
    summary_lines.append("="*80)
    summary_lines.append("各层剪枝统计摘要")
    summary_lines.append("="*80)
    summary_lines.append(f"{'Layer':<8} {'Attention剪枝':<20} {'MLP剪枝':<20} {'总参数剪枝':<20}")
    summary_lines.append("-"*80)

    for layer_idx in sorted(groups_to_prune['layer_idx'].unique()):
        layer_data = groups_to_prune[groups_to_prune['layer_idx'] == layer_idx]
        attn_data = layer_data[layer_data['group_type'] == 'attention']
        mlp_data = layer_data[layer_data['group_type'] == 'mlp']

        attn_count = len(attn_data)
        mlp_count = len(mlp_data)
        attn_params = attn_data['cost'].sum() if len(attn_data) > 0 else 0
        mlp_params = mlp_data['cost'].sum() if len(mlp_data) > 0 else 0
        total_params_prune = attn_params + mlp_params

        summary_lines.append(
            f"{layer_idx:<8} "
            f"{attn_count} groups ({attn_params:,} params)".ljust(20) + " "
            f"{mlp_count} channels ({mlp_params:,} params)".ljust(20) + " "
            f"{total_params_prune:,} params".ljust(20)
        )

    summary_lines.append("-"*80)
    summary_lines.append(f"总计: {len(groups_to_prune)} groups, "
                        f"{groups_to_prune['cost'].sum():,} params")
    summary_lines.append("="*80)

    # 保存摘要文件到 analysis 目录
    summary_path = os.path.join(output_dirs['analysis'], 'pruning_summary_by_layer.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    logger.log(f"✓ 层级统计摘要已保存: {summary_path}")

    # 也在日志中显示
    logger.log("\n" + '\n'.join(summary_lines))

    # ========== Step 6: 执行全局剪枝 ==========
    logger.log(f"\n[Step 6] 执行全局剪枝...")

    pruning_stats = apply_global_pruning(
        model=model,
        groups_to_prune_df=groups_to_prune,
        head_dim=args.head_dim,
        gqa_ratio=args.gqa_ratio,
        logger=logger
    )

    logger.log("\n✓ 全局剪枝完成")

    # ========== Step 6.5: 自动坍缩（H-GSP 必需）==========
    logger.log(f"\n[Step 6.5] 自动坍缩检测（H-GSP Auto-Collapse, ε={args.epsilon}）...")
    additional_empty_layers = auto_collapse(
        model=model,
        pruning_stats=pruning_stats,
        collapse_threshold=args.epsilon,
        logger=logger
    )
    # 将额外的空层加入到 empty_layers 列表
    pruning_stats['empty_layers'].extend(additional_empty_layers)

    # ========== Step 7: 移除空层（自动执行）==========
    # 注：既然 Auto-Collapse 已检测到稀疏层，应自动替换为 Identity 层
    # 这符合 H-GSP 的核心理念："留 10% 不如不留"
    all_empty_layers = pruning_stats['empty_layers']
    if len(all_empty_layers) > 0:
        logger.log(f"\n[Step 7] 移除空层...")
        logger.log(f"  原始空层: {len(all_empty_layers) - len(additional_empty_layers)}")
        if len(additional_empty_layers) > 0:
            logger.log(f"  坍缩触发: {len(additional_empty_layers)}")
        logger.log(f"  总计移除: {len(all_empty_layers)} 层")
        remove_empty_layers(model, all_empty_layers, logger)
    else:
        logger.log(f"\n[Step 7] ✓ 无需移除空层")

    # ========== Step 8: 统计剪枝结果 ==========
    logger.log(f"\n{'='*60}")
    logger.log(f"剪枝统计")
    logger.log(f"{'='*60}")

    after_params = sum(p.numel() for p in model.parameters())
    actual_ratio = (total_params - after_params) / total_params

    logger.log(f"参数统计:")
    logger.log(f"  剪枝前: {total_params:,}")
    logger.log(f"  剪枝后: {after_params:,}")
    logger.log(f"  实际剪枝率: {actual_ratio:.2%}")

    if len(pruning_stats['empty_layers']) > 0:
        logger.log(f"\n自动深度剪枝:")
        logger.log(f"  替换为Identity的层: {pruning_stats['empty_layers']}")
        logger.log(f"  物理层数: {len(model.model.layers)} (保持不变)")
        logger.log(f"  有效层数: {len(model.model.layers) - len(pruning_stats['empty_layers'])}")

    # ========== Step 8.5: 生成详细的模型分析报告 ==========
    logger.log(f"\n[Step 8.5] 生成详细的模型分析报告...")

    # 分析剪枝后的模型
    pruned_analyzer = ModelAnalyzer(model, "剪枝后模型")
    pruned_analysis = pruned_analyzer.analyze()
    logger.log(f"  ✓ 剪枝后模型分析完成")

    # 生成对比报告
    comparator = ModelComparator(
        original_analysis=original_analysis,
        pruned_analysis=pruned_analysis,
        original_name="原始模型",
        pruned_name="剪枝后模型"
    )
    comparison_result = comparator.compare()
    logger.log(f"  ✓ 对比分析完成")

    # 保存分析报告
    import json
    analysis_dir = output_dirs['analysis']

    # 保存原始模型分析
    original_analysis_path = os.path.join(analysis_dir, 'original_model_analysis.json')
    with open(original_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(original_analysis, f, indent=2, ensure_ascii=False)
    logger.log(f"  ✓ 原始模型分析已保存: {original_analysis_path}")

    # 保存剪枝后模型分析
    pruned_analysis_path = os.path.join(analysis_dir, 'pruned_model_analysis.json')
    with open(pruned_analysis_path, 'w', encoding='utf-8') as f:
        json.dump(pruned_analysis, f, indent=2, ensure_ascii=False)
    logger.log(f"  ✓ 剪枝后模型分析已保存: {pruned_analysis_path}")

    # 保存对比报告
    comparison_path = os.path.join(analysis_dir, 'model_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    logger.log(f"  ✓ 对比报告已保存: {comparison_path}")

    # 同时保存为 pruning_comparison.json（兼容可视化工具）
    pruning_comparison_path = os.path.join(analysis_dir, 'pruning_comparison.json')
    with open(pruning_comparison_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_result, f, indent=2, ensure_ascii=False)
    logger.log(f"  ✓ 剪枝对比数据已保存: {pruning_comparison_path}")

    # 在日志中打印详细的对比报告
    logger.log(f"\n{'='*60}")
    logger.log(f"详细对比报告")
    logger.log(f"{'='*60}")

    total = comparison_result['total_params']
    logger.log(f"\n总参数量:")
    logger.log(f"  原始: {total['original']:,}")
    logger.log(f"  剪枝后: {total['pruned']:,}")
    logger.log(f"  减少: {total['reduced']:,} ({total['reduction_ratio']*100:.2f}%)")

    layer_params = comparison_result['layer_params']
    logger.log(f"\nDecoder Layers 参数:")
    logger.log(f"  原始: {layer_params['original']:,}")
    logger.log(f"  剪枝后: {layer_params['pruned']:,}")
    logger.log(f"  减少: {layer_params['reduced']:,} ({layer_params['reduction_ratio']*100:.2f}%)")

    # 统计各层剪枝情况
    logger.log(f"\n每层剪枝详情:")
    logger.log(f"{'-'*60}")

    for layer_comp in comparison_result['layers']:
        layer_idx = layer_comp['layer_idx']
        total_comp = layer_comp['total']
        attn_comp = layer_comp['attention']
        mlp_comp = layer_comp['mlp']

        # 标记特殊层
        special_marker = ""
        if layer_comp['is_zero_layer']:
            special_marker = " [完全剪空]"

        logger.log(f"\nLayer {layer_idx:2d}{special_marker}:")
        logger.log(f"  总参数: {total_comp['original']:,} → {total_comp['pruned']:,} "
                  f"(-{total_comp['reduction_ratio']*100:.2f}%)")

        logger.log(f"  Attention: {attn_comp['original']:,} → {attn_comp['pruned']:,} "
                  f"(-{attn_comp['reduction_ratio']*100:.2f}%)")
        if 'num_heads' in attn_comp:
            orig_q = attn_comp['num_heads']['original']
            pruned_q = attn_comp['num_heads']['pruned']
            orig_kv = attn_comp['num_kv_heads']['original']
            pruned_kv = attn_comp['num_kv_heads']['pruned']
            logger.log(f"    头数: {orig_q}Q:{orig_kv}KV → {pruned_q}Q:{pruned_kv}KV")

        logger.log(f"  MLP: {mlp_comp['original']:,} → {mlp_comp['pruned']:,} "
                  f"(-{mlp_comp['reduction_ratio']*100:.2f}%)")
        if 'intermediate_size' in mlp_comp:
            orig_size = mlp_comp['intermediate_size']['original']
            pruned_size = mlp_comp['intermediate_size']['pruned']
            logger.log(f"    中间维度: {orig_size} → {pruned_size}")

    # 统计完全剪空的层
    zero_layers = [l['layer_idx'] for l in comparison_result['layers'] if l['is_zero_layer']]
    if zero_layers:
        logger.log(f"\n完全剪空的层 ({len(zero_layers)}个): {zero_layers}")

    logger.log(f"\n{'='*60}")

    # ========== Step 8.6: 梯度诊断和可视化（完整版）==========
    if args.importance_method in ['taylor', 'taylor_2nd'] and 'gradient_analyzer' in locals():
        logger.log(f"\n[Step 8.6] 生成梯度诊断和可视化报告...")

        num_layers = len(model.model.layers)

        # 从 comparison_result 中提取每层的剪枝率
        layer_pruning_rates = {}
        for layer_comp in comparison_result['layers']:
            layer_idx = layer_comp['layer_idx']
            # 使用 MLP 的剪枝率作为层剪枝率的代表
            layer_pruning_rates[layer_idx] = layer_comp['mlp'].get('reduction_ratio', 0.0)

        # 从 df (global_analysis_table) 中提取重要性得分
        # 注意：这里我们需要从 group table 中聚合得到每层的平均重要性
        layer_importance_scores = {}

        # 检查 df 是否存在并且不为空
        if 'df' in locals() and df is not None and not df.empty:
            for layer_idx in range(num_layers):
                # 收集该层所有 MLP groups 的重要性
                # 注意：DataFrame 列名是 'group_type'，值是 'mlp'
                layer_groups = df[(df['group_type'] == 'mlp') & (df['layer_idx'] == layer_idx)]

                if not layer_groups.empty:
                    layer_importance_scores[layer_idx] = layer_groups['importance'].mean()
                else:
                    layer_importance_scores[layer_idx] = 0.0
        else:
            # 如果 df 不存在，使用默认值
            logger.log(f"  ⚠️  无法提取重要性得分（df 不存在），将使用默认值")
            for layer_idx in range(num_layers):
                layer_importance_scores[layer_idx] = 1.0

        # 生成完整的梯度可视化（包括重要性和剪枝率对比）
        visualization_dir = output_dirs['visualization']
        gradient_analyzer.visualize_gradient_distribution(
            num_layers=num_layers,
            save_dir=visualization_dir,
            importance_scores=layer_importance_scores,
            pruning_rates=layer_pruning_rates
        )

        # 生成诊断报告
        diagnosis_report = gradient_analyzer.diagnose_extreme_pruning(
            num_layers=num_layers,
            importance_scores=layer_importance_scores,
            pruning_rates=layer_pruning_rates,
            threshold=0.5  # 剪枝率超过 50% 视为极端
        )

        # 打印诊断报告
        gradient_analyzer.print_diagnosis_report(diagnosis_report)

        # 保存诊断报告
        diagnosis_path = os.path.join(output_dirs['analysis'], 'gradient_diagnosis.json')
        with open(diagnosis_path, 'w') as f:
            json.dump(diagnosis_report, f, indent=2)
        logger.log(f"  ✓ 诊断报告已保存: {diagnosis_path}")

        # 如果检测到严重问题，给出建议
        if diagnosis_report['diagnosis']:
            logger.log(f"\n{'⚠️ '*20}")
            logger.log(f"检测到潜在问题，建议:")
            logger.log(f"  1. 检查校准数据集（C4/Wikitext2）是否适合当前模型")
            logger.log(f"  2. 尝试调整序列长度参数（--taylor_seq_len）")
            logger.log(f"  3. 尝试调整样本数参数（--taylor_num_samples）")
            logger.log(f"  4. 使用 temperature > 0 启用块级修正")
            logger.log(f"{'⚠️ '*20}\n")

    # ========== Step 9: LoRA 微调恢复（可选）==========
    if args.finetune:
        logger.log(f"\n[Step 9] LoRA 微调恢复...")
        logger.log(f"  数据集: {args.finetune_data_path}")
        logger.log(f"  训练轮数: {args.finetune_epochs}")
        logger.log(f"  学习率: {args.finetune_lr}")
        logger.log(f"  Batch size: {args.finetune_batch_size} (micro: {args.finetune_micro_batch_size})")
        logger.log(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")

        # 构建微调脚本命令
        import subprocess

        finetune_cmd = [
            "python", "finetune_lora.py",
            "--pruned_model", save_path,
            "--data_path", args.finetune_data_path,
            "--num_epochs", str(args.finetune_epochs),
            "--learning_rate", str(args.finetune_lr),
            "--batch_size", str(args.finetune_batch_size),
            "--micro_batch_size", str(args.finetune_micro_batch_size),
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--device", args.device
        ]

        # 如果指定跳过评估，添加参数
        if args.skip_finetune_evaluation:
            finetune_cmd.append("--skip_evaluation")

        # 执行微调
        logger.log(f"\n  启动 LoRA 微调...")
        try:
            result = subprocess.run(finetune_cmd, check=True, capture_output=False, text=True)
            logger.log(f"✓ LoRA 微调完成")

            # 微调后的模型保存路径
            finetuned_output_dir = os.path.join('results', f"{args.output_name}_finetuned")
            logger.log(f"  微调后的模型保存在: {finetuned_output_dir}")

            # 如果没有跳过微调后评估，评估结果也保存在该目录下
            if not args.skip_finetune_evaluation:
                finetuned_eval_path = os.path.join(finetuned_output_dir, 'evaluation', 'evaluation_results.json')
                logger.log(f"  微调后的评估结果: {finetuned_eval_path}")

        except subprocess.CalledProcessError as e:
            logger.log(f"⚠️ LoRA 微调失败: {e}")
            logger.log(f"  继续执行后续步骤...")
    else:
        logger.log(f"\n[Step 9] 跳过微调（未指定 --finetune）")

    # ========== Step 10: 保存模型 ==========
    # 准备模型字典（无论是否保存，评估都可能需要）
    save_dict = {
        'model': model,
        'tokenizer': tokenizer,
        'pruning_stats': pruning_stats,
        'pruning_ratio': args.pruning_ratio,
        'actual_ratio': actual_ratio,
        'method': 'H-GSP',
        'h-gsp_params': {
            'temperature': args.temperature,
            'tau': args.tau,
            'epsilon': args.epsilon
        },
        'config': args.__dict__
    }

    # 总是保存模型到 models/ 目录
    logger.log(f"\n[Step 10] 保存剪枝后的模型...")
    save_path = os.path.join(output_dirs['models'], 'pruned_model.bin')
    torch.save(save_dict, save_path)
    logger.log(f"✓ 模型已保存: {save_path}")
    logger.log(f"  文件大小: {os.path.getsize(save_path) / (1024**3):.2f} GB")

    # ========== Step 10.5: 生成剪枝可视化图表 ==========
    logger.log(f"\n[Step 10.5] 生成剪枝可视化图表...")
    try:
        # 配置中文字体
        font_used = setup_chinese_font()
        
        if font_used:
            logger.log(f"  使用字体: {font_used}")
            use_english = False
        else:
            logger.log(f"  ⚠️ 未找到中文字体，将使用英文标签")
            logger.log(f"  提示: 可安装中文字体: sudo apt-get install fonts-wqy-microhei")
            use_english = True

        # 生成图表
        generate_pruning_charts(
            pruning_data=comparison_result,
            model_name=args.output_name,
            output_dir=output_dirs['visualization'],
            use_english=use_english,
        )
        logger.log(f"  ✓ 剪枝图表已保存到: {output_dirs['visualization']}")
    except Exception as e:
        logger.log(f"  ⚠️ 图表生成失败: {e}")
        logger.log(f"  提示: 请确保安装了 matplotlib: pip install matplotlib")

    args.run_evaluation = False  # 临时禁用评估以节省时间
    # ========== Step 11: 运行评估测试（可选）==========
    if args.run_evaluation:
        logger.log(f"\n[Step 11] 运行评估测试...")

        # 清理显存：释放剪枝后的模型，为评估腾出空间
        logger.log(f"  清理显存...")
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.log(f"  ✓ 显存已清理")

        # 解析评估类型
        eval_types = [t.strip() for t in args.run_evaluation.split(',')]
        if 'all' in eval_types:
            eval_types = ['ppl', 'zeroshot', 'efficiency']

        logger.log(f"  评估类型: {', '.join(eval_types)}")

        # 解析数据集和任务
        ppl_datasets = [d.strip() for d in args.eval_ppl_datasets.split(',')] if 'ppl' in eval_types else None
        zeroshot_tasks = [t.strip() for t in args.eval_zeroshot_tasks.split(',')] if 'zeroshot' in eval_types else None

        # 运行评估
        logger.log(f"\n  开始评估...")
        eval_results = evaluate_single_model(
            model_path=save_path,
            metrics=eval_types,
            device=args.device,
            ppl_datasets=ppl_datasets,
            ppl_seq_len=args.eval_ppl_seq_len,
            ppl_stride=args.eval_ppl_stride,
            zeroshot_tasks=zeroshot_tasks,
            speed_samples=50,
            verbose=True,
            use_custom_zeroshot=args.eval_use_custom_zeroshot,
            zeroshot_batch_size=8
        )

        # 保存评估结果
        eval_result_path = os.path.join(output_dirs['evaluation'], 'evaluation_results.json')
        with open(eval_result_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        logger.log(f"\n✓ 评估结果已保存: {eval_result_path}")

        # 打印简要评估摘要
        logger.log(f"\n{'='*60}")
        logger.log(f"评估结果摘要")
        logger.log(f"{'='*60}")
        if 'ppl' in eval_results.get('metrics', {}):
            logger.log(f"\nPPL 结果:")
            for dataset, ppl in eval_results['metrics']['ppl'].items():
                logger.log(f"  {dataset}: {ppl:.2f}" if ppl else f"  {dataset}: N/A")

        if 'avg_zeroshot_acc' in eval_results.get('metrics', {}):
            acc = eval_results['metrics']['avg_zeroshot_acc']
            logger.log(f"\nZero-shot 平均准确率: {acc*100:.2f}%")

        if 'efficiency' in eval_results.get('metrics', {}):
            eff = eval_results['metrics']['efficiency']
            if 'speed' in eff:
                throughput = eff['speed'].get('batch_size_1', {}).get('throughput_tokens_per_sec', 'N/A')
                logger.log(f"\n推理速度: {throughput:.1f} tokens/s" if isinstance(throughput, (int, float)) else f"\n推理速度: {throughput}")
            if 'memory' in eff:
                mem = eff['memory'].get('model_memory_mb', 'N/A')
                logger.log(f"GPU 显存: {mem:.0f} MB" if isinstance(mem, (int, float)) else f"GPU 显存: {mem}")
    else:
        logger.log(f"\n[Step 11] 跳过评估测试（未指定 --run_evaluation）")

    logger.log(f"\n{'='*60}")
    logger.log(f"✓ 全部完成！")
    logger.log(f"{'='*60}")
    logger.log(f"\n输出目录: {output_dirs['base']}")
    logger.log(f"  - 模型: {output_dirs['models']}")
    logger.log(f"  - 分析结果: {output_dirs['analysis']}")
    logger.log(f"  - 评估结果: {output_dirs['evaluation']}")
    logger.log(f"  - 日志: {output_dirs['logs']}")


if __name__ == '__main__':
    main()
