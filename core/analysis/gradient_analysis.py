#!/usr/bin/env python3
"""
梯度分析和可视化工具
用于诊断和可视化模型各层的梯度分布，帮助理解极端剪枝问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json
import os


class GradientAnalyzer:
    """梯度分析器 - 收集、分析和可视化模型梯度"""

    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        self.gradient_stats = defaultdict(lambda: {
            'mean': [],
            'std': [],
            'norm': [],
            'max': [],
            'min': []
        })

    def log(self, message):
        """日志输出"""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def collect_gradient_stats(self, layer_prefix='model.layers'):
        """
        收集当前模型的梯度统计信息

        Args:
            layer_prefix: 层名称前缀，用于过滤特定层

        Returns:
            Dict[str, Dict]: 各层的梯度统计
        """
        stats = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            # 只收集指定前缀的层
            if layer_prefix and not name.startswith(layer_prefix):
                continue

            grad = param.grad.detach()

            stats[name] = {
                'mean': grad.abs().mean().item(),
                'std': grad.abs().std().item(),
                'norm': grad.norm(p=2).item(),
                'max': grad.abs().max().item(),
                'min': grad.abs().min().item(),
                'shape': list(grad.shape)
            }

        return stats

    def accumulate_gradient_stats(self, layer_prefix='model.layers'):
        """
        累积梯度统计（用于多批次分析）

        Args:
            layer_prefix: 层名称前缀
        """
        current_stats = self.collect_gradient_stats(layer_prefix)

        for name, stat in current_stats.items():
            self.gradient_stats[name]['mean'].append(stat['mean'])
            self.gradient_stats[name]['std'].append(stat['std'])
            self.gradient_stats[name]['norm'].append(stat['norm'])
            self.gradient_stats[name]['max'].append(stat['max'])
            self.gradient_stats[name]['min'].append(stat['min'])

    def get_layer_gradient_summary(self, num_layers: int) -> Dict[int, Dict]:
        """
        获取每层的梯度汇总统计

        Args:
            num_layers: 层数

        Returns:
            Dict[int, Dict]: {layer_idx: {metric: value}}
        """
        layer_summary = {}

        for layer_idx in range(num_layers):
            # 收集该层所有参数的梯度
            layer_grads = {
                'mean': [],
                'std': [],
                'norm': [],
                'max': [],
                'min': []
            }

            for name, stats in self.gradient_stats.items():
                if f'model.layers.{layer_idx}.' in name:
                    # 对每个指标取平均
                    layer_grads['mean'].extend(stats['mean'])
                    layer_grads['std'].extend(stats['std'])
                    layer_grads['norm'].extend(stats['norm'])
                    layer_grads['max'].extend(stats['max'])
                    layer_grads['min'].extend(stats['min'])

            # 计算该层的统计
            if layer_grads['mean']:
                layer_summary[layer_idx] = {
                    'mean': np.mean(layer_grads['mean']),
                    'std': np.mean(layer_grads['std']),
                    'norm': np.mean(layer_grads['norm']),
                    'max': np.max(layer_grads['max']),
                    'min': np.min(layer_grads['min'])
                }
            else:
                layer_summary[layer_idx] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'norm': 0.0,
                    'max': 0.0,
                    'min': 0.0
                }

        return layer_summary

    def visualize_gradient_distribution(
        self,
        num_layers: int,
        save_dir: str,
        importance_scores: Optional[Dict] = None,
        pruning_rates: Optional[Dict] = None
    ):
        """
        可视化梯度分布

        Args:
            num_layers: 层数
            save_dir: 保存目录
            importance_scores: 重要性得分（可选）
            pruning_rates: 剪枝率（可选）
        """
        os.makedirs(save_dir, exist_ok=True)

        layer_summary = self.get_layer_gradient_summary(num_layers)
        layers = sorted(layer_summary.keys())

        # 提取各项指标
        means = [layer_summary[i]['mean'] for i in layers]
        stds = [layer_summary[i]['std'] for i in layers]
        norms = [layer_summary[i]['norm'] for i in layers]
        maxs = [layer_summary[i]['max'] for i in layers]
        mins = [layer_summary[i]['min'] for i in layers]

        # 创建多子图
        num_plots = 3 if importance_scores and pruning_rates else 2
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots))
        if num_plots == 1:
            axes = [axes]

        # 1. 梯度统计（均值、标准差、范数）
        ax1 = axes[0]
        ax1.plot(layers, means, 'o-', label='Mean', linewidth=2, markersize=6)
        ax1.plot(layers, stds, 's-', label='Std', linewidth=2, markersize=6)
        ax1.plot(layers, norms, '^-', label='Norm', linewidth=2, markersize=6)
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Gradient Magnitude', fontsize=12)
        ax1.set_title('Gradient Statistics by Layer', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 使用对数刻度

        # 2. 梯度范围（最大值和最小值）
        ax2 = axes[1]
        ax2.fill_between(layers, mins, maxs, alpha=0.3, label='Min-Max Range')
        ax2.plot(layers, maxs, 'ro-', label='Max', linewidth=2, markersize=5)
        ax2.plot(layers, mins, 'bo-', label='Min', linewidth=2, markersize=5)
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Gradient Magnitude', fontsize=12)
        ax2.set_title('Gradient Range by Layer', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')

        # 3. 梯度 vs 重要性 vs 剪枝率（如果提供）
        if importance_scores and pruning_rates:
            ax3 = axes[2]

            # 归一化以便在同一图上显示
            norm_means = np.array(means)
            norm_means = (norm_means - norm_means.min()) / (norm_means.max() - norm_means.min() + 1e-8)

            importance_values = [importance_scores.get(i, 0) for i in layers]
            norm_importance = np.array(importance_values)
            if norm_importance.max() > 0:
                norm_importance = (norm_importance - norm_importance.min()) / (norm_importance.max() - norm_importance.min() + 1e-8)

            pruning_values = [pruning_rates.get(i, 0) for i in layers]

            ax3_twin1 = ax3.twinx()
            ax3_twin2 = ax3.twinx()
            ax3_twin2.spines['right'].set_position(('outward', 60))

            p1 = ax3.plot(layers, norm_means, 'g^-', label='Norm. Gradient Mean',
                         linewidth=2, markersize=6)
            p2 = ax3_twin1.plot(layers, norm_importance, 'bs-', label='Norm. Importance',
                               linewidth=2, markersize=6)
            p3 = ax3_twin2.plot(layers, pruning_values, 'ro-', label='Pruning Rate',
                               linewidth=2, markersize=6)

            ax3.set_xlabel('Layer Index', fontsize=12)
            ax3.set_ylabel('Normalized Gradient Mean', fontsize=12, color='g')
            ax3_twin1.set_ylabel('Normalized Importance', fontsize=12, color='b')
            ax3_twin2.set_ylabel('Pruning Rate', fontsize=12, color='r')
            ax3.set_title('Gradient vs Importance vs Pruning Rate', fontsize=14, fontweight='bold')

            ax3.tick_params(axis='y', labelcolor='g')
            ax3_twin1.tick_params(axis='y', labelcolor='b')
            ax3_twin2.tick_params(axis='y', labelcolor='r')

            # 组合图例
            lines = p1 + p2 + p3
            labels = [l.get_label() for l in lines]
            ax3.legend(lines, labels, loc='upper left', fontsize=11)
            ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'gradient_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"✓ 梯度分布可视化已保存: {save_path}")

    def diagnose_extreme_pruning(
        self,
        num_layers: int,
        importance_scores: Dict[int, float],
        pruning_rates: Dict[int, float],
        threshold: float = 0.5
    ) -> Dict:
        """
        诊断极端剪枝问题

        Args:
            num_layers: 层数
            importance_scores: 重要性得分
            pruning_rates: 剪枝率
            threshold: 极端剪枝阈值（默认 0.5）

        Returns:
            Dict: 诊断报告
        """
        layer_summary = self.get_layer_gradient_summary(num_layers)

        # 找出极端剪枝的层
        extreme_layers = []
        for layer_idx, rate in pruning_rates.items():
            if rate > threshold:
                extreme_layers.append({
                    'layer_idx': layer_idx,
                    'pruning_rate': rate,
                    'importance': importance_scores.get(layer_idx, 0),
                    'gradient_mean': layer_summary.get(layer_idx, {}).get('mean', 0),
                    'gradient_norm': layer_summary.get(layer_idx, {}).get('norm', 0)
                })

        # 梯度统计
        all_grad_means = [layer_summary[i]['mean'] for i in range(num_layers)]
        all_grad_norms = [layer_summary[i]['norm'] for i in range(num_layers)]

        report = {
            'extreme_pruning_layers': extreme_layers,
            'num_extreme_layers': len(extreme_layers),
            'gradient_statistics': {
                'mean_range': (min(all_grad_means), max(all_grad_means)),
                'mean_ratio': max(all_grad_means) / (min(all_grad_means) + 1e-10),
                'norm_range': (min(all_grad_norms), max(all_grad_norms)),
                'norm_ratio': max(all_grad_norms) / (min(all_grad_norms) + 1e-10)
            },
            'diagnosis': []
        }

        # 诊断分析
        if report['gradient_statistics']['mean_ratio'] > 1000:
            report['diagnosis'].append({
                'issue': '梯度尺度差异过大',
                'severity': 'high',
                'description': f"梯度均值在不同层间相差 {report['gradient_statistics']['mean_ratio']:.1f} 倍",
                'recommendation': '建议使用 layer-wise 梯度归一化或对数变换'
            })

        if len(extreme_layers) > num_layers * 0.2:
            report['diagnosis'].append({
                'issue': '大量层被过度剪枝',
                'severity': 'high',
                'description': f"{len(extreme_layers)} 层的剪枝率超过 {threshold*100}%",
                'recommendation': '建议限制剪枝率范围（min_rate, max_rate）或使用温度平滑'
            })

        # 检查是否前几层被过度剪枝
        early_extreme = [l for l in extreme_layers if l['layer_idx'] < 5]
        if early_extreme:
            report['diagnosis'].append({
                'issue': '前几层被过度剪枝',
                'severity': 'critical',
                'description': f"前5层中有 {len(early_extreme)} 层被过度剪枝",
                'recommendation': '前几层通常很重要，建议为其设置较低的 max_rate'
            })

        return report

    def print_diagnosis_report(self, report: Dict):
        """打印诊断报告"""
        self.log("\n" + "="*80)
        self.log("梯度诊断报告")
        self.log("="*80)

        self.log(f"\n极端剪枝层数: {report['num_extreme_layers']}")
        if report['extreme_pruning_layers']:
            self.log("\n极端剪枝的层:")
            for layer_info in report['extreme_pruning_layers'][:10]:  # 只显示前10个
                self.log(f"  Layer {layer_info['layer_idx']:2d}: "
                        f"剪枝率={layer_info['pruning_rate']:.2%}, "
                        f"重要性={layer_info['importance']:.4e}, "
                        f"梯度均值={layer_info['gradient_mean']:.4e}")

        self.log(f"\n梯度统计:")
        stats = report['gradient_statistics']
        self.log(f"  梯度均值范围: {stats['mean_range'][0]:.4e} ~ {stats['mean_range'][1]:.4e}")
        self.log(f"  梯度均值比率: {stats['mean_ratio']:.2f}x")
        self.log(f"  梯度范数范围: {stats['norm_range'][0]:.4e} ~ {stats['norm_range'][1]:.4e}")
        self.log(f"  梯度范数比率: {stats['norm_ratio']:.2f}x")

        if report['diagnosis']:
            self.log(f"\n诊断结果:")
            for diag in report['diagnosis']:
                severity_icon = "🔴" if diag['severity'] == 'critical' else "⚠️" if diag['severity'] == 'high' else "ℹ️"
                self.log(f"\n  {severity_icon} {diag['issue']}")
                self.log(f"     描述: {diag['description']}")
                self.log(f"     建议: {diag['recommendation']}")

        self.log("="*80 + "\n")

    def save_gradient_stats(self, save_path: str):
        """保存梯度统计到 JSON 文件"""
        # 转换为可序列化格式
        serializable_stats = {}
        for name, stats in self.gradient_stats.items():
            serializable_stats[name] = {
                key: [float(v) for v in values]
                for key, values in stats.items()
            }

        with open(save_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)

        self.log(f"✓ 梯度统计已保存: {save_path}")


def normalize_importance_scores(
    importance_scores: Dict[int, float],
    method: str = 'minmax',
    epsilon: float = 1e-8
) -> Dict[int, float]:
    """
    归一化重要性得分以缓解极端剪枝

    Args:
        importance_scores: 原始重要性得分
        method: 归一化方法
            - 'minmax': 最小-最大归一化
            - 'zscore': Z-score 标准化
            - 'log': 对数变换
            - 'sqrt': 平方根变换
        epsilon: 防止除零的小常数

    Returns:
        Dict[int, float]: 归一化后的重要性得分
    """
    if not importance_scores:
        return {}

    values = np.array(list(importance_scores.values()))
    keys = list(importance_scores.keys())

    if method == 'minmax':
        # 最小-最大归一化到 [0, 1]
        min_val = values.min()
        max_val = values.max()
        normalized = (values - min_val) / (max_val - min_val + epsilon)

    elif method == 'zscore':
        # Z-score 标准化
        mean_val = values.mean()
        std_val = values.std()
        normalized = (values - mean_val) / (std_val + epsilon)
        # 映射到 [0, 1]
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + epsilon)

    elif method == 'log':
        # 对数变换
        # 先确保所有值为正
        min_val = values.min()
        shifted = values - min_val + 1.0
        normalized = np.log(shifted)
        # 归一化到 [0, 1]
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + epsilon)

    elif method == 'sqrt':
        # 平方根变换
        min_val = values.min()
        shifted = values - min_val
        normalized = np.sqrt(shifted)
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + epsilon)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return {k: float(v) for k, v in zip(keys, normalized)}


def clip_importance_scores(
    importance_scores: Dict[int, float],
    percentile_low: float = 5.0,
    percentile_high: float = 95.0
) -> Dict[int, float]:
    """
    裁剪重要性得分的极端值

    Args:
        importance_scores: 原始重要性得分
        percentile_low: 下限百分位（默认5%）
        percentile_high: 上限百分位（默认95%）

    Returns:
        Dict[int, float]: 裁剪后的重要性得分
    """
    if not importance_scores:
        return {}

    values = np.array(list(importance_scores.values()))
    keys = list(importance_scores.keys())

    low_bound = np.percentile(values, percentile_low)
    high_bound = np.percentile(values, percentile_high)

    clipped = np.clip(values, low_bound, high_bound)

    return {k: float(v) for k, v in zip(keys, clipped)}
