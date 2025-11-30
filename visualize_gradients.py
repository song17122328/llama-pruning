#!/usr/bin/env python3
"""
梯度可视化工具
"""

import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class GradientVisualizer:
    """梯度可视化器"""

    def __init__(self, output_dir='gradient_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.gradient_history = []

    def collect_gradients(self, model, step_name=''):
        """
        收集当前梯度

        Args:
            model: PyTorch模型
            step_name: 步骤名称（如'batch_0', 'batch_1'）
        """
        grad_info = {
            'step_name': step_name,
            'layer_grads': {}
        }

        for name, param in model.named_parameters():
            if param.grad is not None and 'layers.' in name and 'weight' in name:
                layer_idx = int(name.split('layers.')[1].split('.')[0])

                grad_mean = param.grad.abs().mean().item()
                grad_std = param.grad.abs().std().item()
                grad_max = param.grad.abs().max().item()
                grad_min = param.grad.abs().min().item()

                if layer_idx not in grad_info['layer_grads']:
                    grad_info['layer_grads'][layer_idx] = {
                        'mean': [],
                        'std': [],
                        'max': [],
                        'min': [],
                        'names': []
                    }

                grad_info['layer_grads'][layer_idx]['mean'].append(grad_mean)
                grad_info['layer_grads'][layer_idx]['std'].append(grad_std)
                grad_info['layer_grads'][layer_idx]['max'].append(grad_max)
                grad_info['layer_grads'][layer_idx]['min'].append(grad_min)
                grad_info['layer_grads'][layer_idx]['names'].append(name)

        self.gradient_history.append(grad_info)
        return grad_info

    def plot_gradient_distribution(self, step_idx=0, output_name='gradient_dist.png'):
        """
        绘制梯度分布图（按层）

        显示：每层的平均梯度范数
        """
        if step_idx >= len(self.gradient_history):
            print(f"⚠️ step_idx {step_idx} 超出范围")
            return

        grad_info = self.gradient_history[step_idx]
        layer_grads = grad_info['layer_grads']

        # 准备数据
        layers = sorted(layer_grads.keys())
        mean_grads = []
        std_grads = []

        for layer_idx in layers:
            # 该层所有参数的平均梯度
            layer_mean = np.mean(layer_grads[layer_idx]['mean'])
            layer_std = np.std(layer_grads[layer_idx]['mean'])
            mean_grads.append(layer_mean)
            std_grads.append(layer_std)

        # 绘图
        plt.figure(figsize=(14, 6))

        # 子图1: 对数尺度
        plt.subplot(1, 2, 1)
        plt.semilogy(layers, mean_grads, 'b-o', linewidth=2, markersize=6)
        plt.fill_between(layers,
                         np.array(mean_grads) - np.array(std_grads),
                         np.array(mean_grads) + np.array(std_grads),
                         alpha=0.3)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Gradient Magnitude (log scale)', fontsize=12)
        plt.title(f'Gradient Distribution Across Layers\n({grad_info["step_name"]})',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 标注梯度消失区域
        min_grad = min(mean_grads)
        max_grad = max(mean_grads)
        if max_grad / min_grad > 100:  # 超过100倍差异
            plt.axhline(y=min_grad * 10, color='r', linestyle='--', alpha=0.5,
                       label=f'Vanishing threshold ({min_grad*10:.2e})')
            plt.legend()

        # 子图2: 线性尺度
        plt.subplot(1, 2, 2)
        plt.plot(layers, mean_grads, 'g-o', linewidth=2, markersize=6)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Gradient Magnitude', fontsize=12)
        plt.title('Gradient Distribution (Linear Scale)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # 标注异常层
        mean_grad_global = np.mean(mean_grads)
        for i, (layer_idx, grad) in enumerate(zip(layers, mean_grads)):
            if grad < mean_grad_global * 0.1:  # 小于平均值的10%
                plt.scatter([layer_idx], [grad], color='red', s=100, zorder=5)
                plt.annotate(f'L{layer_idx}',
                           xy=(layer_idx, grad),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8,
                           color='red')

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 梯度分布图已保存: {output_path}")

        # 打印统计信息
        print(f"\n梯度统计 ({grad_info['step_name']}):")
        print(f"  最大梯度: {max_grad:.6e} (Layer {layers[mean_grads.index(max_grad)]})")
        print(f"  最小梯度: {min_grad:.6e} (Layer {layers[mean_grads.index(min_grad)]})")
        print(f"  梯度比值: {max_grad/min_grad:.2f}x")

        return output_path

    def plot_gradient_heatmap(self, step_idx=0, output_name='gradient_heatmap.png'):
        """
        绘制梯度热力图

        显示：每层每个参数的梯度大小
        """
        if step_idx >= len(self.gradient_history):
            print(f"⚠️ step_idx {step_idx} 超出范围")
            return

        grad_info = self.gradient_history[step_idx]
        layer_grads = grad_info['layer_grads']

        layers = sorted(layer_grads.keys())

        # 收集所有参数的梯度
        data = []
        param_names = []
        layer_labels = []

        for layer_idx in layers:
            layer_means = layer_grads[layer_idx]['mean']
            layer_names = layer_grads[layer_idx]['names']

            for mean, name in zip(layer_means, layer_names):
                data.append(mean)
                # 提取参数类型（如 self_attn.q_proj）
                param_type = name.split(f'layers.{layer_idx}.')[1].split('.weight')[0]
                param_names.append(param_type)
                layer_labels.append(layer_idx)

        # 转换为矩阵（layer x param_type）
        unique_params = sorted(set(param_names))
        matrix = np.zeros((len(layers), len(unique_params)))

        for i, layer_idx in enumerate(layers):
            for j, param_type in enumerate(unique_params):
                # 找到对应的梯度
                for mean, pname, layer in zip(data, param_names, layer_labels):
                    if layer == layer_idx and pname == param_type:
                        matrix[i, j] = mean
                        break

        # 绘制热力图
        plt.figure(figsize=(16, 10))
        im = plt.imshow(np.log10(matrix + 1e-10), aspect='auto', cmap='viridis')

        plt.xlabel('Parameter Type', fontsize=12)
        plt.ylabel('Layer Index', fontsize=12)
        plt.title(f'Gradient Heatmap (log10 scale)\n({grad_info["step_name"]})',
                  fontsize=14, fontweight='bold')

        # 设置坐标轴
        plt.xticks(range(len(unique_params)), unique_params, rotation=45, ha='right')
        plt.yticks(range(len(layers)), layers)

        # 颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('log10(Gradient Magnitude)', fontsize=12)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 梯度热力图已保存: {output_path}")

        return output_path

    def plot_gradient_comparison(self, output_name='gradient_comparison.png'):
        """
        对比多个步骤的梯度变化

        显示：梯度随训练/批次的变化
        """
        if len(self.gradient_history) < 2:
            print("⚠️ 需要至少2个步骤的数据")
            return

        plt.figure(figsize=(14, 8))

        # 为每个步骤绘制一条线
        for step_idx, grad_info in enumerate(self.gradient_history):
            layer_grads = grad_info['layer_grads']
            layers = sorted(layer_grads.keys())

            mean_grads = []
            for layer_idx in layers:
                layer_mean = np.mean(layer_grads[layer_idx]['mean'])
                mean_grads.append(layer_mean)

            plt.semilogy(layers, mean_grads, '-o',
                        label=grad_info['step_name'],
                        linewidth=2, markersize=4, alpha=0.7)

        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Gradient Magnitude (log scale)', fontsize=12)
        plt.title('Gradient Evolution Across Steps', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 梯度对比图已保存: {output_path}")

        return output_path

    def plot_layer_variance(self, output_name='layer_variance.png'):
        """
        绘制层内梯度方差

        显示：每层内部参数的梯度差异
        """
        if len(self.gradient_history) == 0:
            print("⚠️ 没有梯度数据")
            return

        grad_info = self.gradient_history[0]
        layer_grads = grad_info['layer_grads']

        layers = sorted(layer_grads.keys())
        within_layer_variance = []
        between_param_variance = []

        for layer_idx in layers:
            # 层内方差（同一层不同参数的差异）
            layer_means = layer_grads[layer_idx]['mean']
            within_var = np.std(layer_means)
            within_layer_variance.append(within_var)

            # 参数内方差（同一参数不同元素的差异）
            layer_stds = layer_grads[layer_idx]['std']
            between_var = np.mean(layer_stds)
            between_param_variance.append(between_var)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.semilogy(layers, within_layer_variance, 'b-o', linewidth=2)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Std of Parameter Gradients', fontsize=12)
        plt.title('Within-Layer Gradient Variance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.semilogy(layers, between_param_variance, 'r-o', linewidth=2)
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Avg Gradient Std', fontsize=12)
        plt.title('Within-Parameter Gradient Variance', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ 层内方差图已保存: {output_path}")

        return output_path

    def generate_report(self, output_name='gradient_report.txt'):
        """生成文本报告"""
        if len(self.gradient_history) == 0:
            print("⚠️ 没有梯度数据")
            return

        output_path = self.output_dir / output_name

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("梯度分析报告\n")
            f.write("="*80 + "\n\n")

            for step_idx, grad_info in enumerate(self.gradient_history):
                f.write(f"【步骤 {step_idx}: {grad_info['step_name']}】\n\n")

                layer_grads = grad_info['layer_grads']
                layers = sorted(layer_grads.keys())

                # 计算统计信息
                all_means = []
                for layer_idx in layers:
                    layer_mean = np.mean(layer_grads[layer_idx]['mean'])
                    all_means.append(layer_mean)

                max_grad = max(all_means)
                min_grad = min(all_means)
                avg_grad = np.mean(all_means)

                f.write(f"全局统计：\n")
                f.write(f"  最大梯度: {max_grad:.6e}\n")
                f.write(f"  最小梯度: {min_grad:.6e}\n")
                f.write(f"  平均梯度: {avg_grad:.6e}\n")
                f.write(f"  梯度比值: {max_grad/min_grad:.2f}x\n\n")

                # 检测梯度消失
                if max_grad / min_grad > 100:
                    f.write("⚠️ 检测到梯度消失！\n")
                    f.write(f"   前几层与后几层梯度相差 {max_grad/min_grad:.0f} 倍\n\n")

                # 列出梯度最小的层
                sorted_layers = sorted(zip(layers, all_means), key=lambda x: x[1])
                f.write("梯度最小的5层：\n")
                for layer_idx, grad in sorted_layers[:5]:
                    f.write(f"  Layer {layer_idx:2d}: {grad:.6e}\n")
                f.write("\n")

                # 列出梯度最大的层
                f.write("梯度最大的5层：\n")
                for layer_idx, grad in sorted_layers[-5:]:
                    f.write(f"  Layer {layer_idx:2d}: {grad:.6e}\n")
                f.write("\n")

                f.write("-"*80 + "\n\n")

        print(f"✓ 梯度报告已保存: {output_path}")

        return output_path


# ========== 使用示例 ==========

if __name__ == "__main__":
    print("="*80)
    print("梯度可视化工具")
    print("="*80)
    print()

    print("【使用方法】")
    print()
    print("在 run_global_pruning.py 中：")
    print()
    print("```python")
    print("from visualize_gradients import GradientVisualizer")
    print()
    print("# 创建可视化器")
    print("visualizer = GradientVisualizer(output_dir='gradient_analysis')")
    print()
    print("# 在梯度计算循环中")
    print("for batch_idx in range(num_batches):")
    print("    # ... 前向传播 ...")
    print("    loss.backward()")
    print()
    print("    # 收集梯度（只在前几个batch）")
    print("    if batch_idx < 5:")
    print("        visualizer.collect_gradients(model, step_name=f'batch_{batch_idx}')")
    print()
    print("# 梯度计算结束后，生成可视化")
    print("visualizer.plot_gradient_distribution(step_idx=0, output_name='grad_dist_batch0.png')")
    print("visualizer.plot_gradient_heatmap(step_idx=0, output_name='grad_heatmap_batch0.png')")
    print("visualizer.plot_gradient_comparison(output_name='grad_comparison.png')")
    print("visualizer.plot_layer_variance(output_name='layer_variance.png')")
    print("visualizer.generate_report(output_name='gradient_report.txt')")
    print("```")
    print()

    print("【生成的文件】")
    print()
    print("gradient_analysis/")
    print("├── grad_dist_batch0.png      # 梯度分布图（对数+线性）")
    print("├── grad_heatmap_batch0.png   # 梯度热力图")
    print("├── grad_comparison.png       # 多批次梯度对比")
    print("├── layer_variance.png        # 层内方差分析")
    print("└── gradient_report.txt       # 文本报告")
    print()

    print("="*80)
