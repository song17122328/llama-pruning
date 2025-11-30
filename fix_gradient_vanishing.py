#!/usr/bin/env python3
"""
梯度消失修复方案
"""

import torch

# ========== 方案1: 层级归一化 ==========

def normalize_gradients_per_layer(model):
    """
    对每层的梯度进行归一化

    原理：
    - 问题：不同层的梯度scale差异大（Layer 0: 1e-9, Layer 31: 1e-5）
    - 解决：每层内部归一化，只保留相对重要性

    注意：这会改变绝对scale，但保留每层内部的相对排序
    """
    # 按层分组
    layer_gradients = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            # 提取层号
            if 'layers.' in name:
                layer_idx = int(name.split('layers.')[1].split('.')[0])
                if layer_idx not in layer_gradients:
                    layer_gradients[layer_idx] = []
                layer_gradients[layer_idx].append((name, param))

    # 每层独立归一化
    for layer_idx, params in layer_gradients.items():
        # 计算该层所有参数梯度的范数
        grad_norms = []
        for name, param in params:
            grad_norms.append(param.grad.norm().item())

        # 使用该层的平均范数归一化
        layer_avg_norm = sum(grad_norms) / len(grad_norms)

        if layer_avg_norm > 0:
            for name, param in params:
                param.grad = param.grad / layer_avg_norm

    print("✓ 层级梯度归一化完成")


# ========== 方案2: 梯度裁剪 + 缩放 ==========

def clip_and_scale_gradients(model, clip_value=1.0):
    """
    梯度裁剪 + 全局缩放

    原理：
    - 裁剪：防止极端大的梯度
    - 缩放：将所有梯度映射到合理范围
    """
    # 全局梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

    # 计算全局梯度统计
    all_grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            all_grad_norms.append(param.grad.norm().item())

    global_avg = sum(all_grad_norms) / len(all_grad_norms)

    # 缩放到统一范围
    for param in model.parameters():
        if param.grad is not None:
            param.grad = param.grad / global_avg

    print(f"✓ 梯度裁剪完成 (clip_value={clip_value})")


# ========== 方案3: 相对重要性排序 ==========

def compute_relative_importance(model, hessian_diag=None):
    """
    计算每层内部的相对重要性

    原理：
    - 不使用绝对的Taylor分数
    - 在每层内部独立排序
    - 每层剪枝相同比例
    """
    layer_importance = {}

    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            # 提取层号
            if 'layers.' in name:
                layer_idx = int(name.split('layers.')[1].split('.')[0])

                # 计算Taylor重要性
                importance = (param.data.abs() * param.grad.abs()).cpu()

                # 二阶泰勒
                if hessian_diag is not None and name in hessian_diag:
                    hess = hessian_diag[name]
                    second_order = 0.5 * (param.data ** 2 * hess).abs().cpu()
                    importance = importance + second_order

                if layer_idx not in layer_importance:
                    layer_importance[layer_idx] = {}

                layer_importance[layer_idx][name] = importance

    # 每层内部归一化到[0, 1]
    for layer_idx in layer_importance:
        for name in layer_importance[layer_idx]:
            imp = layer_importance[layer_idx][name]
            imp_min = imp.min()
            imp_max = imp.max()
            if imp_max > imp_min:
                layer_importance[layer_idx][name] = (imp - imp_min) / (imp_max - imp_min)

    return layer_importance


# ========== 方案4: 梯度加权（基于层深度）==========

def weight_gradients_by_depth(model, strategy='linear'):
    """
    根据层深度对梯度加权

    策略：
    - linear: 前面层权重大（补偿梯度消失）
    - sqrt: 平方根加权
    - log: 对数加权
    """
    num_layers = max([
        int(name.split('layers.')[1].split('.')[0])
        for name, _ in model.named_parameters()
        if 'layers.' in name
    ]) + 1

    for name, param in model.named_parameters():
        if param.grad is not None and 'layers.' in name:
            layer_idx = int(name.split('layers.')[1].split('.')[0])

            # 计算权重（前面层权重更大）
            if strategy == 'linear':
                weight = (num_layers - layer_idx) / num_layers
            elif strategy == 'sqrt':
                weight = ((num_layers - layer_idx) / num_layers) ** 0.5
            elif strategy == 'log':
                import math
                weight = math.log(num_layers - layer_idx + 1) / math.log(num_layers + 1)
            else:
                weight = 1.0

            # 放大前面层的梯度
            param.grad = param.grad * (weight + 0.5)  # +0.5避免权重过小

    print(f"✓ 梯度深度加权完成 (strategy={strategy})")


# ========== 方案5: 自适应缩放 ==========

def adaptive_gradient_scaling(model):
    """
    自适应梯度缩放

    原理：
    - 检测每层的梯度范数
    - 将所有层缩放到相似范围
    - 保留相对差异，但缩小绝对差异
    """
    # 收集每层的梯度范数
    layer_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None and 'layers.' in name:
            layer_idx = int(name.split('layers.')[1].split('.')[0])

            if layer_idx not in layer_stats:
                layer_stats[layer_idx] = {'norms': [], 'params': []}

            grad_norm = param.grad.norm().item()
            layer_stats[layer_idx]['norms'].append(grad_norm)
            layer_stats[layer_idx]['params'].append((name, param))

    # 计算目标范数（所有层的中位数）
    all_norms = []
    for layer_idx in layer_stats:
        all_norms.extend(layer_stats[layer_idx]['norms'])

    target_norm = sorted(all_norms)[len(all_norms) // 2]  # 中位数

    # 缩放每层到目标范数
    for layer_idx in layer_stats:
        layer_avg_norm = sum(layer_stats[layer_idx]['norms']) / len(layer_stats[layer_idx]['norms'])

        if layer_avg_norm > 0:
            scale = target_norm / layer_avg_norm
            # 限制缩放比例（避免过度放大/缩小）
            scale = max(0.1, min(scale, 10.0))

            for name, param in layer_stats[layer_idx]['params']:
                param.grad = param.grad * scale

    print(f"✓ 自适应梯度缩放完成 (target_norm={target_norm:.6e})")


# ========== 使用示例 ==========

if __name__ == "__main__":
    print("="*80)
    print("梯度消失修复方案")
    print("="*80)
    print()

    print("【使用方法】")
    print()
    print("在 run_global_pruning.py 的梯度计算循环中：")
    print()
    print("```python")
    print("# 在 loss.backward() 之后，累加 Hessian 之前")
    print("loss.backward()")
    print()
    print("# 方案1: 层级归一化（推荐）")
    print("from fix_gradient_vanishing import normalize_gradients_per_layer")
    print("normalize_gradients_per_layer(model)")
    print()
    print("# 或方案2: 自适应缩放")
    print("from fix_gradient_vanishing import adaptive_gradient_scaling")
    print("adaptive_gradient_scaling(model)")
    print()
    print("# 或方案3: 深度加权")
    print("from fix_gradient_vanishing import weight_gradients_by_depth")
    print("weight_gradients_by_depth(model, strategy='sqrt')")
    print()
    print("# 然后继续累加 Hessian")
    print("if args.importance_method == 'taylor_2nd':")
    print("    for name, param in model.named_parameters():")
    print("        ...")
    print("```")
    print()

    print("【方案对比】")
    print()
    print("┌─────────────────┬──────────┬──────────┬─────────────┐")
    print("│ 方案            │ 复杂度   │ 效果     │ 适用场景    │")
    print("├─────────────────┼──────────┼──────────┼─────────────┤")
    print("│ 层级归一化      │ 低       │ 好       │ 通用        │")
    print("│ 自适应缩放      │ 中       │ 很好     │ 梯度差异大  │")
    print("│ 深度加权        │ 低       │ 中       │ 深层网络    │")
    print("│ 梯度裁剪        │ 低       │ 中       │ 梯度爆炸    │")
    print("│ 相对重要性      │ 高       │ 很好     │ 需要改算法  │")
    print("└─────────────────┴──────────┴──────────┴─────────────┘")
    print()

    print("【推荐策略】")
    print()
    print("1. 先尝试: 自适应缩放（效果最好，代码简单）")
    print("2. 如果不够: 自适应缩放 + 深度加权（组合使用）")
    print("3. 最保守: 只用层级归一化（最安全）")
    print()

    print("="*80)
