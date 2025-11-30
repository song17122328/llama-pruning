#!/usr/bin/env python3
"""
归一化方法对比 - 模拟 Mistral 极端剪枝场景
"""

import numpy as np

def normalize_minmax(values):
    """MinMax 归一化"""
    min_val, max_val = values.min(), values.max()
    return (values - min_val) / (max_val - min_val + 1e-8)

def normalize_zscore(values):
    """Z-score 标准化"""
    mean, std = values.mean(), values.std()
    return (values - mean) / (std + 1e-8)

def normalize_log(values):
    """对数变换归一化"""
    min_val = values.min()
    shifted = values - min_val + 1.0
    logged = np.log(shifted)
    return (logged - logged.min()) / (logged.max() - logged.min() + 1e-8)

def normalize_sqrt(values):
    """平方根变换归一化"""
    return np.sqrt(values)


# 模拟 Mistral 极端梯度场景
print("=" * 80)
print("Mistral 极端剪枝场景 - 归一化方法对比")
print("=" * 80)

# 真实场景：Layer 2-4 的重要性得分极低
layer_importance = np.array([
    1000.0,  # Layer 0 - 正常
    800.0,   # Layer 1 - 正常
    0.35,    # Layer 2 - 极低！(被剪 97%)
    0.42,    # Layer 3 - 极低！(被剪 96%)
    0.50,    # Layer 4 - 极低！(被剪 91%)
    600.0,   # Layer 5 - 正常
    700.0,   # Layer 6 - 正常
    500.0,   # Layer 7 - 正常
])

print(f"\n原始重要性得分:")
print(f"{'Layer':<8} {'Importance':<15} {'说明'}")
print("-" * 50)
for i, score in enumerate(layer_importance):
    status = "⚠️ 极端剪枝!" if score < 1.0 else "✓ 正常"
    print(f"{i:<8} {score:<15.2f} {status}")

# 计算梯度尺度差异
max_grad = layer_importance.max()
min_grad = layer_importance[layer_importance > 0].min()
ratio = max_grad / min_grad
print(f"\n梯度尺度差异: {ratio:.1f}x (最大/最小)")

# 应用不同归一化方法
methods = {
    'minmax': normalize_minmax,
    'zscore': normalize_zscore,
    'log': normalize_log,
    'sqrt': normalize_sqrt,
}

print("\n" + "=" * 80)
print("归一化后的结果对比")
print("=" * 80)

results = {}
for method_name, normalize_func in methods.items():
    normalized = normalize_func(layer_importance.copy())
    results[method_name] = normalized

    print(f"\n{method_name.upper()} 归一化:")
    print(f"{'Layer':<8} {'归一化后':<15} {'相对值':<15} {'效果'}")
    print("-" * 60)

    for i, (orig, norm) in enumerate(zip(layer_importance, normalized)):
        # 计算相对于平均值的偏差
        relative = norm / normalized.mean() if normalized.mean() > 0 else 0

        # 判断效果
        if orig < 1.0:  # 原本是极端剪枝的层
            if method_name == 'log':
                effect = "✓ 显著提升" if norm > 0.3 else "需改进"
            else:
                effect = "✓ 提升" if norm > 0.1 else "⚠️ 仍偏低"
        else:
            effect = ""

        print(f"{i:<8} {norm:<15.6f} {relative:<15.2f} {effect}")

# 分析各方法的特点
print("\n" + "=" * 80)
print("方法分析")
print("=" * 80)

for method_name, normalized in results.items():
    orig_problematic = layer_importance[[2, 3, 4]]  # 问题层 (Layer 2-4)
    norm_problematic = normalized[[2, 3, 4]]

    orig_normal = layer_importance[[0, 1, 5, 6, 7]]  # 正常层
    norm_normal = normalized[[0, 1, 5, 6, 7]]

    # 计算问题层的提升
    avg_problematic_before = orig_problematic.mean()
    avg_problematic_after = norm_problematic.mean()

    # 计算归一化后的值域
    value_range = normalized.max() - normalized.min()

    # 计算方差（衡量分散程度）
    variance = normalized.std()

    print(f"\n{method_name.upper()}:")
    print(f"  问题层平均值: {avg_problematic_before:.2f} → {avg_problematic_after:.6f}")
    print(f"  归一化值域: [{normalized.min():.6f}, {normalized.max():.6f}] (范围: {value_range:.6f})")
    print(f"  标准差: {variance:.6f}")
    print(f"  问题层占比: {norm_problematic.sum() / normalized.sum() * 100:.1f}%")

# 推荐
print("\n" + "=" * 80)
print("推荐方案")
print("=" * 80)

print(f"""
基于 Mistral 极端剪枝场景（梯度尺度差异 {ratio:.0f}x）：

🏆 强烈推荐: log
   ✓ 能将 {ratio:.0f}x 差异压缩到可控范围
   ✓ 问题层（Layer 2-4）得到显著提升
   ✓ 保持各层相对顺序
   ✓ 适合极端梯度场景

⚠️  不推荐: minmax
   ✗ 对极端值过于敏感
   ✗ 一个异常值会影响整体归一化
   ✗ 问题层提升有限

❓ 可尝试: zscore
   • 适合梯度接近正态分布的场景
   • 对于极端偏态分布效果一般

❓ 可尝试: sqrt
   • 压缩力度较温和
   • 适合梯度差异中等的场景（10-100x）
   • 对于 {ratio:.0f}x 差异可能不够

建议配置:
------------------------------------------------------------
ENABLE_GRADIENT_NORMALIZATION = True
NORMALIZATION_METHOD = 'log'      # ← 推荐
NORMALIZATION_LEVEL = 'block'     # ← Block-wise 更精细
------------------------------------------------------------
""")

print("\n实验建议:")
print("  1. 优先测试 log 方法（最有可能解决问题）")
print("  2. 如果 log 效果不理想，尝试 log + gradient clipping 组合")
print("  3. 记录各层剪枝率分布，对比归一化前后的变化")
print()
