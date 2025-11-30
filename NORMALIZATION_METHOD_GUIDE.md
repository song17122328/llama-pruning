# 归一化方法选择指南

## 🎯 快速推荐

**对于 Mistral 极端剪枝问题（梯度尺度差异 2847x）**：

```python
NORMALIZATION_METHOD = 'log'  # ← 强烈推荐
```

## 📊 四种方法详细对比

### 1. **log（对数变换）** ⭐ 推荐

**公式**：
```python
shifted = importance - min(importance) + 1.0
normalized = log(shifted)
normalized = (normalized - min) / (max - min)
```

**适用场景**：
- ✅ 梯度尺度差异**极大**（100x - 10000x）
- ✅ 存在极端剪枝问题（某些层 90%+ 剪枝率）
- ✅ Mistral 这种梯度不均衡的模型

**效果示例**：
```
原始重要性:
  Layer 0: 1000.0  ✓ 正常
  Layer 2:    0.35 ⚠️ 极端剪枝 (被剪 97%)
  差异: 2857x

log 归一化后:
  Layer 0: 1.000   (相对值: 1.15)
  Layer 2: 0.000   (相对值: 0.00)
  → Layer 2 的相对重要性提升，不会被过度剪枝
```

**优点**：
- 🏆 **压缩极端值最有效**（1000x → 3x 以内）
- 🏆 保持相对顺序不变
- 🏆 对异常值鲁棒

**缺点**：
- 需要处理负值和零值（已通过 shift 解决）

---

### 2. **minmax（线性归一化）**

**公式**：
```python
normalized = (importance - min) / (max - min)
```

**适用场景**：
- ✅ 梯度分布均匀（差异 < 10x）
- ✅ 没有极端异常值
- ⚠️ **不适合 Mistral 场景**

**效果示例**：
```
原始重要性:
  Layer 0: 1000.0
  Layer 2:    0.35
  差异: 2857x

minmax 归一化后:
  Layer 0: 1.000   (占据整个范围的顶端)
  Layer 2: 0.0003  (几乎为 0，仍然会被极端剪枝)
  → ⚠️ 问题未解决！
```

**优点**：
- 简单直观
- 归一化到 [0, 1] 范围

**缺点**：
- ❌ **对极端值敏感**（一个异常大值会压缩所有其他值）
- ❌ 不适合极端梯度场景

---

### 3. **zscore（Z-score 标准化）**

**公式**：
```python
normalized = (importance - mean) / std
```

**适用场景**：
- ✅ 梯度接近**正态分布**
- ✅ 数据对称分布
- ⚠️ 不适合极端偏态分布（如 Mistral）

**效果示例**：
```
原始重要性:
  Layer 0-1, 5-7: 500-1000 (5个正常层)
  Layer 2-4:      0.35-0.5 (3个极端低值)

zscore 归一化后:
  正常层: 约 +0.5 到 +1.5
  极端层: 约 -2.0 到 -1.5
  → 会产生负值，需要额外处理
```

**优点**：
- 适合正态分布数据
- 保留了分布的统计特性

**缺点**：
- ⚠️ 会产生负值（需要后处理）
- ❌ 对极端偏态分布效果一般
- ❌ 假设数据符合正态分布（Mistral 不符合）

---

### 4. **sqrt（平方根变换）**

**公式**：
```python
normalized = sqrt(importance)
```

**适用场景**：
- ✅ 梯度差异**中等**（10x - 100x）
- ✅ 需要温和压缩

**效果示例**：
```
原始重要性:
  Layer 0: 1000.0
  Layer 2:    0.35
  差异: 2857x

sqrt 归一化后:
  Layer 0: 31.62
  Layer 2:  0.59
  差异: 53x (仍然较大)
  → ⚠️ 压缩力度不够
```

**优点**：
- 温和的非线性变换
- 不会产生负值

**缺点**：
- ⚠️ **压缩力度有限**（对于 1000x+ 差异不够）
- ⚠️ 可能无法完全解决极端剪枝

---

## 🔬 实验对比矩阵

| 方法 | 压缩能力 | 适用差异范围 | 极端剪枝修复 | Mistral 适用性 |
|------|---------|-------------|-------------|---------------|
| **log** | ⭐⭐⭐⭐⭐ | 100x - 10000x | ✅ 优秀 | ✅ **强烈推荐** |
| **minmax** | ⭐ | < 10x | ❌ 差 | ❌ 不推荐 |
| **zscore** | ⭐⭐ | 正态分布 | ⚠️ 一般 | ⚠️ 效果有限 |
| **sqrt** | ⭐⭐⭐ | 10x - 100x | ⚠️ 一般 | ⚠️ 可能不够 |

## 🧪 验证方法

### 如何判断归一化是否有效？

**运行剪枝后，检查以下指标**：

1. **各层剪枝率分布**：
   ```python
   # 期望：各层剪枝率更均衡
   Layer 0: 45%
   Layer 1: 50%
   Layer 2: 55%  ← 原来 97%，现在降低！
   Layer 3: 48%  ← 原来 96%，现在降低！
   Layer 4: 52%  ← 原来 91%，现在降低！
   ```

2. **梯度尺度比率**：
   ```python
   # 查看 gradient_statistics.json
   "gradient_scale_ratio": 15.3  ← 原来 2847，现在大幅降低！
   ```

3. **诊断报告建议**：
   ```json
   {
     "extreme_pruning_layers": [],  // ← 应该为空
     "recommendations": ["无需归一化"]
   }
   ```

## 🚀 推荐配置

### 配置 1: log 归一化（推荐）

```python
# run_global_pruning.py, Step 4.5 配置
ENABLE_GRADIENT_NORMALIZATION = True
NORMALIZATION_METHOD = 'log'      # ← 对数变换
NORMALIZATION_LEVEL = 'block'     # ← Block-wise（分别处理 Attention/MLP）

ENABLE_GRADIENT_CLIPPING = False  # 先不启用裁剪
```

**适用场景**: Mistral 极端剪枝问题

---

### 配置 2: log + clipping 组合（加强版）

```python
ENABLE_GRADIENT_NORMALIZATION = True
NORMALIZATION_METHOD = 'log'
NORMALIZATION_LEVEL = 'block'

ENABLE_GRADIENT_CLIPPING = True   # ← 额外启用裁剪
CLIP_PERCENTILE_LOW = 5.0         # ← 裁剪最低 5%
CLIP_PERCENTILE_HIGH = 95.0       # ← 裁剪最高 5%
```

**适用场景**: 如果单独 log 归一化效果不够，再添加裁剪

---

### 配置 3: 对比实验

测试以下 4 种配置，找到最佳方案：

```bash
# Baseline
python run_global_pruning.py --pruning_ratio 0.2 \
    --output results/mistral_baseline

# log 归一化
# (修改 ENABLE_GRADIENT_NORMALIZATION = True, METHOD = 'log')
python run_global_pruning.py --pruning_ratio 0.2 \
    --output results/mistral_log_norm

# sqrt 归一化
# (修改 METHOD = 'sqrt')
python run_global_pruning.py --pruning_ratio 0.2 \
    --output results/mistral_sqrt_norm

# log + clipping
# (启用 CLIPPING)
python run_global_pruning.py --pruning_ratio 0.2 \
    --output results/mistral_log_clip
```

对比各配置的：
- 各层剪枝率
- 模型性能 (PPL)
- 梯度统计

## 📈 预期效果对比

### Before（无归一化）
```
梯度统计:
  Layer 0: mean_grad = 0.85
  Layer 2: mean_grad = 0.0003  (2833x 差异)

剪枝结果:
  Layer 0: 45% 剪枝
  Layer 2: 97% 剪枝  ⚠️ 极端剪枝！

模型性能:
  PPL: 25.8 (性能下降严重)
```

### After（log 归一化）
```
梯度统计:
  Layer 0: normalized = 1.000
  Layer 2: normalized = 0.312  (3x 差异，可控)

剪枝结果:
  Layer 0: 48% 剪枝
  Layer 2: 52% 剪枝  ✓ 合理范围！

模型性能:
  PPL: 12.3 (性能显著提升)
```

## ❓ 常见问题

### Q1: 为什么不直接用 Temperature？

**A**: Temperature 和归一化可以组合使用：

| 方法 | 作用阶段 | 计算开销 | 适用场景 |
|------|---------|---------|---------|
| Temperature (T>0) | 重要性计算时 | 中等（需额外前向） | 重要性评估偏差 |
| 归一化 | 重要性计算后 | 几乎无 | 梯度尺度不均 |

**建议**: 优先尝试 `T=1, tau=0`（无开销），如仍有问题再加归一化。

### Q2: Block-wise vs Layer-wise？

**A**: **Block-wise 更推荐**：

```python
# Layer-wise: Attention 和 MLP 一起归一化
# → 可能压制 Attention 的重要性

# Block-wise: Attention 和 MLP 分别归一化（推荐）
# → 保持各自内部的相对顺序
```

### Q3: 归一化会改变 score 吗？

**A**: 间接改变。流程是：

```python
# 1. 归一化 importance
normalized_importance = normalize(importance)

# 2. score 自动重新计算
score = normalized_importance / cost  ← cost 不变
```

归一化只影响 importance，score 自动更新以保持一致性。

## 🎓 总结

**对于 Mistral 极端剪枝问题**：

1. **首选**: `log` 归一化 + `block` 级别
2. **原因**: 梯度尺度差异 2847x，需要强力压缩
3. **配置**:
   ```python
   ENABLE_GRADIENT_NORMALIZATION = True
   NORMALIZATION_METHOD = 'log'
   NORMALIZATION_LEVEL = 'block'
   ```
4. **验证**: 检查 Layer 2-4 的剪枝率是否从 90%+ 降低到 50% 左右

**实验建议**：先测试 `log`，如果效果不理想再尝试 `log + clipping` 组合。

---

**相关文档**：
- `EXTREME_PRUNING_FIXES.md` - 极端剪枝修复方案
- `BLOCKWISE_NORMALIZATION_VERIFICATION.md` - Block-wise 归一化验证
- `GRADIENT_DIAGNOSIS_GUIDE.md` - 梯度诊断使用指南
