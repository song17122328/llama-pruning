# 极端剪枝问题修复方案

## 问题诊断

如果您遇到以下情况：
- 某些层被过度剪枝（如 Layer 2-4 被剪掉 90%+）
- 梯度诊断报告显示梯度尺度差异过大（> 1000x）
- 前几层被剪得比后几层严重

## 快速修复方案

### 方案 1：启用梯度归一化（推荐）⭐

**修改位置**：`run_global_pruning.py` 第 1205 行

```python
# 找到这一行并修改为 True
ENABLE_GRADIENT_NORMALIZATION = True  # 改为 True 启用

# 可选：选择归一化方法
NORMALIZATION_METHOD = 'log'  # 推荐使用 'log'，也可以试 'minmax', 'zscore', 'sqrt'
```

**效果**：
- 对每层的重要性得分分别归一化
- 压缩极端值，使不同层的得分更均衡
- **Layer-wise 归一化**：确保每层内部的相对排序不变，只调整层间关系

**适用场景**：
- ✅ 梯度尺度差异 > 100x
- ✅ 前几层被过度剪枝
- ✅ 不同层之间剪枝率差异巨大

---

### 方案 2：启用梯度裁剪

**修改位置**：`run_global_pruning.py` 第 1207 行

```python
ENABLE_GRADIENT_CLIPPING = True  # 改为 True 启用

# 可选：调整裁剪范围
CLIP_PERCENTILE_LOW = 5.0   # 裁剪最低 5% 的极端值
CLIP_PERCENTILE_HIGH = 95.0 # 裁剪最高 5% 的极端值
```

**效果**：
- 裁剪全局的极端重要性得分
- 保留 5%-95% 范围内的值

**适用场景**：
- ✅ 有少量层的梯度异常大或异常小
- ✅ 梯度分布有明显的离群值

---

### 方案 3：同时启用归一化 + 裁剪（最强）🔥

```python
ENABLE_GRADIENT_NORMALIZATION = True
NORMALIZATION_METHOD = 'log'
ENABLE_GRADIENT_CLIPPING = True
CLIP_PERCENTILE_LOW = 5.0
CLIP_PERCENTILE_HIGH = 95.0
```

**效果**：
- 先裁剪极端值
- 再按层归一化
- 双重保险，效果最稳定

---

### 方案 4：使用 Temperature > 0（块级修正）

**不修改代码，直接在命令行中**：

```bash
python run_global_pruning.py \
    --base_model /newdata/LLMs/Mistral-7B-v0.3 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --temperature 1.0 \  # 改为 1.0
    --tau 0.0 \
    --dataset c4 \
    --output results/Mistral-7B-v0.3/taylor_T1_tau0_c4
```

**效果**：
- 启用块级重要性修正
- 平衡 Attention 和 MLP 的剪枝
- 减少层间差异

**优点**：
- ✅ 无需修改代码
- ✅ 您已经验证过 T=1 效果更好

---

## 完整的修复流程

### Step 1: 先用梯度诊断分析问题

运行一次基础版本，查看诊断报告：

```bash
python run_global_pruning.py \
    --base_model /newdata/LLMs/Mistral-7B-v0.3 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --temperature 0.0 \
    --dataset c4 \
    --output results/Mistral-7B-v0.3/taylor_baseline_diagnostic
```

查看诊断结果：
```bash
cat results/results/Mistral-7B-v0.3/taylor_baseline_diagnostic/analysis/gradient_diagnosis.json
```

### Step 2: 根据诊断结果选择修复方案

**如果诊断显示**：
- `mean_ratio > 1000`: 使用方案 1（梯度归一化）
- `num_extreme_layers > 5`: 使用方案 3（归一化 + 裁剪）
- 前几层被过度剪枝: 使用方案 1 + 方案 4

### Step 3: 应用修复并对比

**启用归一化**：
1. 编辑 `run_global_pruning.py`
2. 修改第 1205 行：`ENABLE_GRADIENT_NORMALIZATION = True`
3. 重新运行

**对比结果**：
```bash
# 查看剪枝对比
diff results/*/taylor_*/analysis/pruning_comparison.json

# 对比梯度诊断
diff results/*/taylor_*/analysis/gradient_diagnosis.json
```

---

## 不同归一化方法的选择

### `'log'` - 对数变换（推荐）⭐

```python
NORMALIZATION_METHOD = 'log'
```

**特点**：
- 压缩大值，提升小值
- 保持相对顺序
- 对极端值不敏感

**适用**：
- ✅ 梯度尺度差异非常大（> 1000x）
- ✅ Mistral 等容易出现极端剪枝的模型

### `'minmax'` - 最小-最大归一化

```python
NORMALIZATION_METHOD = 'minmax'
```

**特点**：
- 线性缩放到 [0, 1]
- 简单直观

**适用**：
- ✅ 梯度尺度差异适中（100x ~ 1000x）
- ✅ 需要严格控制重要性范围

### `'zscore'` - Z-score 标准化

```python
NORMALIZATION_METHOD = 'zscore'
```

**特点**：
- 中心化到均值，按标准差缩放
- 适合正态分布的数据

**适用**：
- ✅ 梯度分布接近正态分布
- ✅ 需要考虑分布特性

### `'sqrt'` - 平方根变换

```python
NORMALIZATION_METHOD = 'sqrt'
```

**特点**：
- 温和压缩大值
- 变换强度介于线性和对数之间

**适用**：
- ✅ 梯度尺度差异适中
- ✅ 不希望过度压缩极端值

---

## 实验建议

### 对比实验矩阵

| 实验 | T | 归一化 | 裁剪 | 数据集 | 备注 |
|------|---|--------|------|--------|------|
| baseline | 0 | ❌ | ❌ | c4 | 基准 |
| norm_log | 0 | ✅ log | ❌ | c4 | 对数归一化 |
| norm_minmax | 0 | ✅ minmax | ❌ | c4 | 线性归一化 |
| clip | 0 | ❌ | ✅ | c4 | 仅裁剪 |
| norm+clip | 0 | ✅ log | ✅ | c4 | 组合方案 |
| temp1 | 1 | ❌ | ❌ | c4 | 块级修正 |
| temp1+norm | 1 | ✅ log | ❌ | c4 | 块级+归一化 |
| wikitext2 | 0 | ❌ | ❌ | wikitext2 | 换数据集 |

### 快速对比脚本

创建 `run_experiments.sh`：

```bash
#!/bin/bash

MODEL="/newdata/LLMs/Mistral-7B-v0.3"
RATIO=0.2
METHOD="taylor"

# 实验 1: baseline
python run_global_pruning.py \
    --base_model $MODEL \
    --pruning_ratio $RATIO \
    --importance_method $METHOD \
    --temperature 0.0 \
    --dataset c4 \
    --output results/Mistral-7B-v0.3/exp1_baseline

# 实验 2: log normalization (需要先修改代码启用)
# ENABLE_GRADIENT_NORMALIZATION = True
# NORMALIZATION_METHOD = 'log'
python run_global_pruning.py \
    --base_model $MODEL \
    --pruning_ratio $RATIO \
    --importance_method $METHOD \
    --temperature 0.0 \
    --dataset c4 \
    --output results/Mistral-7B-v0.3/exp2_norm_log

# 实验 3: temperature 1.0
python run_global_pruning.py \
    --base_model $MODEL \
    --pruning_ratio $RATIO \
    --importance_method $METHOD \
    --temperature 1.0 \
    --tau 0.0 \
    --dataset c4 \
    --output results/Mistral-7B-v0.3/exp3_temp1

# 对比结果
echo "=== 剪枝率对比 ==="
for exp in exp1_baseline exp2_norm_log exp3_temp1; do
    echo "## $exp"
    cat results/results/Mistral-7B-v0.3/$exp/analysis/pruning_summary_by_layer.txt | grep "Layer 2\|Layer 3\|Layer 4"
    echo ""
done
```

---

## 效果评估

### 判断修复是否成功

查看以下指标：

1. **剪枝率分布**：
   ```bash
   cat results/*/analysis/pruning_summary_by_layer.txt
   ```
   - ✅ 第 2-4 层剪枝率 < 50%
   - ✅ 各层剪枝率差异 < 30%

2. **梯度诊断**：
   ```bash
   cat results/*/analysis/gradient_diagnosis.json | grep mean_ratio
   ```
   - ✅ `mean_ratio` < 100
   - ✅ `num_extreme_layers` < 3

3. **模型性能**：
   ```bash
   cat results/*/evaluation/evaluation_results.json
   ```
   - ✅ PPL 下降幅度合理
   - ✅ 任务性能保持稳定

---

## 常见问题

### Q1: 启用归一化后效果反而变差？

**A**: 可能的原因：
- 归一化方法不合适：尝试换成 `'log'`
- 需要同时启用裁剪
- 数据集不匹配：尝试换成 wikitext2

### Q2: 归一化会影响剪枝效果吗？

**A**:
- Layer-wise 归一化：**不影响**每层内部的相对排序
- 只调整层间的重要性平衡
- 剪枝效果通常会**更好**，因为避免了极端剪枝

### Q3: 如何选择最佳的归一化方法？

**A**:
1. 先看梯度诊断的 `mean_ratio`
2. 如果 > 1000: 用 `'log'`
3. 如果 100-1000: 用 `'minmax'` 或 `'sqrt'`
4. 如果 < 100: 可能不需要归一化，直接用 T=1

### Q4: 可以通过命令行参数控制吗？

**A**:
目前需要修改代码。如果需要命令行参数，可以添加：

```python
# 在 argparse 中添加
parser.add_argument('--enable_grad_norm', action='store_true',
                    help='启用梯度归一化')
parser.add_argument('--norm_method', type=str, default='log',
                    choices=['minmax', 'zscore', 'log', 'sqrt'],
                    help='归一化方法')

# 使用
ENABLE_GRADIENT_NORMALIZATION = args.enable_grad_norm
NORMALIZATION_METHOD = args.norm_method
```

---

## 总结

**推荐修复流程**：

1. ✅ **先诊断**：运行基础版本，查看 `gradient_diagnosis.json`
2. ✅ **选方案**：
   - 梯度差异 > 1000x → 启用 log 归一化
   - 极端剪枝层 > 5 → 归一化 + 裁剪
   - 想简单快速 → 直接用 T=1
3. ✅ **验证效果**：对比剪枝率分布和模型性能
4. ✅ **迭代优化**：根据结果调整参数

**最稳妥的方案**：
```python
ENABLE_GRADIENT_NORMALIZATION = True
NORMALIZATION_METHOD = 'log'
ENABLE_GRADIENT_CLIPPING = True
```

配合：
```bash
--temperature 1.0 --tau 0.0
```

这样可以从多个角度缓解极端剪枝问题！
