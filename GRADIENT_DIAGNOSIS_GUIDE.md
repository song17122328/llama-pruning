# 梯度诊断和极端剪枝问题解决指南

## 问题背景

在使用全局泰勒剪枝（Global Taylor Pruning）时，可能会遇到**极端剪枝**问题，表现为：
- 某些层（特别是前几层）被过度剪枝，剪枝率达到 90% 甚至 99%
- 不同层之间的剪枝率差异巨大
- 剪枝后的模型性能严重下降

## 可能的原因

1. **梯度消失/爆炸**：某些层的梯度非常小或非常大
2. **梯度尺度不均**：不同层之间的梯度量级差异很大（可能相差 1000+ 倍）
3. **序列长度影响**：较长的序列（如 512）可能放大梯度差异
4. **模型架构差异**：不同模型（LLaMA vs Mistral）对校准数据集的敏感度不同
5. **校准数据集不匹配**：C4/Wikitext2 可能不适合某些特定模型

## 诊断工具

### 1. 自动梯度诊断

运行全局剪枝时，工具会自动：
- 收集每层的梯度统计（均值、标准差、范数等）
- 生成梯度分布可视化图表
- 诊断极端剪枝问题并给出建议

### 2. 诊断输出

**梯度统计文件**：
```
results/<model>/<experiment>/analysis/gradient_statistics.json
```

**可视化图表**：
```
results/<model>/<experiment>/visualization/gradient_analysis.png
```

**诊断报告**：
```
results/<model>/<experiment>/analysis/gradient_diagnosis.json
```

### 3. 诊断报告内容

诊断报告包含：
- **极端剪枝的层列表**：剪枝率超过阈值的层及其梯度/重要性信息
- **梯度统计**：
  - 梯度均值范围和比率（最大值/最小值）
  - 梯度范数范围和比率
- **具体问题诊断**：
  - 梯度尺度差异过大
  - 大量层被过度剪枝
  - 前几层被过度剪枝（特别严重）
- **改进建议**：针对每个问题的具体解决方案

### 4. 可视化图表说明

梯度分析图表包含 3 个子图：

1. **梯度统计**（对数刻度）：
   - 每层的梯度均值、标准差和范数
   - 帮助识别梯度消失/爆炸的层

2. **梯度范围**（对数刻度）：
   - 每层梯度的最大值和最小值
   - 显示梯度变化的范围

3. **梯度 vs 重要性 vs 剪枝率**：
   - 对比归一化后的梯度、重要性得分和剪枝率
   - 帮助理解剪枝决策是否合理

## 解决方案

### 方案 1：调整剪枝参数

**使用 temperature > 0 启用块级修正**：
```bash
python run_global_pruning.py \
    --model_name_or_path /path/to/Mistral-7B-v0.3 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --temperature 1.0 \
    --tau 0.0 \
    --dataset c4 \
    --output results/Mistral-7B-v0.3/taylor_T1_tau0_20_c4
```

**关键参数**：
- `--temperature 1.0`：启用块级重要性修正，平衡层间差异
- `--tau 0.0`：仅使用块级修正，不引入额外的层级修正

### 方案 2：限制剪枝率范围

虽然当前代码中没有直接暴露 `--min-rate` 和 `--max-rate` 参数，但可以修改代码来设置：

在 `core/methods/global_pruning.py` 中，修改 `build_global_group_table` 函数：

```python
# 在计算剪枝率时添加限制
MIN_RATE = 0.0   # 最小剪枝率
MAX_RATE = 0.5   # 最大剪枝率（避免剪掉超过50%）

# 特别保护前几层
EARLY_LAYER_MAX_RATE = 0.3  # 前5层最多剪30%
```

### 方案 3：调整序列长度

如果发现 `TAYLOR_SEQ_LEN=512` 导致极端剪枝，尝试调整为 256 或 128：

在 `run_global_pruning.py` 中修改：
```python
TAYLOR_SEQ_LEN = 256  # 从 512 改为 256（或 128）
```

**观察**：序列长度越长，梯度差异可能越大

### 方案 4：使用梯度归一化（编程方式）

可以在代码中使用归一化函数来平滑重要性得分：

```python
from core.analysis.gradient_analysis import (
    normalize_importance_scores,
    clip_importance_scores
)

# 在构建 global_analysis_table 之前

# 方法 1：对数变换（压缩极端值）
normalized_importance = normalize_importance_scores(
    importance_scores,
    method='log'  # 'minmax', 'zscore', 'log', 'sqrt'
)

# 方法 2：裁剪极端值（保留 5%-95% 范围）
clipped_importance = clip_importance_scores(
    importance_scores,
    percentile_low=5.0,
    percentile_high=95.0
)

# 使用归一化后的重要性得分
importance_scores = normalized_importance
```

**归一化方法说明**：
- `'minmax'`：线性归一化到 [0, 1]
- `'zscore'`：Z-score 标准化
- `'log'`：对数变换，压缩极端值（推荐）
- `'sqrt'`：平方根变换，温和压缩

### 方案 5：更换校准数据集

如果 C4/Wikitext2 不适合，尝试：
- 使用与模型训练分布更接近的数据集
- 使用领域特定的数据集（如果模型是针对特定领域的）

修改数据集：
```bash
python run_global_pruning.py \
    --dataset wikitext2 \  # 或 c4
    ...
```

### 方案 6：Layer-wise 渐进式剪枝

对不同深度的层使用不同的剪枝策略：

```python
# 在 build_global_group_table 中添加层深度权重
def get_layer_depth_weight(layer_idx, num_layers):
    """前面的层权重更高（更重要）"""
    # 线性衰减：前层权重高，后层权重低
    return 1.0 + (1.0 - layer_idx / num_layers) * 0.5

# 应用到重要性得分
importance *= get_layer_depth_weight(layer_idx, num_layers)
```

## 实验建议

### 对比实验

建议进行以下对比实验：

1. **Baseline（T=0）**：
   ```bash
   python run_global_pruning.py \
       --model_name_or_path /path/to/model \
       --pruning_ratio 0.2 \
       --importance_method taylor \
       --temperature 0.0 \
       --dataset c4
   ```

2. **块级修正（T=1, tau=0）**：
   ```bash
   python run_global_pruning.py \
       --model_name_or_path /path/to/model \
       --pruning_ratio 0.2 \
       --importance_method taylor \
       --temperature 1.0 \
       --tau 0.0 \
       --dataset c4
   ```

3. **不同序列长度**：
   - 修改 `TAYLOR_SEQ_LEN` 为 128, 256, 512
   - 对比梯度分布和剪枝结果

4. **不同数据集**：
   ```bash
   --dataset c4        # 对比
   --dataset wikitext2  # 对比
   ```

### 分析流程

1. **运行剪枝**
2. **查看梯度诊断报告**：
   ```bash
   cat results/<model>/<experiment>/analysis/gradient_diagnosis.json
   ```
3. **查看可视化图表**：
   ```bash
   results/<model>/<experiment>/visualization/gradient_analysis.png
   ```
4. **对比不同实验的梯度分布**
5. **根据诊断建议调整参数**

## 案例分析

### 案例：Mistral 极端剪枝问题

**问题描述**：
- Mistral-7B-v0.3 使用 T=0, TAYLOR_SEQ_LEN=512
- 第 2, 3, 4 层的 MLP 被剪掉 99%+
- 性能严重下降

**诊断结果**：
```json
{
  "diagnosis": [
    {
      "issue": "梯度尺度差异过大",
      "severity": "high",
      "description": "梯度均值在不同层间相差 2847.3 倍",
      "recommendation": "建议使用 layer-wise 梯度归一化或对数变换"
    },
    {
      "issue": "前几层被过度剪枝",
      "severity": "critical",
      "description": "前5层中有 3 层被过度剪枝",
      "recommendation": "前几层通常很重要，建议为其设置较低的 max_rate"
    }
  ]
}
```

**解决方案**：
1. 使用 T=1, tau=0 启用块级修正
2. 将 TAYLOR_SEQ_LEN 从 512 改回 256
3. 尝试 wikitext2 数据集

**结果**：
- 块级修正后，第 2 层剪枝率从 99% 降至 75%
- 第 3-4 层剪枝率也更加合理（48%-49%）
- 模型性能显著提升

## 最佳实践

1. **始终启用梯度诊断**：工具会自动运行，无需额外配置
2. **优先使用块级修正（T=1）**：对大多数模型更稳定
3. **从较短序列开始**：先用 256，如果效果好再尝试 512
4. **保护前几层**：可以为前 5 层设置更低的 max_rate
5. **对比多个数据集**：找到最适合当前模型的校准数据
6. **查看可视化**：图表比数字更直观，能快速发现问题
7. **渐进式调整**：一次只改一个参数，便于定位问题

## 常见问题

### Q1: 为什么 LLaMA 效果好，Mistral 效果差？

**A**: 可能的原因：
- 模型架构差异（层归一化、激活函数等）
- 训练数据分布不同
- 梯度传播特性不同

**建议**：为不同模型使用不同的配置，特别是序列长度和 temperature 参数。

### Q2: 梯度诊断的开销大吗？

**A**: 开销很小：
- 每个 batch 只记录统计信息，不额外计算
- 可视化在剪枝完成后一次性生成
- 总耗时增加 < 5%

### Q3: 如何判断剪枝是否合理？

**A**: 查看 3 个指标：
1. **梯度分布**：不应有极端的尖峰或低谷
2. **重要性-剪枝率相关性**：低重要性 → 高剪枝率应该成正比
3. **层间差异**：剪枝率不应相差过大（如 5% vs 95%）

### Q4: 可以完全禁用某些层的剪枝吗？

**A**: 可以，修改 `build_global_group_table` 函数：
```python
# 保护特定层（如前3层和最后1层）
protected_layers = [0, 1, 2, 31]
if layer_idx in protected_layers:
    continue  # 跳过该层的 groups
```

## 参考资料

- **ShortGPT**: Layer-wise importance analysis
- **Taylor Pruning**: First/second-order Taylor expansion
- **Wanda**: Activation-based pruning
- **论文**: "Structured Pruning Learns Compact and Accurate Models"

## 总结

梯度诊断工具帮助您：
1. **快速发现**极端剪枝问题
2. **定位根因**：梯度消失、尺度不均等
3. **自动建议**：基于诊断结果的改进方案
4. **可视化对比**：直观理解梯度、重要性和剪枝率的关系

通过合理使用诊断工具和调整参数，可以在大多数情况下缓解极端剪枝问题，获得更好的剪枝效果。
