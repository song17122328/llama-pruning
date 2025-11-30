# 梯度统计与性能相关性分析 🔬

## 概述

本工具集帮助您发现梯度统计指标与剪枝后性能（ACC）的关系，**适合科研论文使用**。

### 核心功能

1. ✅ **参数网格搜索** - 自动测试多种参数组合
2. ✅ **梯度统计收集** - 提取梯度范数比率、均值比率等关键指标
3. ✅ **相关性分析** - 发现梯度指标与 ACC 的关系
4. ✅ **性能预测模型** - 无需完整评估即可预测性能
5. ✅ **科研价值** - 提供理论支撑和论文写作素材

## 快速开始（3 步）

### 步骤 1: 运行参数搜索

```bash
# 1. 修改配置文件中的模型路径
vim configs/mistral_param_search.json

# 2. 运行搜索（测试 15 种参数组合）
python param_search/search_best_params.py --config configs/mistral_param_search.json
```

**输出**: `results/param_search_mistral_20/search_results.csv`

包含的信息：
- 参数配置（`taylor_seq_len`, `taylor_num_samples`）
- 7 个 zero-shot 任务的单独 ACC
- 平均 ACC (`acc_mean`)
- 梯度统计（`grad_norm_ratio`, `grad_mean_ratio` 等）
- PPL, 剪枝率等

### 步骤 2: 分析相关性

```bash
python param_search/analyze_param_correlations.py \
    --results results/param_search_mistral_20/search_results.csv
```

**输出**:
- `correlation_heatmap.png` - 相关性热力图
- `scatter_matrix.png` - 散点图矩阵
- `correlation_report.txt` - 详细报告
- `prediction_model.json` - 预测公式

### 步骤 3: 查看关键发现

```bash
# 查看相关性报告
cat results/param_search_mistral_20/correlation_report.txt

# 查看预测模型
cat results/param_search_mistral_20/prediction_model.json
```

## 📊 收集的指标

### ACC 指标（7 个任务）

| 指标 | 任务 |
|------|------|
| `acc_boolq` | BoolQ |
| `acc_piqa` | PIQA |
| `acc_hellaswag` | HellaSwag |
| `acc_winogrande` | WinoGrande |
| `acc_arc_easy` | ARC-Easy |
| `acc_arc_challenge` | ARC-Challenge |
| `acc_openbookqa` | OpenBookQA |
| **`acc_mean`** | **7 个任务平均** ⭐ |

### 梯度统计指标

| 指标 | 说明 | 预期与 ACC 的关系 |
|------|------|------------------|
| **`grad_norm_ratio`** ⭐ | 梯度范数比率（最大/最小）| **强负相关**：比率越小，ACC 越高 |
| `grad_mean_ratio` | 梯度均值比率 | 负相关 |
| `grad_std_ratio` | 梯度标准差比率 | 负相关 |
| `grad_max_ratio` | 梯度最大值比率 | 负相关 |
| `extreme_pruning_layers` | 极端剪枝层数量（>80%）| **强负相关**：越少越好 |

## 🔬 预期发现

### 发现 1: 梯度范数比率是关键

**假设**: `grad_norm_ratio` 与 `acc_mean` 强负相关

**如果验证成功**:
- 论文价值：✅ 可以用梯度统计快速预测性能
- 实践价值：✅ 无需完整评估即可选择最佳配置
- 理论价值：✅ 解释为什么某些配置更好

**论文表述示例**:
> "我们发现梯度范数比率与剪枝后准确率存在显著负相关（Spearman r = -0.85, p < 0.001）。
> 当梯度范数比率低于 10x 时，剪枝后准确率平均提高 3.2%。"

### 发现 2: 短序列提高性能

**假设**: `taylor_seq_len` 与 `acc_mean` 负相关（越短越好）

**如果验证成功**:
- 支持您的观察：短序列确实提高 ACC
- 理论解释：减少梯度累积导致的数值不稳定

**论文表述示例**:
> "实验表明，使用 32-64 tokens 的短序列计算 Taylor 重要性可显著提高剪枝后准确率。
> 我们推测这是因为短序列减少了梯度累积带来的数值不稳定性。"

### 发现 3: 可建立预测模型

**预期**: 可以用梯度指标预测 ACC（R² > 0.7）

**论文表述示例**:
> "基于梯度统计指标，我们建立了性能预测模型：
>   ACC ≈ 0.72 - 0.012 × grad_norm_ratio - 0.0008 × seq_len
> 该模型可以解释 82% 的性能变化（R² = 0.82），平均误差为 1.45%。"

## 📈 可视化示例

### 相关性热力图

显示所有指标之间的 Spearman 相关性：

```
                    taylor_seq_len  grad_norm_ratio  acc_mean  ppl
taylor_seq_len            1.00          0.34        -0.78    0.45
grad_norm_ratio           0.34          1.00        -0.85    0.67
acc_mean                 -0.78         -0.85         1.00   -0.72
ppl                       0.45          0.67        -0.72    1.00
```

**关键观察**:
- `grad_norm_ratio` vs `acc_mean`: **-0.85（强负相关！）**
- `taylor_seq_len` vs `acc_mean`: -0.78（负相关）

### 散点图矩阵

每个子图显示一个指标与 ACC 的关系，包括：
- 散点图
- 趋势线
- 相关系数和显著性标记

## 🎓 如何写入论文

### 1. 方法部分

```markdown
为了找到最佳的 Taylor 重要性计算参数，我们进行了系统的网格搜索，
测试了不同的序列长度（16, 32, 64, 128, 256）和样本数（128, 256, 512）。
同时，我们收集了梯度统计指标，包括梯度范数比率、梯度均值比率等，
并分析了这些指标与剪枝后准确率的相关性。
```

### 2. 结果部分

```markdown
图 X 展示了梯度范数比率与平均准确率的关系。我们发现两者存在显著的
负相关（Spearman r = -0.85, p < 0.001），这表明梯度范数比率可以作为
剪枝质量的快速预测指标。

此外，我们发现序列长度对性能有显著影响（r = -0.78, p < 0.01）。
使用 32-64 tokens 的短序列可以减少梯度累积带来的数值不稳定性。

基于这些发现，我们建立了一个预测模型，可以解释 82% 的性能变化（R² = 0.82）。
```

### 3. 讨论部分

```markdown
我们的分析揭示了梯度统计指标与剪枝后性能之间的内在联系。
梯度范数比率反映了不同层之间梯度的不均衡程度。当比率较高时，
某些层的梯度会被显著高估或低估，导致重要性评估偏差，进而造成极端剪枝。

这一发现具有重要的实践意义：在实际应用中，可以通过监控梯度统计指标
来快速评估剪枝配置的质量，无需进行完整的模型评估。
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `search_best_params.py` | 参数搜索脚本（已增强，收集梯度统计）|
| `analyze_param_correlations.py` | 相关性分析脚本 |
| `CORRELATION_ANALYSIS_GUIDE.md` | 详细使用指南 |
| `configs/mistral_param_search.json` | 搜索配置（包含 7 个 zero-shot 任务）|

## 🔧 高级用法

### 分析特定任务

```bash
# 分析 HellaSwag 任务
python param_search/analyze_param_correlations.py \
    --results results/param_search_mistral_20/search_results.csv \
    --target acc_hellaswag
```

### 自定义输出目录

```bash
python param_search/analyze_param_correlations.py \
    --results results/param_search_mistral_20/search_results.csv \
    --output_dir my_analysis/
```

### 查看详细数据

```python
import pandas as pd

df = pd.read_csv('results/param_search_mistral_20/search_results.csv')
df = df[df['success'] == True]

# 显示所有列
print(df.columns.tolist())

# 查看梯度统计的分布
print(df[['grad_norm_ratio', 'grad_mean_ratio', 'acc_mean']].describe())

# 找出最佳配置
best = df.loc[df['acc_mean'].idxmax()]
print(f"最佳配置: taylor_seq_len={best['taylor_seq_len']}, "
      f"grad_norm_ratio={best['grad_norm_ratio']:.2f}")
```

## 🎯 关键要点

1. ✅ **收集梯度统计**: 每次剪枝实验都会记录梯度范数比率等关键指标
2. ✅ **详细的 ACC**: 记录所有 7 个 zero-shot 任务的单独 ACC
3. ✅ **相关性分析**: 自动发现梯度指标与 ACC 的关系
4. ✅ **预测模型**: 建立基于梯度统计的性能预测模型
5. ✅ **科研价值**: 提供理论支撑和论文写作素材

## 📚 相关文档

- **快速开始**: `QUICK_START_PARAM_SEARCH.md`
- **详细指南**: `PARAM_SEARCH_GUIDE.md`
- **相关性分析**: `CORRELATION_ANALYSIS_GUIDE.md`
- **归一化方法**: `NORMALIZATION_METHOD_GUIDE.md`

## 💡 示例工作流

```bash
# 1. 运行参数搜索（6-8 小时）
python param_search/search_best_params.py --config configs/mistral_param_search.json

# 2. 分析相关性（< 1 分钟）
python param_search/analyze_param_correlations.py \
    --results results/param_search_mistral_20/search_results.csv

# 3. 查看发现
cat results/param_search_mistral_20/correlation_report.txt

# 4. 查看可视化
open results/param_search_mistral_20/correlation_heatmap.png
open results/param_search_mistral_20/scatter_matrix.png

# 5. 使用最佳配置运行完整实验
python run_global_pruning.py \
    --base_model /path/to/Mistral-7B-v0.3 \
    --output_name mistral_final_best \
    --pruning_ratio 0.2 \
    --taylor_seq_len 64 \
    --taylor_num_samples 256 \
    --run_evaluation all
```

## 🚀 开始探索

运行您的第一个相关性分析：

```bash
# 快速测试（2-3 小时，只测试 3 个配置）
python param_search/search_best_params.py --config configs/quick_param_search.json

# 分析
python param_search/analyze_param_correlations.py \
    --results results/quick_search/search_results.csv
```

**祝您发现有价值的规律，写出高质量的论文！** 📝✨
