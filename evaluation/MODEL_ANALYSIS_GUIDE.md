# 模型结构分析完整指南

本指南介绍如何使用批量模型分析工具来快速了解所有剪枝模型的结构特征。

## 📋 工具概览

| 工具 | 功能 | 适用场景 |
|------|------|---------|
| `model_analysis.py` | 分析单个模型的详细结构 | 详细分析特定模型 |
| `batch_model_analysis.py` | 批量分析多个模型结构 | 有模型文件时批量处理 |
| **`summarize_model_structures.py`** | **汇总已有的分析结果** | **快速汇总现有分析** ⭐ |
| `generate_results_table.py` | 汇总评估性能结果 | 生成性能对比表格 |

## 🚀 快速开始

### 场景 1: 汇总已有的分析结果（推荐）

如果你的模型目录下已经有 `analysis/` 文件夹，直接使用汇总工具：

```bash
python evaluation/summarize_model_structures.py \
    --dirs baselines results \
    --output models_structure_summary
```

**输出：**
- `models_structure_summary.txt`: 汇总表格（文本格式）
- `models_structure_summary.json`: 详细数据（JSON 格式）

**示例输出：**
```
模型名称                                    总参数       剪枝比例     层数   完全剪空的层
baselines/Magnitude_2000         6,424,219,648     20.00%     32            0
baselines/Wanda_2000             6,424,219,648     20.00%     32            0
results/ShortGPT_remove_7        6,503,477,248     19.01%     25            0
```

### 场景 2: 从头分析模型结构（有模型文件时）

如果你有实际的模型文件，可以批量分析：

```bash
# 第一步：分析标准模型
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --output_dir baselines_analysis/

# 第二步：运行 SliceGPT 脚本（如果有）
conda activate slicegpt
bash baselines_analysis/analyze_slicegpt.sh

# 第三步：汇总所有结果
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --merge_results \
    --output_dir baselines_analysis/
```

### 场景 3: 分析单个模型

如果只想分析某个特定模型：

```bash
python core/analysis/model_analysis.py \
    --model_path /path/to/pruned_model.bin \
    --compare_with /newdata/LLMs/Llama-3-8B-Instruct \
    --output_dir results/MyModel/analysis/
```

## 📊 输出文件说明

### 1. 汇总文件

#### `models_structure_summary.json`

包含所有模型的详细信息：

```json
{
  "timestamp": "2025-11-27T16:33:17",
  "total_models": 13,
  "models": [
    {
      "name": "baselines/Magnitude_2000",
      "params": {
        "total": 6424219648,
        "num_layers": 32
      },
      "pruning": {
        "reduction_ratio": 0.2
      },
      "layer_pruning": [
        {
          "layer_idx": 0,
          "reduction_ratio": 0.808,
          "is_zero": false
        }
        // ... 更多层
      ]
    }
    // ... 更多模型
  ]
}
```

#### `models_structure_summary.txt`

人类可读的表格格式，包含：
- 模型名称、总参数、剪枝比例、层数
- 统计信息（最小/最大/平均）

### 2. 单个模型的分析文件

每个模型在其 `analysis/` 目录下有：

```
baselines/Magnitude_2000/analysis/
├── model_structure.json          # 模型结构详情
├── model_comparison.json         # 与原模型对比
├── original_model_analysis.json  # 原始模型分析
├── pruned_model_analysis.json    # 剪枝模型分析
└── pruning_summary_by_layer.txt  # 按层的剪枝摘要
```

## 🔍 关键指标解读

### 1. 剪枝比例 (Reduction Ratio)

```
reduction_ratio = (原始参数 - 剪枝后参数) / 原始参数
```

- **0.2 (20%)**: 轻度剪枝，保留 80% 参数
- **0.5 (50%)**: 中度剪枝，保留 50% 参数
- **0.8 (80%)**: 重度剪枝，仅保留 20% 参数

### 2. 层级剪枝分布

每层的剪枝比例反映了剪枝策略：

- **均匀剪枝**: 所有层剪枝比例相近（Magnitude, Wanda）
- **非均匀剪枝**: 不同层剪枝比例差异大（HGSP, 分层剪枝）
- **完全剪空层**: 某些层被完全移除（ShortGPT, 层移除方法）

### 3. 特殊层识别

- **Zero Layer**: 完全剪空的层（`is_zero_layer: true`）
- **Heavy Pruned**: 剪枝比例 > 80% 的层
- **Light Pruned**: 剪枝比例 < 20% 的层

## 📈 高级用法

### 按剪枝方法分组分析

```bash
# 只分析 Magnitude 和 Wanda
python evaluation/summarize_model_structures.py \
    --dirs baselines \
    --output magnitude_wanda_summary

# 然后手动筛选 JSON 结果
```

### 提取特定信息

使用 Python 脚本处理 JSON：

```python
import json

with open('models_structure_summary.json', 'r') as f:
    data = json.load(f)

# 找出剪枝比例最高的层
for model in data['models']:
    if 'layer_pruning' in model:
        max_pruning_layer = max(
            model['layer_pruning'],
            key=lambda x: x['reduction_ratio']
        )
        print(f"{model['name']}: Layer {max_pruning_layer['layer_idx']} "
              f"pruned {max_pruning_layer['reduction_ratio']*100:.2f}%")
```

### 对比不同剪枝比例的效果

```bash
# 分别汇总 2000 和 5000 的模型
python evaluation/summarize_model_structures.py \
    --dirs results \
    --output results_2000_5000

# 分析输出的 JSON 文件
```

## 🛠️ 故障排除

### 问题 1: 找不到分析结果

**症状**: `⊘ 模型名称 [无分析结果]`

**原因**: 该模型目录下没有 `analysis/` 文件夹或分析文件不完整

**解决**:
1. 检查是否有 `analysis/model_comparison.json`
2. 如果有模型文件，重新运行分析：
   ```bash
   python core/analysis/model_analysis.py \
       --model_path <模型路径> \
       --compare_with <原始模型>
   ```

### 问题 2: SliceGPT 模型无法分析

**症状**: SliceGPT 模型显示"无分析结果"

**原因**: SliceGPT 使用特殊结构，需要在 `slicegpt` 环境中处理

**解决**: 参考 [SLICEGPT_CONVERSION.md](./SLICEGPT_CONVERSION.md)

### 问题 3: JSON 文件损坏

**症状**: `无法加载 xxx.json: ...`

**原因**: 分析文件可能在生成时中断

**解决**: 重新生成该模型的分析结果

## 🎯 最佳实践

### 1. 定期汇总

每次完成新模型的训练和评估后，运行汇总：

```bash
# 训练完成后
python evaluation/summarize_model_structures.py \
    --dirs baselines results \
    --output models_structure_summary_$(date +%Y%m%d)
```

### 2. 版本管理

保存不同时间点的汇总结果：

```bash
mkdir -p summaries/
python evaluation/summarize_model_structures.py \
    --dirs baselines results \
    --output summaries/summary_$(date +%Y%m%d)
```

### 3. 结合性能评估

同时使用结构分析和性能评估：

```bash
# 1. 结构分析
python evaluation/summarize_model_structures.py \
    --dirs baselines results \
    --output structure_summary

# 2. 性能汇总
python core/visualization/generate_results_table.py \
    --result_dir results \
    --output performance_summary.xlsx
```

## 📚 相关文档

- [批量分析说明](./BATCH_ANALYSIS.md) - `batch_model_analysis.py` 详细文档
- [SliceGPT 处理指南](./SLICEGPT_CONVERSION.md) - SliceGPT 特殊处理
- [模型分析 API](../core/analysis/model_analysis.py) - 单模型分析 API

## 🔗 工作流程示例

### 完整的模型分析流程

```bash
# 1. 训练模型
python baselines/run_magnitude.py ...

# 2. 评估性能
python evaluation/run_evaluation.py ...

# 3. 分析结构（如果有模型文件）
python core/analysis/model_analysis.py \
    --model_path results/MyModel/pruned_model.bin \
    --compare_with /newdata/LLMs/Llama-3-8B-Instruct

# 4. 汇总所有模型
python evaluation/summarize_model_structures.py \
    --dirs results \
    --output final_summary

# 5. 生成性能表格
python core/visualization/generate_results_table.py \
    --result_dir results \
    --output performance_table.xlsx
```

### 批量处理多个模型

```bash
# 假设有多个新训练的模型
for model in results/NewModel_*/; do
    echo "分析: $model"
    python core/analysis/model_analysis.py \
        --model_path "$model/pruned_model.bin" \
        --compare_with /newdata/LLMs/Llama-3-8B-Instruct
done

# 汇总
python evaluation/summarize_model_structures.py \
    --dirs results \
    --output batch_summary
```

## 💡 提示

1. **汇总工具很快**: `summarize_model_structures.py` 只读取已有的 JSON 文件，速度很快
2. **JSON 格式便于二次处理**: 可以用 Python/jq 等工具进一步分析
3. **结合可视化**: 可以将 JSON 数据导入 Excel/pandas 做可视化分析
4. **保留历史记录**: 建议保存不同版本的汇总结果，便于追踪模型演进

## 📊 数据流程图

```
模型训练
   ↓
[pruned_model.bin]
   ↓
model_analysis.py → analysis/model_comparison.json
   ↓
summarize_model_structures.py → models_structure_summary.json
   ↓
[进一步分析 / 可视化]
```

## 🎓 总结

- **快速汇总**: 使用 `summarize_model_structures.py`
- **详细分析**: 使用 `model_analysis.py`
- **批量处理**: 使用 `batch_model_analysis.py`
- **性能评估**: 使用 `generate_results_table.py`

根据你的需求选择合适的工具！
