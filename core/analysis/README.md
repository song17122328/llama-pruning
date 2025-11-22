# 模型分析模块使用说明

## 功能概述

模型分析模块提供了详细的模型参数统计和剪枝前后对比功能，包括：

1. **参数统计**: 统计模型的总参数量、每层参数量（Attention、MLP、LayerNorm）
2. **剪枝对比**: 对比剪枝前后的差异，计算实际剪枝比例
3. **详细报告**: 生成 JSON 格式的分析报告，便于后续分析
4. **独立运行**: 可以作为独立脚本运行，也可以被其他模块导入使用

## 集成在 run_global_pruning.py 中的使用

在运行 `run_global_pruning.py` 进行剪枝时，会自动生成以下分析报告：

### 自动生成的报告

剪枝完成后，在输出目录的 `analysis/` 子目录下会自动生成以下文件：

```
results/{output_name}/
├── analysis/
│   ├── original_model_analysis.json      # 原始模型的参数分析
│   ├── pruned_model_analysis.json        # 剪枝后模型的参数分析
│   └── model_comparison.json             # 剪枝前后对比报告
```

### 报告内容

**1. 原始模型分析 (`original_model_analysis.json`)**
```json
{
  "model_name": "原始模型",
  "total_params": 6738415616,
  "embedding_params": 262144000,
  "lm_head_params": 262144000,
  "layers": [
    {
      "layer_idx": 0,
      "total": 67108864,
      "attention": {
        "type": "LlamaAttention",
        "total": 33554432,
        "q_proj": 16777216,
        "k_proj": 4194304,
        "v_proj": 4194304,
        "o_proj": 8388608,
        "num_heads": 32,
        "num_kv_heads": 8
      },
      "mlp": {
        "type": "LlamaMLP",
        "total": 33554432,
        "gate_proj": 11184810,
        "up_proj": 11184810,
        "down_proj": 11184812,
        "intermediate_size": 11008
      },
      "norm": 8192,
      "is_zero_layer": false
    }
    // ... 更多层
  ]
}
```

**2. 对比报告 (`model_comparison.json`)**
```json
{
  "original_name": "原始模型",
  "pruned_name": "剪枝后模型",
  "total_params": {
    "original": 6738415616,
    "pruned": 5390732492,
    "reduced": 1347683124,
    "reduction_ratio": 0.2
  },
  "layer_params": {
    "original": 6214127616,
    "pruned": 4866444492,
    "reduced": 1347683124,
    "reduction_ratio": 0.217
  },
  "layers": [
    {
      "layer_idx": 0,
      "total": {
        "original": 67108864,
        "pruned": 53687296,
        "reduced": 13421568,
        "reduction_ratio": 0.2
      },
      "attention": {
        "original": 33554432,
        "pruned": 26843520,
        "reduced": 6710912,
        "reduction_ratio": 0.2,
        "num_heads": {
          "original": 32,
          "pruned": 26
        },
        "num_kv_heads": {
          "original": 8,
          "pruned": 6
        }
      },
      "mlp": {
        "original": 33554432,
        "pruned": 26843776,
        "reduced": 6710656,
        "reduction_ratio": 0.2,
        "intermediate_size": {
          "original": 11008,
          "pruned": 8806
        }
      },
      "is_zero_layer": false
    }
    // ... 更多层
  ]
}
```

### 日志输出

在剪枝过程中，日志会显示详细的对比信息：

```
[Step 8.5] 生成详细的模型分析报告...
  ✓ 剪枝后模型分析完成
  ✓ 对比分析完成
  ✓ 原始模型分析已保存: results/my_pruning/analysis/original_model_analysis.json
  ✓ 剪枝后模型分析已保存: results/my_pruning/analysis/pruned_model_analysis.json
  ✓ 对比报告已保存: results/my_pruning/analysis/model_comparison.json

============================================================
详细对比报告
============================================================

总参数量:
  原始: 6,738,415,616
  剪枝后: 5,390,732,492
  减少: 1,347,683,124 (20.00%)

Decoder Layers 参数:
  原始: 6,214,127,616
  剪枝后: 4,866,444,492
  减少: 1,347,683,124 (21.68%)

每层剪枝详情:
------------------------------------------------------------

Layer  0:
  总参数: 67,108,864 → 53,687,296 (-20.00%)
  Attention: 33,554,432 → 26,843,520 (-20.00%)
    头数: 32Q:8KV → 26Q:6KV
  MLP: 33,554,432 → 26,843,776 (-20.00%)
    中间维度: 11008 → 8806

Layer  1 [完全剪空]:
  总参数: 67,108,864 → 0 (-100.00%)
  Attention: 33,554,432 → 0 (-100.00%)
    头数: 32Q:8KV → 0Q:0KV
  MLP: 33,554,432 → 0 (-100.00%)
    中间维度: 11008 → 0

...
```

## 作为独立脚本使用

### 基本用法

```bash
# 1. 分析单个模型
python core/analysis/model_analysis.py \
    --model_path /path/to/model \
    --model_name "我的模型" \
    --save_json analysis.json

# 2. 对比两个模型（剪枝前后）
python core/analysis/model_analysis.py \
    --model_path /path/to/original_model \
    --model_name "原始模型" \
    --compare_with /path/to/pruned_model \
    --compare_name "剪枝后模型" \
    --save_json comparison.json
```

### 参数说明

- `--model_path`: （必需）模型路径（HuggingFace 模型名或本地路径）
- `--model_name`: 模型名称（用于报告，默认使用路径名）
- `--save_json`: 保存 JSON 报告的路径（可选）
- `--compare_with`: 对比的模型路径（可选，用于对比剪枝前后）
- `--compare_name`: 对比模型的名称（可选）
- `--verbose`: 显示详细信息（每层参数）

### 示例

```bash
# 分析一个已剪枝的模型
python core/analysis/model_analysis.py \
    --model_path results/my_pruning/models/pruned_model.bin \
    --model_name "剪枝后的模型" \
    --save_json pruned_analysis.json \
    --verbose

# 对比原始模型和剪枝后的模型
python core/analysis/model_analysis.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --model_name "Llama-2-7B 原始" \
    --compare_with results/my_pruning/models/pruned_model.bin \
    --compare_name "Llama-2-7B 剪枝20%" \
    --save_json comparison_report.json
```

## 在 Python 代码中使用

### 导入模块

```python
from core.analysis import ModelAnalyzer, ModelComparator
from transformers import AutoModelForCausalLM
```

### 分析单个模型

```python
# 加载模型
model = AutoModelForCausalLM.from_pretrained("path/to/model")

# 创建分析器
analyzer = ModelAnalyzer(model, "我的模型")

# 运行分析
result = analyzer.analyze()

# 打印报告
analyzer.print_report(verbose=True)

# 保存 JSON 报告
analyzer.save_report("analysis.json")
```

### 对比两个模型

```python
# 加载两个模型
original_model = AutoModelForCausalLM.from_pretrained("original_model")
pruned_model = AutoModelForCausalLM.from_pretrained("pruned_model")

# 分析两个模型
original_analyzer = ModelAnalyzer(original_model, "原始模型")
original_analysis = original_analyzer.analyze()

pruned_analyzer = ModelAnalyzer(pruned_model, "剪枝后模型")
pruned_analysis = pruned_analyzer.analyze()

# 创建对比器
comparator = ModelComparator(
    original_analysis=original_analysis,
    pruned_analysis=pruned_analysis,
    original_name="原始模型",
    pruned_name="剪枝后模型"
)

# 运行对比
comparison_result = comparator.compare()

# 打印对比报告
comparator.print_report(verbose=True)

# 保存对比报告
comparator.save_report("comparison.json")
```

## 报告字段说明

### ModelAnalyzer 输出字段

- `model_name`: 模型名称
- `total_params`: 总参数量
- `embedding_params`: Embedding 层参数量
- `lm_head_params`: LM Head 层参数量
- `layer_summary`: 层的汇总信息
  - `num_layers`: 层数
  - `total_layer_params`: 所有 decoder layer 的总参数量
- `layers`: 每一层的详细信息
  - `layer_idx`: 层索引
  - `total`: 该层总参数量
  - `attention`: Attention 模块信息
    - `type`: 模块类型（LlamaAttention 或 ZeroAttention）
    - `total`: Attention 总参数量
    - `q_proj`, `k_proj`, `v_proj`, `o_proj`: 各投影层参数量
    - `num_heads`: Query 头数
    - `num_kv_heads`: Key/Value 头数
  - `mlp`: MLP 模块信息
    - `type`: 模块类型（LlamaMLP 或 ZeroMLP）
    - `total`: MLP 总参数量
    - `gate_proj`, `up_proj`, `down_proj`: 各投影层参数量
    - `intermediate_size`: 中间层维度
  - `norm`: LayerNorm 参数量
  - `is_zero_layer`: 是否为完全剪空的层

### ModelComparator 输出字段

- `original_name`: 原始模型名称
- `pruned_name`: 剪枝后模型名称
- `total_params`: 总参数对比
  - `original`: 原始参数量
  - `pruned`: 剪枝后参数量
  - `reduced`: 减少的参数量
  - `reduction_ratio`: 减少比例（0-1）
- `layer_params`: Decoder Layers 参数对比（结构同上）
- `layers`: 每一层的对比信息
  - `layer_idx`: 层索引
  - `total`: 该层总参数对比
  - `attention`: Attention 参数对比
    - `original`, `pruned`, `reduced`, `reduction_ratio`: 参数量对比
    - `num_heads`: 头数对比（original, pruned）
    - `num_kv_heads`: KV 头数对比
  - `mlp`: MLP 参数对比
    - `original`, `pruned`, `reduced`, `reduction_ratio`: 参数量对比
    - `intermediate_size`: 中间维度对比
  - `is_zero_layer`: 是否为完全剪空的层

## 注意事项

1. **ZeroAttention 和 ZeroMLP**: 完全剪空的模块会被替换为 ZeroAttention 或 ZeroMLP，这些模块没有参数，利用残差连接返回零值
2. **GQA (Grouped Query Attention)**: 对于使用 GQA 的模型（如 Llama），会显示 Q 头数和 KV 头数
3. **内存占用**: 分析大模型时可能需要较多内存，建议在 GPU 上运行
4. **JSON 报告**: 所有报告都保存为 JSON 格式，便于后续分析和可视化

## 输出示例

运行剪枝后的完整输出示例：

```
[Step 8.5] 生成详细的模型分析报告...
  ✓ 剪枝后模型分析完成
  ✓ 对比分析完成
  ✓ 原始模型分析已保存: results/llama2-7b_pruning_0.2/analysis/original_model_analysis.json
  ✓ 剪枝后模型分析已保存: results/llama2-7b_pruning_0.2/analysis/pruned_model_analysis.json
  ✓ 对比报告已保存: results/llama2-7b_pruning_0.2/analysis/model_comparison.json

============================================================
详细对比报告
============================================================

总参数量:
  原始: 6,738,415,616
  剪枝后: 5,390,732,492
  减少: 1,347,683,124 (20.00%)

Decoder Layers 参数:
  原始: 6,214,127,616
  剪枝后: 4,866,444,492
  减少: 1,347,683,124 (21.68%)

每层剪枝详情:
------------------------------------------------------------

Layer  0:
  总参数: 67,108,864 → 53,687,296 (-20.00%)
  Attention: 33,554,432 → 26,843,520 (-20.00%)
    头数: 32Q:8KV → 26Q:6KV
  MLP: 33,554,432 → 26,843,776 (-20.00%)
    中间维度: 11008 → 8806

完全剪空的层 (3个): [5, 12, 18]

============================================================
```
