# 批量模型结构分析说明

## 功能概述

`batch_model_analysis.py` 是一个批量分析工具，用于自动分析多个剪枝模型的结构参数。

**主要功能：**
- ✅ 自动扫描目录下的所有模型
- ✅ 识别模型类型（标准模型 vs SliceGPT）
- ✅ 批量分析标准模型的结构和参数
- ✅ 对比剪枝前后的差异
- ✅ 为 SliceGPT 生成单独的分析脚本
- ✅ 汇总所有模型的分析结果

## 使用流程

### 第一步：分析标准模型

```bash
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --output_dir baselines_analysis/
```

**参数说明：**
- `--models_dir`: 模型目录（例如 `baselines/`）
- `--base_model`: 原始基础模型路径（用于对比）
- `--output_dir`: 输出目录（默认：`baselines_analysis/`）
- `--device`: 设备（默认：`cuda`）

**输出：**
- 每个模型的分析结果保存在：`<模型目录>/analysis/`
  - `model_structure.json`: 详细的结构信息
  - `model_comparison.json`: 与原模型的对比
  - `structure_summary.txt`: 人类可读的摘要
- 批量分析结果：`baselines_analysis/batch_analysis_results.json`
- SliceGPT 脚本（如有）：`baselines_analysis/analyze_slicegpt.sh`

### 第二步：运行 SliceGPT 分析（可选）

如果有 SliceGPT 模型，脚本会生成 `analyze_slicegpt.sh`。运行：

```bash
conda activate slicegpt
bash baselines_analysis/analyze_slicegpt.sh
```

**注意：** 目前 SliceGPT 需要手动处理，因为其结构与标准模型不同。

### 第三步：汇总所有结果

```bash
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --merge_results \
    --output_dir baselines_analysis/
```

**输出：**
- `baselines_analysis/models_summary.json`: 所有模型的汇总统计

## 模型识别逻辑

脚本会自动识别以下类型的模型：

| 模型类型 | 识别特征 | 处理方式 |
|---------|---------|---------|
| **标准 HF 模型** | 存在 `config.json` | 在当前环境直接分析 |
| **`.bin` 模型** | 存在 `pruned_model.bin` | 在当前环境直接分析 |
| **SliceGPT 模型** | 存在 `*.json` 配置（非 config.json）| 生成单独脚本 |
| **无模型** | 只有评估结果 | 跳过 |

## 输出文件结构

### 每个模型的分析结果

```
baselines/Magnitude_2000/
└── analysis/
    ├── model_structure.json       # 详细结构信息
    ├── model_comparison.json      # 与原模型对比
    └── structure_summary.txt      # 人类可读摘要
```

### `model_structure.json` 格式

```json
{
  "model_name": "Magnitude_2000",
  "total_params": 6234567890,
  "embedding_params": 524288000,
  "lm_head_params": 524288000,
  "layer_summary": {
    "num_layers": 32,
    "total_layer_params": 5186191872
  },
  "layers": [
    {
      "layer_idx": 0,
      "total": 162068480,
      "attention": {
        "type": "LlamaAttention",
        "total": 67108864,
        "q_proj": 16777216,
        "k_proj": 4194304,
        "v_proj": 4194304,
        "o_proj": 16777216,
        "num_heads": 32,
        "num_kv_heads": 8
      },
      "mlp": {
        "type": "LlamaMLP",
        "total": 94371840,
        "gate_proj": 45088768,
        "up_proj": 45088768,
        "down_proj": 45088768,
        "intermediate_size": 14336
      },
      "norm": 8192,
      "is_zero_layer": false
    }
    // ... 更多层
  ]
}
```

### `model_comparison.json` 格式

```json
{
  "original_name": "Llama-3-8B-Instruct",
  "pruned_name": "Magnitude_2000",
  "total_params": {
    "original": 8030000000,
    "pruned": 6234567890,
    "reduced": 1795432110,
    "reduction_ratio": 0.2236
  },
  "layer_params": {
    "original": 6542843904,
    "pruned": 5186191872,
    "reduced": 1356652032,
    "reduction_ratio": 0.2074
  },
  "layers": [
    {
      "layer_idx": 0,
      "total": {
        "original": 204521472,
        "pruned": 162068480,
        "reduced": 42452992,
        "reduction_ratio": 0.2076
      },
      "attention": { /* ... */ },
      "mlp": { /* ... */ },
      "is_zero_layer": false
    }
    // ... 更多层
  ]
}
```

### `models_summary.json` 格式

```json
{
  "timestamp": "2025-11-27T16:30:00",
  "models_dir": "baselines/",
  "models": [
    {
      "name": "Magnitude_2000",
      "total_params": 6234567890,
      "num_layers": 32,
      "layer_params": 5186191872,
      "embedding_params": 524288000,
      "lm_head_params": 524288000,
      "pruning": {
        "original_params": 8030000000,
        "pruned_params": 6234567890,
        "reduction_ratio": 0.2236
      }
    }
    // ... 更多模型
  ]
}
```

## 与现有工具的关系

| 工具 | 功能 | 适用场景 |
|-----|------|---------|
| `model_analysis.py` | 分析单个模型结构 | 详细分析单个模型 |
| `generate_results_table.py` | 汇总评估结果 | 生成性能对比表格 |
| **`batch_model_analysis.py`** | **批量分析模型结构** | **批量获取所有模型的结构信息** |

## 常见问题

### Q1: SliceGPT 模型为什么不能直接分析？

A: SliceGPT 使用了特殊的结构（旋转矩阵、融合 LayerNorm），需要在 `slicegpt` 环境中使用特殊的加载方式。当前脚本会为 SliceGPT 生成单独的分析脚本，但需要手动处理。

### Q2: 如何只分析特定的模型？

A: 可以创建一个临时目录，只包含需要分析的模型的软链接：

```bash
mkdir temp_models
ln -s ../baselines/Magnitude_2000 temp_models/
ln -s ../baselines/Wanda_2000 temp_models/

python evaluation/batch_model_analysis.py \
    --models_dir temp_models/ \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct
```

### Q3: 分析过程中内存不足怎么办？

A: 脚本会在分析每个模型后清理 GPU 内存。如果仍然不足，可以：
1. 指定 `--device cpu` 使用 CPU（较慢）
2. 分批处理模型

### Q4: 如何重新分析某个模型？

A: 删除该模型的 `analysis/` 目录，重新运行脚本：

```bash
rm -rf baselines/Magnitude_2000/analysis/
python evaluation/batch_model_analysis.py ...
```

## 高级用法

### 只分析结构，不对比

如果想单独分析某个模型的结构（不与原模型对比），使用：

```bash
python core/analysis/model_analysis.py \
    --model_path baselines/Magnitude_2000/pruned_model.bin
```

### 自定义输出目录

```bash
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --output_dir my_custom_analysis/
```

### 使用 CPU

```bash
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --device cpu
```

## 注意事项

1. **确保有足够的 GPU 内存**：每个模型需要约 16GB GPU 内存
2. **原始模型路径**：确保 `--base_model` 路径正确
3. **分析时间**：每个模型分析约需 2-5 分钟
4. **SliceGPT 特殊处理**：需要在 `slicegpt` 环境中手动分析

## 示例输出

```
================================================================================
扫描模型目录: baselines/
================================================================================

✓ 标准模型: Magnitude_2000 (类型: huggingface)
✓ 标准模型: Wanda_2000 (类型: huggingface)
✓ 标准模型: ShortGPT_remove_7 (类型: huggingface)
✓ SliceGPT 模型: SliceGPT_2000
⊘ 跳过: Llama-3-8B-Instruct (无模型文件)

汇总:
  - 标准模型: 3 个
  - SliceGPT 模型: 1 个
  - 无模型文件: 1 个

================================================================================
分析标准模型 (3 个)
================================================================================

正在加载原始模型...
✓ 原始模型加载完成

================================================================================
[1/3] 分析模型: Magnitude_2000
================================================================================
模型路径: baselines/Magnitude_2000/pruned_model
正在加载模型: baselines/Magnitude_2000/pruned_model
✓ 模型加载完成
✓ 结构分析已保存: baselines/Magnitude_2000/analysis/model_structure.json
✓ 对比结果已保存: baselines/Magnitude_2000/analysis/model_comparison.json
✓ 摘要已保存: baselines/Magnitude_2000/analysis/structure_summary.txt

...

================================================================================
✓ 标准模型分析完成
✓ 批量分析结果已保存: baselines_analysis/batch_analysis_results.json
================================================================================

================================================================================
生成 SliceGPT 分析脚本
================================================================================

✓ SliceGPT 分析脚本已生成: baselines_analysis/analyze_slicegpt.sh

使用方法:
  conda activate slicegpt
  bash baselines_analysis/analyze_slicegpt.sh

注意: SliceGPT 模型需要在 slicegpt 环境中手动处理

================================================================================
下一步操作:
================================================================================

1. 运行 SliceGPT 分析脚本:
   conda activate slicegpt
   bash baselines_analysis/analyze_slicegpt.sh

2. 汇总所有结果:
   python evaluation/batch_model_analysis.py \
       --models_dir baselines/ \
       --merge_results \
       --output_dir baselines_analysis/

================================================================================
```

## 总结

`batch_model_analysis.py` 提供了一个便捷的方式来批量分析多个剪枝模型的结构，帮助你快速了解：
- 每个模型的参数分布
- 剪枝对不同层的影响
- 各模型之间的结构差异

结合 `generate_results_table.py`，你可以全面了解模型的性能和结构。
