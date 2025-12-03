# 参数搜索结果分析脚本使用说明

本目录包含用于分析6个模型（Llama, Qwen, Mistral 及其 Instruct 版本）参数搜索结果的完整工具链。

## 脚本概览

### 1. 单模型结果汇总

**脚本**: `consolidate_model_results.py`

**功能**: 汇总单个模型的所有剪枝方法（Taylor、Layerwise、Blockwise）的结果

**用法**:
```bash
# Base 模型
python param_search/consolidate_model_results.py --model Llama
python param_search/consolidate_model_results.py --model Qwen
python param_search/consolidate_model_results.py --model Mistral

# Instruct 模型
python param_search/consolidate_model_results.py --model Llama-Instruct
python param_search/consolidate_model_results.py --model Qwen-Instruct
python param_search/consolidate_model_results.py --model Mistral-Instruct
```

**输出**:
- `results/consolidated_{model}_20/all_methods_results.csv` - 所有剪枝方法的完整结果
- `results/consolidated_{model}_20/global_best_config.json` - 全局最佳配置
- `results/consolidated_{model}_20/method_comparison.json` - 剪枝方法对比统计

### 2. 批量汇总所有模型

**脚本**: `consolidate_all_models.py`

**功能**: 自动对所有6个模型运行汇总

**用法**:
```bash
python param_search/consolidate_all_models.py
```

**输出**: 为每个模型生成 `consolidated_{model}_20` 目录及其中的结果文件

### 3. 跨模型综合分析

**脚本**: `analyze_all_models.py`

**功能**: 对比分析所有6个模型的最佳配置，包括：
- 所有模型最佳配置总览
- Base vs Instruct 对比
- 剪枝方法偏好统计
- 参数分布统计
- 模型架构对比（Llama vs Qwen vs Mistral）

**用法**:
```bash
python param_search/analyze_all_models.py
```

**输出**:
- `results/cross_model_analysis/all_models_best_configs.csv` - 所有模型最佳配置对比表
- `results/cross_model_analysis/analysis_summary.json` - 统计摘要（JSON格式）
- 终端输出完整的分析报告

### 4. Instruct 模型结果提取

**脚本**: `extract_instruct_results.py`

**功能**: 批量提取所有 Instruct 模型的实验结果（仅在首次需要时使用）

**用法**:
```bash
python param_search/extract_instruct_results.py
```

**说明**: 该脚本会自动扫描所有 Instruct 模型的搜索目录并提取结果到 CSV 文件

### 5. 单目录结果重提取

**脚本**: `re_extract_results.py`

**功能**: 从单个搜索目录重新提取结果（用于修复或更新现有结果）

**用法**:
```bash
python param_search/re_extract_results.py --search_dir results/search_Llama_20
```

## 完整工作流程

### 首次分析

```bash
# Step 1: 提取 Instruct 模型结果（如果还没有 CSV 文件）
python param_search/extract_instruct_results.py

# Step 2: 汇总所有模型的结果
python param_search/consolidate_all_models.py

# Step 3: 生成跨模型综合分析
python param_search/analyze_all_models.py
```

### 后续分析（已有 CSV 文件）

```bash
# 直接运行步骤 2 和 3
python param_search/consolidate_all_models.py
python param_search/analyze_all_models.py
```

## 关键发现摘要（基于当前结果）

### 最佳模型配置

| 模型 | 类型 | 方法 | ACC | PPL | seq_len | samples |
|------|------|------|-----|-----|---------|---------|
| **Mistral** | **Instruct** | **BLOCKWISE** | **0.6552** | 24.33 | 32 | 256 |
| Llama | Instruct | BLOCKWISE | 0.6318 | 13.29 | 32 | 512 |
| Qwen | Instruct | LAYERWISE | 0.6202 | 13.42 | 32 | 4 |
| Qwen | Base | LAYERWISE | 0.6161 | 10.80 | 128 | 512 |
| Llama | Base | BLOCKWISE | 0.5980 | 13.17 | 64 | 128 |
| Mistral | Base | BLOCKWISE | 0.5947 | 13.29 | 64 | 128 |

### Base vs Instruct 性能提升

- **Mistral**: +10.18% (0.5947 → 0.6552) - 最大提升
- **Llama**: +5.64% (0.5980 → 0.6318)
- **Qwen**: +0.66% (0.6161 → 0.6202) - 最小提升，Base 已经很强

### 剪枝方法偏好

- **BLOCKWISE**: 4/6 模型 (66.7%) - 最受欢迎
- **LAYERWISE**: 2/6 模型 (33.3%) - Qwen 系列偏好
- **TAYLOR**: 0/6 模型 (0.0%) - 未被选为最佳

### 参数偏好

**taylor_seq_len**:
- 32: 3/6 模型 (50.0%) - **Instruct 模型全部使用 32**
- 64: 2/6 模型 (33.3%) - Llama/Mistral Base
- 128: 1/6 模型 (16.7%) - Qwen Base

**taylor_num_samples**:
- 分布较为均匀：4, 128, 256, 512 各有模型使用
- Instruct 模型倾向更多样本（4-512）

### 架构对比（平均 Base + Instruct）

| 架构 | 平均 ACC | 平均 PPL |
|------|---------|---------|
| Mistral | 0.6249 | 18.81 |
| Qwen | 0.6181 | 12.11 |
| Llama | 0.6149 | 13.23 |

## 重要观察

1. **Instruct 模型普遍偏好更小的 seq_len (32)**，而 Base 模型需要更大的值
2. **BLOCKWISE 方法在大多数情况下表现最好**，尤其是 Llama 和 Mistral
3. **Qwen 是唯一偏好 LAYERWISE 方法的架构**（Base 和 Instruct 都是）
4. **Mistral-Instruct 取得最高 ACC (0.6552)**，但 PPL 较高 (24.33)
5. **Qwen Base 在所有 Base 模型中表现最好** (ACC: 0.6161, PPL: 10.80)

## 用于论文的数据

所有分析结果都已保存为 CSV 和 JSON 格式，可直接用于论文：
- 表格数据: `results/cross_model_analysis/all_models_best_configs.csv`
- 统计数据: `results/cross_model_analysis/analysis_summary.json`
- 单模型详细数据: `results/consolidated_{model}_20/`
