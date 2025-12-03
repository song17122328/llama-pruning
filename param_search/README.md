# 参数搜索和结果分析工具

本目录包含用于 LLaMA、Qwen、Mistral 及其 Instruct 版本的参数搜索和结果分析的完整工具链。

## 📂 目录结构

```
param_search/
├── search_best_params.py          # 主参数搜索脚本
├── consolidate_model_results.py   # 单模型结果汇总
├── consolidate_all_models.py      # 批量汇总所有6个模型
├── analyze_all_models.py          # 跨模型综合分析
├── re_extract_results.py          # 重提取结果工具
├── copy_best_results.py           # 复制最佳结果到独立目录
└── README.md                      # 本文件
```

## 🚀 快速开始

### 完整工作流程

```bash
# 1. 运行参数搜索（针对某个模型）
python param_search/search_best_params.py --config configs/Llama_param_search.json

# 2. 如果需要重提取结果（可选）
python param_search/re_extract_results.py --search_dir results/search_Llama_20

# 3. 汇总单个模型的所有剪枝方法结果
python param_search/consolidate_model_results.py --model Llama

# 4. 批量汇总所有6个模型
python param_search/consolidate_all_models.py

# 5. 生成跨模型综合分析报告
python param_search/analyze_all_models.py
```

## 📖 脚本详解

### 1. search_best_params.py - 主参数搜索

**功能**: 自动化参数网格搜索，测试不同的 Taylor 重要性计算参数组合

**用法**:
```bash
# Base 模型
python param_search/search_best_params.py --config configs/Llama_param_search.json
python param_search/search_best_params.py --config configs/Qwen_param_search.json
python param_search/search_best_params.py --config configs/Mistral_param_search.json

# Instruct 模型
python param_search/search_best_params.py --config configs/Llama-Instruct_param_search.json
python param_search/search_best_params.py --config configs/Qwen-Instruct_param_search.json
python param_search/search_best_params.py --config configs/Mistral-Instruct_param_search.json

# 使用 --resume 参数继续中断的搜索
python param_search/search_best_params.py --config configs/Llama_param_search.json --resume
```

**配置参数**:
- `taylor_seq_len`: 序列长度（如 [32, 64, 128, 256]）
- `taylor_num_samples`: 样本数量（如 [4, 64, 128, 256, 512]）
- `pruning_ratio`: 剪枝率（如 0.2 表示 20%）
- `importance_method`: 重要性计算方法（taylor, layerwise, blockwise）

**输出**:
- `results/search_{model}_20/search_results.csv` - 所有实验结果
- `results/search_{model}_20/best_config.json` - 最佳配置
- `results/search_{model}_20/exp_*` - 每个实验的详细结果

**收集的指标**:
- **ACC 指标**: 7个 zero-shot 任务（BoolQ, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA）
- **PPL**: WikiText2 和 PTB 数据集上的困惑度
- **梯度统计**: grad_norm_ratio, grad_mean_ratio, extreme_pruning_layers 等

---

### 2. re_extract_results.py - 重提取结果

**功能**: 从已完成的实验目录重新提取结果到 CSV（用于修复或更新）

**用法**:
```bash
python param_search/re_extract_results.py --search_dir results/search_Llama_20
```

**适用场景**:
- CSV 文件损坏或丢失
- 需要更新输出格式
- 修复参数提取错误

---

### 3. consolidate_model_results.py - 单模型结果汇总

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

---

### 4. consolidate_all_models.py - 批量汇总所有模型

**功能**: 自动对所有6个模型运行汇总

**用法**:
```bash
python param_search/consolidate_all_models.py
```

**说明**: 等价于对每个模型运行 `consolidate_model_results.py`

---

### 5. analyze_all_models.py - 跨模型综合分析 ⭐

**功能**: 对比分析所有6个模型的最佳配置，生成综合报告

**用法**:
```bash
python param_search/analyze_all_models.py
```

**分析内容**:
1. 所有模型最佳配置总览
2. Base vs Instruct 性能对比
3. 剪枝方法偏好统计
4. 参数分布统计（taylor_seq_len, taylor_num_samples）
5. 模型架构对比（Llama vs Qwen vs Mistral）

**输出**:
- `results/cross_model_analysis/all_models_best_configs.csv` - 所有模型最佳配置对比表
- `results/cross_model_analysis/analysis_summary.json` - 统计摘要（JSON格式）
- 终端输出完整的分析报告

**示例输出**:
```
====================================================================================================
所有模型最佳配置总览
====================================================================================================
模型                   类型         方法           ACC        PPL        seq_len    samples
----------------------------------------------------------------------------------------------------
Llama                Base       BLOCKWISE    0.5980     13.17      64         128
Qwen                 Base       LAYERWISE    0.6161     10.80      128        512
Mistral              Base       BLOCKWISE    0.5947     13.29      64         128
----------------------------------------------------------------------------------------------------
Llama                Instruct   BLOCKWISE    0.6318     13.29      32         512
Qwen                 Instruct   LAYERWISE    0.6202     13.42      32         4
Mistral              Instruct   BLOCKWISE    0.6552     24.33      32         256
```

---

### 6. copy_best_results.py - 复制最佳结果

**功能**: 将最佳实验结果复制到独立目录以便查看和分析

**用法**:
```bash
python param_search/copy_best_results.py --model Llama
python param_search/copy_best_results.py --all  # 复制所有模型
```

**输出**: `results/best_{model}_20/` 目录

---

## 📊 关键发现（基于当前实验结果）

### 最佳模型配置

| 排名 | 模型 | 类型 | 方法 | ACC | PPL | seq_len | samples |
|------|------|------|------|-----|-----|---------|---------|
| 🥇 | **Mistral** | **Instruct** | **BLOCKWISE** | **0.6552** | 24.33 | 32 | 256 |
| 🥈 | Llama | Instruct | BLOCKWISE | 0.6318 | 13.29 | 32 | 512 |
| 🥉 | Qwen | Instruct | LAYERWISE | 0.6202 | 13.42 | 32 | 4 |
| 4 | Qwen | Base | LAYERWISE | 0.6161 | 10.80 | 128 | 512 |
| 5 | Llama | Base | BLOCKWISE | 0.5980 | 13.17 | 64 | 128 |
| 6 | Mistral | Base | BLOCKWISE | 0.5947 | 13.29 | 64 | 128 |

### Base vs Instruct 性能提升

- **Mistral**: +10.18% (0.5947 → 0.6552) - 🔥 最大提升
- **Llama**: +5.64% (0.5980 → 0.6318)
- **Qwen**: +0.66% (0.6161 → 0.6202) - Base 已经很强

### 剪枝方法偏好

- **BLOCKWISE**: 4/6 模型 (66.7%) - 最受欢迎
- **LAYERWISE**: 2/6 模型 (33.3%) - Qwen 系列专属偏好
- **TAYLOR**: 0/6 模型 (0.0%) - 未被选为最佳

### 参数规律发现 🔍

**taylor_seq_len**:
- **32**: 3/6 模型 (50.0%) - **所有 Instruct 模型都使用 32**
- **64**: 2/6 模型 (33.3%) - Llama/Mistral Base
- **128**: 1/6 模型 (16.7%) - Qwen Base

**关键观察**: Instruct 模型普遍偏好更小的 seq_len (32)，而 Base 模型需要更大的值

**taylor_num_samples**:
- 分布较为均匀：4, 128, 256, 512 各有模型使用
- Qwen-Instruct 仅需 4 个样本即可达到最佳性能（极高效率）

### 架构对比（平均 Base + Instruct）

| 排名 | 架构 | 平均 ACC | 平均 PPL |
|------|------|---------|---------|
| 🥇 | Mistral | 0.6249 | 18.81 |
| 🥈 | Qwen | 0.6181 | 12.11 ⭐ 最低 PPL |
| 🥉 | Llama | 0.6149 | 13.23 |

---

## 💡 重要观察

1. **Instruct 模型的特殊性**:
   - 全部使用 `taylor_seq_len=32`（更小的序列长度）
   - Base 模型需要 64-128 的更大值
   - 这可能与 Instruct 模型的对齐训练有关

2. **剪枝方法选择**:
   - **BLOCKWISE** 在大多数情况下表现最好（尤其是 Llama 和 Mistral）
   - **Qwen 是唯一偏好 LAYERWISE 的架构**（Base 和 Instruct 都是）
   - **TAYLOR** 方法从未成为最佳（可能需要优化或不适合这个任务）

3. **性能与效率权衡**:
   - **Mistral-Instruct**: 最高 ACC (0.6552) 但 PPL 较高 (24.33)
   - **Qwen Base**: 所有 Base 模型中表现最好 (ACC: 0.6161, PPL: 10.80)
   - **Qwen-Instruct**: 最高效率（仅需 4 个样本）

4. **PPL 与 ACC 的关系**:
   - 两者不完全正相关
   - Mistral-Instruct 虽然 PPL 高但 ACC 最好
   - 在剪枝场景中，zero-shot ACC 可能比 PPL 更重要

---

## 📈 用于论文的数据

所有分析结果都已保存为 CSV 和 JSON 格式，可直接用于论文：

- **表格数据**: `results/cross_model_analysis/all_models_best_configs.csv`
- **统计数据**: `results/cross_model_analysis/analysis_summary.json`
- **单模型详细数据**: `results/consolidated_{model}_20/`

### 建议的论文呈现方式

1. **主表**: 展示所有6个模型的最佳配置（使用 all_models_best_configs.csv）
2. **对比图**: Base vs Instruct 性能提升柱状图
3. **分布图**: 参数偏好分布（seq_len 和 num_samples）
4. **方法对比**: 三种剪枝方法的 ACC 对比（按模型分组）

---

## 🔧 配置文件

配置文件位于 `configs/` 目录：

**Base 模型**:
- `configs/Llama_param_search.json`
- `configs/Qwen_param_search.json`
- `configs/Mistral_param_search.json`

**Instruct 模型**:
- `configs/Llama-Instruct_param_search.json`
- `configs/Qwen-Instruct_param_search.json`
- `configs/Mistral-Instruct_param_search.json`

**剪枝方法变体**:
- `configs/{model}_layerwise_param_search.json`
- `configs/{model}_blockwise_param_search.json`

---

## ⚙️ 实验环境要求

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- 足够的 GPU 内存（建议 40GB+ 用于 7B-8B 模型）

---

## 📝 常见问题

**Q: 实验中断了怎么办？**
A: 使用 `--resume` 参数继续：
```bash
python param_search/search_best_params.py --config configs/Llama_param_search.json --resume
```

**Q: CSV 文件损坏了怎么办？**
A: 使用 `re_extract_results.py` 重新提取：
```bash
python param_search/re_extract_results.py --search_dir results/search_Llama_20
```

**Q: 如何快速测试流程？**
A: 创建一个小的测试配置，只使用 2-3 个参数组合

**Q: 分析脚本报错找不到文件？**
A: 确保先运行了 `consolidate_all_models.py` 生成汇总文件

---

## 🎯 后续优化方向

基于当前实验结果，建议：

1. **针对 TAYLOR 方法**:
   - 可能需要调整 H-GSP 的温度参数和门控阈值
   - 尝试二阶 Taylor 展开（`importance_method: taylor_2nd`）

2. **针对 Instruct 模型**:
   - 探索为什么 seq_len=32 总是最优
   - 研究对齐训练对重要性评估的影响

3. **针对 Qwen**:
   - 深入分析为什么偏好 LAYERWISE
   - 研究其架构特点（如 GQA ratio, layer 数量等）

---

## 📚 相关文档

- 主项目 README: `../README.md`
- 配置文件说明: `../configs/README.md`
- 剪枝方法文档: `../docs/`

---

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。
