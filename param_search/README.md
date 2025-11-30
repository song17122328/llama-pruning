# 参数搜索和相关性分析工具

本文件夹包含参数搜索和梯度统计相关性分析的所有工具和文档。

## 📂 文件结构

```
param_search/
├── search_best_params.py           # 参数网格搜索脚本
├── analyze_param_correlations.py   # 相关性分析脚本
├── docs/                           # 文档目录
│   ├── README_CORRELATION_ANALYSIS.md       # 相关性分析总览
│   ├── QUICK_START_PARAM_SEARCH.md          # 快速开始指南
│   ├── PARAM_SEARCH_GUIDE.md                # 参数搜索详细指南
│   └── CORRELATION_ANALYSIS_GUIDE.md        # 相关性分析详细指南
└── README.md                       # 本文件
```

## 🚀 快速开始

### 1. 参数搜索

```bash
# 运行参数搜索（测试多种参数组合）
python param_search/search_best_params.py --config configs/mistral_param_search.json
```

### 2. 相关性分析

```bash
# 分析梯度统计指标与 ACC 的相关性
python param_search/analyze_param_correlations.py \
    --results results/param_search_mistral_20/search_results.csv
```

## 📖 详细文档

- **快速开始**: [docs/QUICK_START_PARAM_SEARCH.md](docs/QUICK_START_PARAM_SEARCH.md)
- **参数搜索指南**: [docs/PARAM_SEARCH_GUIDE.md](docs/PARAM_SEARCH_GUIDE.md)
- **相关性分析总览**: [docs/README_CORRELATION_ANALYSIS.md](docs/README_CORRELATION_ANALYSIS.md)
- **相关性分析详细指南**: [docs/CORRELATION_ANALYSIS_GUIDE.md](docs/CORRELATION_ANALYSIS_GUIDE.md)

## 🎯 功能概述

### search_best_params.py

**功能**: 自动化参数网格搜索，测试不同的 Taylor 重要性计算参数组合

**主要参数**:
- `taylor_seq_len`: 序列长度 (如 [16, 32, 64, 128, 256])
- `taylor_num_samples`: 样本数量 (如 [128, 256, 512])

**输出**:
- `search_results.csv`: 所有实验结果
- `best_config.json`: 最佳配置
- 包含 7 个 zero-shot 任务的单独 ACC 和平均值
- 梯度统计指标（grad_norm_ratio, grad_mean_ratio 等）

### analyze_param_correlations.py

**功能**: 分析梯度统计指标与剪枝后性能（ACC）的相关性

**输出**:
- `correlation_heatmap.png`: 相关性热力图
- `scatter_matrix.png`: 散点图矩阵
- `correlation_report.txt`: 详细分析报告
- `prediction_model.json`: 性能预测模型

**科研价值**:
- 发现梯度指标与性能的关系
- 建立预测模型，无需完整评估即可预测性能
- 提供理论支撑，适合写入科研论文

## 📊 收集的指标

### ACC 指标（7 个任务）

- `acc_boolq`, `acc_piqa`, `acc_hellaswag`, `acc_winogrande`
- `acc_arc_easy`, `acc_arc_challenge`, `acc_openbookqa`
- **`acc_mean`**: 7 个任务平均值 ⭐

### 梯度统计指标

- **`grad_norm_ratio`** ⭐: 梯度范数比率（最大/最小）
- `grad_mean_ratio`: 梯度均值比率
- `grad_std_ratio`: 梯度标准差比率
- `grad_max_ratio`: 梯度最大值比率
- `extreme_pruning_layers`: 极端剪枝层数量（>80%）

## 📝 示例脚本

查看 `examples/` 文件夹中的示例脚本：
- `examples/quick_param_search.sh`: 快速参数搜索示例
- `examples/analyze_correlations_example.sh`: 完整相关性分析示例

## 🔬 科研应用

这些工具特别适合科研论文写作：

1. **发现规律**: 通过相关性分析找出梯度统计指标与性能的关系
2. **快速预测**: 无需完整评估即可预测剪枝后性能
3. **理论解释**: 提供为什么某些参数配置更好的理论支撑
4. **可重复性**: 完整记录所有参数组合和结果

## ⚙️ 配置文件

配置文件位于 `configs/` 目录：
- `configs/mistral_param_search.json`: Mistral 模型参数搜索配置（15 种组合）
- `configs/quick_param_search.json`: 快速测试配置（3 种组合）

## 🎓 更多信息

详细使用方法和科研应用，请参阅文档目录中的各个指南。
