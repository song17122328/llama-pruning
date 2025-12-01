# 实验结果总结

## 📊 数据完整性

所有135个实验全部有效（100%完整率）：

| 模型 | Taylor | Layerwise | Blockwise | 总计 |
|------|--------|-----------|-----------|------|
| Llama | 15 | 15 | 15 | **45** |
| Qwen | 15 | 15 | 15 | **45** |
| Mistral | 15 | 15 | 15 | **45** |
| **总计** | 45 | 45 | 45 | **135** |

## 🏆 最佳结果

### 总体冠军：Qwen + Layerwise

- **ACC**: 0.6161 (最高)
- **PPL**: 10.80 (最低)
- **参数**:
  - taylor_seq_len: 128
  - taylor_num_samples: 512
- **结果目录**: `results/best_Qwen_20/`

### 各模型最佳配置

#### 1. Llama (亚军)
- **剪枝方法**: Blockwise
- **ACC**: 0.5980
- **PPL**: 13.17
- **参数**:
  - taylor_seq_len: 64
  - taylor_num_samples: 128
- **结果目录**: `results/best_Llama_20/`

#### 2. Mistral (季军)
- **剪枝方法**: Blockwise
- **ACC**: 0.5947
- **PPL**: 13.29
- **参数**:
  - taylor_seq_len: 64
  - taylor_num_samples: 128
- **结果目录**: `results/best_Mistral_20/`

## 📁 目录结构

```
results/
├── consolidated_Llama_20/          # Llama汇总结果（45个实验）
│   ├── all_methods_results.csv     # 所有方法的完整数据
│   ├── global_best_config.json     # 全局最佳配置
│   └── method_comparison.json      # 方法对比统计
│
├── consolidated_Qwen_20/           # Qwen汇总结果（45个实验）
│   ├── all_methods_results.csv
│   ├── global_best_config.json
│   └── method_comparison.json
│
├── consolidated_Mistral_20/        # Mistral汇总结果（45个实验）
│   ├── all_methods_results.csv
│   ├── global_best_config.json
│   └── method_comparison.json
│
├── best_Llama_20/                  # Llama最佳实验完整结果
│   ├── best_config.json
│   ├── evaluation/
│   │   └── evaluation_results.json
│   ├── analysis/
│   │   ├── gradient_diagnosis.json
│   │   ├── pruning_comparison.json
│   │   └── ...
│   └── visualization/
│       ├── gradient_analysis.png
│       ├── pruning_ratio.png
│       └── retention_ratio.png
│
├── best_Qwen_20/                   # Qwen最佳实验完整结果（总冠军）
│   └── (结构同上)
│
├── best_Mistral_20/                # Mistral最佳实验完整结果
│   └── (结构同上)
│
├── cross_model_comparison.json     # 跨模型对比
└── final_analysis_report.txt       # 完整分析报告
```

## 🔍 关键发现

### 1. 剪枝方法对比

- **Blockwise**: Llama和Mistral的最佳选择
- **Layerwise**: Qwen的最佳选择（获得全局最高ACC）
- **Taylor**: 方差较大，性能不够稳定

### 2. Taylor参数影响

| 模型 | 最佳seq_len | 最佳num_samples | 趋势 |
|------|-------------|-----------------|------|
| Llama | 256 | 512 | seq_len越大越好 |
| Qwen | 128 | 512 | 中等seq_len最佳 |
| Mistral | 64 | 128 | 较小参数即可 |

### 3. 梯度指标相关性

- **grad_mean_ratio** 与 ACC 呈中等正相关（0.697）
- **PPL** 与 ACC 呈强负相关（-0.5 ~ -0.8）
- PPL越低，ACC通常越高

### 4. 极端剪枝影响

- **最优范围**: 3-5层极端剪枝
- **负面影响**: 超过6层会显著降低性能
- 与ACC呈弱到中等负相关（-0.13 ~ -0.29）

## 📈 各任务详细表现

### Qwen Layerwise (最佳)
| 任务 | ACC |
|------|-----|
| BoolQ | 0.7618 |
| PIQA | 0.7465 |
| HellaSwag | 0.6579 |
| WinoGrande | 0.6259 |
| ARC-Easy | 0.6940 |
| ARC-Challenge | 0.4505 |
| OpenBookQA | 0.3760 |

### Llama Blockwise
| 任务 | ACC |
|------|-----|
| BoolQ | 0.7324 |
| PIQA | 0.7301 |
| HellaSwag | 0.6476 |
| WinoGrande | 0.6953 |
| ARC-Easy | 0.5981 |
| ARC-Challenge | 0.3968 |
| OpenBookQA | 0.3860 |

### Mistral Blockwise
| 任务 | ACC |
|------|-----|
| BoolQ | 0.6875 |
| PIQA | 0.7693 |
| HellaSwag | 0.6377 |
| WinoGrande | 0.6440 |
| ARC-Easy | 0.6503 |
| ARC-Challenge | 0.3959 |
| OpenBookQA | 0.3780 |

## 🛠️ 使用工具

### 1. 汇总结果
```bash
python param_search/consolidate_model_results.py --model Llama
python param_search/consolidate_model_results.py --model Qwen
python param_search/consolidate_model_results.py --model Mistral
```

### 2. 分析结果
```bash
# 分析单个模型
python param_search/analyze_consolidated_results.py --model Llama

# 分析所有模型并进行跨模型对比
python param_search/analyze_consolidated_results.py --all
```

### 3. 复制最佳结果
```bash
# 复制单个模型的最佳结果
python param_search/copy_best_results.py --model Llama

# 复制所有模型的最佳结果
python param_search/copy_best_results.py --all
```

## 📝 数据说明

### CSV文件字段

- `output_dir`: 实验输出目录
- `ppl`: Perplexity (WikiText2)
- `acc_mean`: 平均准确率（7个zero-shot任务）
- `acc_*`: 各个任务的详细准确率
- `params_count`: 模型参数数量
- `pruning_ratio`: 剪枝比率
- `grad_mean_ratio`: 梯度均值比率
- `grad_norm_ratio`: 梯度范数比率
- `extreme_pruning_layers`: 极端剪枝层数
- `pruning_method`: 剪枝方法（taylor/layerwise/blockwise）

### JSON配置文件

- `global_best_config.json`: 包含模型全局最佳配置的完整信息
- `method_comparison.json`: 每种剪枝方法的统计和最佳配置
- `cross_model_comparison.json`: 所有模型的跨模型对比

## 🎯 建议

基于实验结果，建议：

1. **Qwen模型**: 优先使用Layerwise方法，seq_len=128, samples=512
2. **Llama模型**: 优先使用Blockwise方法，seq_len=64, samples=128
3. **Mistral模型**: 优先使用Blockwise方法，seq_len=64, samples=128
4. **通用原则**:
   - 控制极端剪枝层数在3-5层
   - 监控PPL指标，保持在较低水平
   - 较大的seq_len通常带来更好的性能
