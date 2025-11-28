# 项目结构说明

## 📁 目录结构

```
llama-pruning/
├── baselines/                  # Baseline 模型和剪枝脚本
│   ├── LLM-Pruner_1937/       # LLM-Pruner 剪枝结果
│   ├── Magnitude_2000/        # Magnitude 剪枝结果
│   ├── Wanda_2000/            # Wanda 剪枝结果
│   ├── ShortGPT_remove_7/     # ShortGPT 剪枝结果
│   ├── taylor_only_2000/      # Taylor 剪枝结果
│   ├── SliceGPT_2000/         # SliceGPT 剪枝结果
│   ├── SliceGPT_PCA_2000/     # SliceGPT PCA 剪枝结果
│   ├── run_*.py               # 各剪枝方法的运行脚本
│   └── *_utils.py             # 工具函数
│
├── results/                    # 实验结果目录
│   ├── HGSP_*/                # HGSP 方法结果
│   ├── ShortGPT_*/            # ShortGPT 各种配置结果
│   ├── *_2000/                # 20% 剪枝率的结果
│   ├── *_5000/                # 50% 剪枝率的结果
│   └── *_finetuned/           # 微调后的结果
│
├── core/                       # 核心代码
│   ├── analysis/              # 模型分析工具
│   │   └── model_analysis.py  # 单模型分析脚本
│   ├── models/                # 模型定义
│   ├── pruning/               # 剪枝算法
│   ├── utils/                 # 通用工具
│   └── visualization/         # 可视化工具
│       ├── generate_pruning_charts.py    # 生成剪枝图表
│       └── generate_results_table.py     # 生成结果表格
│
├── evaluation/                 # 评估工具
│   ├── batch_model_analysis.py         # 批量模型分析
│   ├── summarize_model_structures.py   # 汇总模型结构
│   ├── run_evaluation.py               # 运行评估
│   ├── metrics/                        # 评估指标
│   └── utils/                          # 评估工具
│
├── outputs/                    # 输出文件（git 忽略）
│   ├── models_structure_summary.*      # 模型结构汇总
│   ├── baselines_compare.xlsx          # Baseline 对比
│   └── *.csv, *.xlsx                   # 各种汇总表格
│
├── docs/                       # 文档目录
├── scripts/                    # 辅助脚本
│
├── layer_pruning.py           # 层级剪枝主脚本
├── run_global_pruning.py      # 全局剪枝主脚本
├── finetune_lora.py           # LoRA 微调脚本
│
├── .gitignore                 # Git 忽略文件
├── README.md                  # 项目说明
├── USAGE.md                   # 使用指南
├── PROJECT_STRUCTURE.md       # 项目结构说明（本文件）
└── cleanup_project.sh         # 项目清理脚本
```

## 📝 关键文件说明

### 主要脚本

| 文件 | 功能 | 用法 |
|------|------|------|
| `layer_pruning.py` | 层级剪枝 | 基于层重要性的剪枝方法 |
| `run_global_pruning.py` | 全局剪枝 | 全局权重剪枝方法 |
| `finetune_lora.py` | LoRA 微调 | 剪枝后模型的微调 |

### 分析工具

| 文件 | 功能 | 输出 |
|------|------|------|
| `evaluation/batch_model_analysis.py` | 批量分析模型结构 | `analysis/model_*.json` |
| `evaluation/summarize_model_structures.py` | 汇总模型结构 | `outputs/models_structure_summary.*` |
| `core/visualization/generate_pruning_charts.py` | 生成剪枝图表 | `pruning_charts/*/*.png` |
| `core/visualization/generate_results_table.py` | 生成结果表格 | `outputs/*.xlsx` |

### 配置文件

| 文件 | 说明 |
|------|------|
| `.gitignore` | Git 忽略规则 |
| `evaluation and finetuned_cmd.md` | 评估和微调命令 |

## 🗂️ 数据组织

### 每个模型目录结构

```
<模型名称>/
├── analysis/                           # 模型分析结果
│   ├── model_structure.json           # 模型结构详情
│   ├── model_comparison.json          # 与原模型对比
│   ├── pruning_comparison.json        # 剪枝对比（用于可视化）
│   ├── structure_summary.txt          # 结构摘要
│   ├── original_model_analysis.json   # 原始模型分析
│   ├── pruned_model_analysis.json     # 剪枝模型分析
│   └── pruning_summary_by_layer.txt   # 按层的剪枝摘要
│
├── evaluation/                         # 评估结果
│   └── evaluation_results.json        # 性能评估结果
│
├── logs/                               # 训练日志
│   └── <timestamp>/
│       ├── description.txt            # 训练配置描述
│       └── train.sh                   # 训练命令
│
├── pruned_model.bin                    # 剪枝后的模型文件（忽略）
└── description.txt                     # 模型描述
```

## 🔧 工具脚本

### cleanup_project.sh

清理项目临时文件和整理输出：

```bash
./cleanup_project.sh
```

功能：
- 创建 `outputs/` 目录
- 移动输出文件到 `outputs/`
- 删除临时测试文件
- 清理 Python 缓存
- 删除旧的汇总表文件

## 📊 输出文件

所有分析和可视化的输出文件应该放在 `outputs/` 目录中，该目录已在 `.gitignore` 中忽略。

### 推荐的输出组织

```
outputs/
├── structure_analysis/              # 结构分析结果
│   ├── models_structure_summary.json
│   └── models_structure_summary.txt
│
├── performance_comparison/          # 性能对比
│   ├── baselines_compare.xlsx
│   └── results_table.xlsx
│
└── charts/                          # 图表（符号链接到 pruning_charts）
    └── ...
```

## 🚫 被忽略的文件类型

根据 `.gitignore`，以下文件类型不会被 Git 跟踪：

- **模型文件**: `*.pt`, `*.pth`, `*.bin`, `*.safetensors`
- **输出文件**: `outputs/`, `pruning_charts/`, `baselines_analysis/`
- **表格文件**: `*.xlsx`, `*.csv`, `*.html`
- **Python 缓存**: `__pycache__/`, `*.pyc`
- **临时文件**: `test_*.py`, `*_test.py`, `*.tmp`, `*.bak`
- **系统文件**: `.DS_Store`, `Thumbs.db`
- **IDE 文件**: `.vscode/`, `.idea/`

## 📚 文档

- **README.md**: 项目整体说明
- **USAGE.md**: 使用指南
- **evaluation/BATCH_ANALYSIS.md**: 批量分析说明
- **evaluation/MODEL_ANALYSIS_GUIDE.md**: 模型分析完整指南
- **evaluation/SLICEGPT_CONVERSION.md**: SliceGPT 转换指南

## 🔄 工作流程

### 1. 训练剪枝模型

```bash
python run_global_pruning.py --config <config>
```

### 2. 批量分析模型结构

```bash
python evaluation/batch_model_analysis.py \
    --models_dir baselines/ \
    --base_model <base_model_path>
```

### 3. 生成可视化图表

```bash
python core/visualization/generate_pruning_charts.py \
    --result_dir baselines \
    --output_dir pruning_charts
```

### 4. 汇总分析结果

```bash
python evaluation/summarize_model_structures.py \
    --dirs baselines results \
    --output outputs/models_structure_summary
```

### 5. 清理项目

```bash
./cleanup_project.sh
```

## 💡 最佳实践

1. **保持根目录整洁**: 所有输出文件放在 `outputs/` 目录
2. **定期清理**: 使用 `cleanup_project.sh` 清理临时文件
3. **版本控制**: 只提交代码和文档，不提交输出文件
4. **模型文件**: 大模型文件使用软链接或外部存储
5. **文档更新**: 每次重大变更后更新相关文档

## 🗑️ 清理建议

定期执行以下清理操作：

```bash
# 1. 清理 Python 缓存
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# 2. 清理临时文件
find . -type f -name "*~" -delete
find . -type f -name "*.swp" -delete

# 3. 清理旧的输出文件
rm -rf pruning_charts/  # 可重新生成

# 4. 或使用清理脚本
./cleanup_project.sh
```
