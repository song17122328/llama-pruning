# 评估结果汇总脚本使用说明

## 📋 功能说明

`generate_results_table.py` 脚本用于自动汇总和可视化多个剪枝模型的评估结果和剪枝统计。

## 📁 目录结构要求

脚本会遍历 `results` 目录下的所有子文件夹，每个子文件夹代表一个剪枝模型：

```
results/
├── model_1/
│   ├── evaluation/
│   │   └── evaluation_results.json      # 评估指标
│   └── analysis/
│       ├── pruning_comparison.json      # 剪枝对比（优先）
│       └── pruning_comparison.txt       # 剪枝对比（备用）
├── model_2/
│   ├── evaluation/
│   │   └── evaluation_results.json
│   └── analysis/
│       └── pruning_comparison.json
└── ...
```

## 🚀 使用方法

### 基本用法

```bash
# 生成 CSV 格式表格
python generate_results_table.py \
    --result_dir results \
    --output summary_table.csv

# 生成 Markdown 格式表格
python generate_results_table.py \
    --result_dir results \
    --output summary_table.md

# 生成 HTML 格式表格（带样式）
python generate_results_table.py \
    --result_dir results \
    --output summary_table.html

# 生成 Excel 格式表格
python generate_results_table.py \
    --result_dir results \
    --output summary_table.xlsx
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--result_dir` | 结果目录路径 | `results` |
| `--output` | 输出文件路径（必需） | - |
| `--format` | 输出格式（auto/csv/markdown/html/excel/latex） | `auto` |
| `--decimal_places` | 数值保留的小数位数 | `2` |
| `--no_summary` | 不打印汇总统计信息 | `False` |

## 📊 提取的指标

脚本会自动提取以下所有指标（共 24 列）：

### 1. 模型信息 (2列)
- **模型大小 (GB)**: 模型文件大小
- **参数量 (B)**: 总参数量（十亿）

### 2. 困惑度 (2列)
- **PPL (WikiText-2)**: WikiText-2 数据集上的困惑度
- **PPL (PTB)**: Penn TreeBank 数据集上的困惑度

### 3. Zero-shot 准确率 (8列)
- **ZS-BoolQ (%)**: BoolQ 任务准确率
- **ZS-PIQA (%)**: PIQA 任务准确率
- **ZS-HellaSwag (%)**: HellaSwag 任务准确率
- **ZS-WinoGrande (%)**: WinoGrande 任务准确率
- **ZS-ARC-e (%)**: ARC-Easy 任务准确率
- **ZS-ARC-c (%)**: ARC-Challenge 任务准确率
- **ZS-OBQA (%)**: OpenBookQA 任务准确率
- **ZS-平均 (%)**: 7个任务的平均准确率

### 4. 效率指标 (5列)
- **吞吐量-BS1 (tokens/s)**: Batch Size=1 时的吞吐量
- **延迟-BS1 (ms/token)**: Batch Size=1 时的延迟
- **吞吐量-BS4 (tokens/s)**: Batch Size=4 时的吞吐量
- **延迟-BS4 (ms/token)**: Batch Size=4 时的延迟
- **显存占用 (MB)**: GPU 显存占用

### 5. 剪枝统计分析 (6列) ✨ 新增
- **剪枝标准差**: 32层剪枝保留比例的标准差（衡量剪枝的不均衡程度）
- **剪枝方差**: 32层剪枝保留比例的方差
- **保留<5%的层**: 保留比例小于5%的层索引列表
- **保留<10%的层**: 保留比例小于10%的层索引列表
- **保留<15%的层**: 保留比例小于15%的层索引列表
- **保留<20%的层**: 保留比例小于20%的层索引列表

## 📝 输出示例

### Markdown 格式

| 模型名称 | 模型大小 (GB) | PPL (WikiText-2) | ZS-平均 (%) | 剪枝标准差 | 保留<5%的层 |
|---------|--------------|------------------|-------------|-----------|-----------|
| HGSP_2000 | 11.966 | 39.87 | 51.41 | 0.3644 | [11, 12, 25] |
| HGSP_5000 | 7.479 | 315.21 | 36.96 | 0.3379 | [9, 10, 11, 12, 25, 26] |
| taylor_only_2000 | 11.966 | 37.32 | 62.05 | 0.1524 | [] |

### CSV 格式

```csv
模型名称,模型大小 (GB),PPL (WikiText-2),ZS-平均 (%),剪枝标准差,保留<5%的层
HGSP_2000,11.966,39.87,51.41,0.3644,"[11, 12, 25]"
HGSP_5000,7.479,315.21,36.96,0.3379,"[9, 10, 11, 12, 25, 26]"
taylor_only_2000,11.966,37.32,62.05,0.1524,[]
```

### HTML 格式

生成的 HTML 文件包含美化的样式，可以直接在浏览器中查看。

## 🔄 完整工作流程

```bash
# 1. 运行剪枝实验（生成剪枝模型）
python run_global_pruning.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.25 \
    --output_model results/my_model/pruned_model.bin

# 2. 运行模型分析（生成 pruning_comparison.json）
python core/analysis/model_analysis.py \
    --model_path /path/to/original_model \
    --compare_with results/my_model/pruned_model.bin

# 3. 运行评估（生成 evaluation_results.json）
python evaluation/run_evaluation.py \
    --model_path results/my_model/pruned_model.bin \
    --metrics all \
    --output results/my_model/evaluation/evaluation_results.json

# 4. 汇总所有模型结果
python generate_results_table.py \
    --result_dir results \
    --output analysis/summary_table.md

# 5. 查看结果
cat analysis/summary_table.md
```

## 📌 剪枝统计说明

### 标准差和方差

- **作用**: 衡量各层剪枝的不均衡程度
- **计算方式**: 基于每层的保留比例（1 - 剪枝率）
- **解读**:
  - 标准差越大，说明不同层之间的剪枝差异越大
  - 标准差越小，说明剪枝较为均匀

### 保留比例阈值

- **保留比例** = 剪枝后参数量 / 原始参数量
- **示例**:
  - 某层原始参数 218M，剪枝后剩余 10M，保留比例 = 10/218 ≈ 4.6%
  - 该层会出现在"保留<5%的层"列表中

### 层索引列表格式

- 以 Python 列表形式显示：`[9, 11, 12]`
- 空列表表示没有层满足条件：`[]`
- CSV/Excel 中以字符串形式存储，便于后续处理

## 📈 数据分析技巧

### 使用 pandas 进行二次分析

```python
import pandas as pd

# 加载生成的 CSV
df = pd.read_csv('summary_table.csv')

# 按 PPL 升序排序
df_sorted = df.sort_values('PPL (WikiText-2)')

# 筛选剪枝标准差小于 0.2 的模型
df_uniform = df[df['剪枝标准差'] < 0.2]

# 查看 Zero-shot 平均准确率最高的模型
best_model = df.loc[df['ZS-平均 (%)'].idxmax()]
print(f"最佳模型: {best_model['模型名称']}")
```

### 可视化剪枝统计

```python
import matplotlib.pyplot as plt

# 绘制剪枝标准差 vs PPL
plt.scatter(df['剪枝标准差'], df['PPL (WikiText-2)'])
plt.xlabel('剪枝标准差')
plt.ylabel('PPL (WikiText-2)')
plt.title('剪枝不均衡性 vs 模型困惑度')
plt.show()
```

## 🐛 故障排除

**问题**: 找不到 `evaluation_results.json`

**解决方案**:
- 确保已运行 `evaluation/run_evaluation.py` 生成结果
- 检查文件路径是否为 `<model_dir>/evaluation/evaluation_results.json`

**问题**: 找不到 `pruning_comparison.json`

**解决方案**:
- 确保已运行 `core/analysis/model_analysis.py` 生成剪枝对比
- 脚本会自动尝试 JSON 和 TXT 两种格式

**问题**: 某些列显示 "N/A"

**解决方案**:
- 检查评估时是否包含了相应的指标
- 运行评估时使用 `--metrics all` 包含所有指标

**问题**: Excel 导出失败

**解决方案**:
```bash
pip install openpyxl
```

**问题**: Markdown 导出失败

**解决方案**:
```bash
pip install tabulate
```

## 💡 最佳实践

1. **优先使用 JSON 格式**: `pruning_comparison.json` 解析更可靠
2. **定期备份结果**: 将生成的表格保存到版本控制系统
3. **自动化流程**: 将评估和汇总脚本集成到实验流程中
4. **多格式导出**: 同时生成 CSV（数据分析）和 Markdown（文档）版本

## 🔍 依赖库

```bash
pip install pandas numpy tabulate openpyxl
```

## 📚 相关文档

- `evaluation/run_evaluation.py` - 评估脚本
- `core/analysis/model_analysis.py` - 模型分析脚本
- `README.md` - 项目主文档
