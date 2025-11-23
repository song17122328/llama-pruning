# Visualization and Analysis Tools

本目录包含用于分析和可视化 LLaMA 模型剪枝实验结果的工具集。这些工具可以帮助您快速生成用于研究论文的表格和图表。

## 📋 目录

- [工具概览](#工具概览)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细使用说明](#详细使用说明)
  - [结果表格生成器](#1-结果表格生成器-generate_results_tablepy)
  - [剪枝比例可视化](#2-剪枝比例可视化-generate_pruning_chartspy)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 工具概览

| 工具 | 功能 | 输出格式 | 适用场景 |
|------|------|---------|---------|
| `generate_results_table.py` | 汇总所有模型的评估指标和剪枝统计 | CSV/Markdown/HTML/Excel | 论文表格、性能对比 |
| `generate_pruning_charts.py` | 生成层级剪枝和保留比例可视化图表 | PNG (300 DPI) | 论文插图、剪枝模式分析 |

---

## 环境要求

```bash
# 必需依赖
pip install pandas numpy matplotlib tabulate openpyxl
```

**Python 版本**: >= 3.7

---

## 快速开始

### 从项目根目录运行

```bash
# 1. 生成结果汇总表格（CSV 格式）
python core/visualization/generate_results_table.py --result_dir results --output results_summary.csv

# 2. 生成所有模型的剪枝比例图表
python core/visualization/generate_pruning_charts.py --result_dir results --output_dir pruning_charts
```

### 从 core/visualization 目录运行

```bash
cd core/visualization

# 1. 生成 Markdown 表格（适合直接插入论文）
python generate_results_table.py --result_dir ../../results --output summary.md --format markdown

# 2. 为特定模型生成图表
python generate_pruning_charts.py --result_dir ../../results --models HGSP_2000,layerwise_only_5000 --output_dir charts
```

---

## 详细使用说明

### 1. 结果表格生成器 (`generate_results_table.py`)

#### 功能描述

自动遍历实验结果目录，提取并汇总所有剪枝模型的评估指标，生成对比表格。

#### 提取的指标（24列）

**基本信息**:
- 模型名称
- 模型大小 (GB)
- 参数量 (Billion)

**困惑度 (Perplexity)**:
- WikiText-2 PPL
- PTB PPL

**零样本任务准确率**:
- BoolQ, PIQA, HellaSwag, WinoGrande
- ARC-easy, ARC-challenge, OBQA
- 平均准确率

**性能指标**:
- Batch Size 1/4 吞吐量 (tokens/s)
- Batch Size 1/4 延迟 (ms)
- 显存占用 (GB)

**剪枝统计**:
- 32层保留率的标准差和方差
- 保留率 <5%/10%/15%/20% 的层列表

#### 命令行参数

```bash
python generate_results_table.py [OPTIONS]
```

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--result_dir` | 结果目录路径 | `results` | `--result_dir results` |
| `--output` | 输出文件路径 | `results_summary.csv` | `--output table.csv` |
| `--format` | 输出格式 | `csv` | `--format markdown` |
| `--verbose` | 显示详细信息 | False | `--verbose` |

**支持的输出格式**:
- `csv`: 逗号分隔值文件（Excel 可打开）
- `markdown`: Markdown 表格（GitHub/论文友好）
- `html`: HTML 表格（网页展示）
- `excel`: Excel 工作簿（.xlsx 格式）

#### 使用示例

```bash
# 示例 1: 生成 CSV 表格
python generate_results_table.py \
    --result_dir results \
    --output paper_table_1.csv \
    --format csv

# 示例 2: 生成 Markdown 表格（适合插入论文）
python generate_results_table.py \
    --result_dir results \
    --output paper_table_1.md \
    --format markdown

# 示例 3: 生成 Excel 表格（便于进一步编辑）
python generate_results_table.py \
    --result_dir results \
    --output analysis.xlsx \
    --format excel \
    --verbose

# 示例 4: 生成 HTML 网页
python generate_results_table.py \
    --result_dir results \
    --output results.html \
    --format html
```

#### 输入数据结构

脚本期望以下目录结构：

```
results/
├── model1/
│   ├── evaluation/
│   │   └── evaluation_results.json
│   └── analysis/
│       └── pruning_comparison.json
├── model2/
│   ├── evaluation/
│   │   └── evaluation_results.json
│   └── analysis/
│       └── pruning_comparison.json
...
```

#### 输出示例

**Markdown 格式输出片段**:

```markdown
| 模型名称 | 模型大小 (GB) | 参数量 (B) | WikiText-2 PPL | PTB PPL | ... |
|---------|--------------|-----------|----------------|---------|-----|
| Llama-3-8B-Instruct | 15.01 | 8.03 | 10.23 | 18.45 | ... |
| HGSP_5000 | 3.56 | 1.91 | 12.87 | 21.34 | ... |
| layerwise_only_2000 | 3.51 | 1.88 | 13.21 | 22.11 | ... |
```

---

### 2. 剪枝比例可视化 (`generate_pruning_charts.py`)

#### 功能描述

为每个剪枝模型生成两张高质量的柱状图：
1. **剪枝比例图** (Pruning Ratio): 显示每层被剪枝的参数百分比
2. **保留比例图** (Retention Ratio): 显示每层保留的参数百分比

#### 图表特性

- **分辨率**: 300 DPI（适合论文发表）
- **尺寸**: 14×6 英寸
- **自动中文字体**: 自动检测并使用系统可用的中文字体（无字体警告）
- **颜色编码**:
  - 剪枝比例图: 红色(>80%) > 橙色(>50%) > 蓝色(≤50%)
  - 保留比例图: 绿色(>80%) > 黄色(>50%) > 红色(≤50%)
- **智能标注**: 柱子比例 ≥5% 时显示精确百分比（避免低值柱上文字重叠）
- **网格线系统**:
  - 浅色虚线：20%, 40%, 60%, 80%, 100% 标记
  - 深色虚线：50% 参考线
  - 醒目红线：80% 阈值线
  - **醒目彩线**：模型平均剪枝/保留比例（自动计算，保留1位小数）

#### 命令行参数

```bash
python generate_pruning_charts.py [OPTIONS]
```

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--result_dir` | 结果目录路径 | `results` | `--result_dir results` |
| `--output_dir` | 图表输出目录 | `pruning_charts` | `--output_dir charts` |
| `--models` | 指定模型（逗号分隔） | 所有模型 | `--models HGSP_2000,HGSP_5000` |
| `--dpi` | 图像分辨率 | `300` | `--dpi 600` |
| `--format` | 图像格式 | `png` | `--format pdf` |
| `--verbose` | 显示详细信息 | False | `--verbose` |

#### 使用示例

```bash
# 示例 1: 为所有模型生成图表
python generate_pruning_charts.py \
    --result_dir results \
    --output_dir pruning_charts

# 示例 2: 只为特定模型生成图表
python generate_pruning_charts.py \
    --result_dir results \
    --models "HGSP_2000,HGSP_5000,layerwise_only_2000" \
    --output_dir paper_figures

# 示例 3: 生成高分辨率 PDF 格式（适合论文）
python generate_pruning_charts.py \
    --result_dir results \
    --output_dir paper_figures_pdf \
    --format pdf \
    --dpi 600

# 示例 4: 详细模式（显示处理进度）
python generate_pruning_charts.py \
    --result_dir results \
    --output_dir charts \
    --verbose
```

#### 输入数据结构

脚本从以下文件读取剪枝数据：

```
results/
├── model1/
│   └── analysis/
│       ├── pruning_comparison.json  # 优先使用
│       └── pruning_comparison.txt   # 备选
└── model2/
    └── analysis/
        └── pruning_comparison.json
```

#### 输出结构

```
pruning_charts/
├── HGSP_2000/
│   ├── HGSP_2000_pruning_ratio.png
│   └── HGSP_2000_retention_ratio.png
├── HGSP_5000/
│   ├── HGSP_5000_pruning_ratio.png
│   └── HGSP_5000_retention_ratio.png
...
```

#### 图表解读

**剪枝比例图** (Pruning Ratio):
- 横轴: 层索引 (0-31)
- 纵轴: 剪枝百分比 (0-100%)
- 柱子越高 = 该层剪枝越激进
- 红色区域表示重度剪枝层 (>80%)

**保留比例图** (Retention Ratio):
- 横轴: 层索引 (0-31)
- 纵轴: 保留百分比 (0-100%)
- 柱子越高 = 该层参数保留越多
- 绿色区域表示参数保留良好 (>80%)

---

## 输出说明

### 表格输出

生成的表格会按照**模型大小降序排列**（最大的模型在最上面），便于对比不同剪枝程度的效果。

**典型用途**:
- 📊 论文中的性能对比表格
- 📈 实验结果汇总报告
- 🔍 不同剪枝策略的效果分析

### 图表输出

每个模型生成 2 张图表，共计 `N × 2` 张图（N 为模型数量）。

**典型用途**:
- 📊 论文插图：展示剪枝策略的层级分布
- 🔬 分析报告：识别哪些层被重点剪枝
- 🎓 学术演讲：可视化剪枝模式

---

## 常见问题

### Q1: 找不到 evaluation_results.json 文件

**问题**: 脚本提示某些模型缺少评估结果文件

**解决方法**:
```bash
# 检查目录结构
ls -R results/your_model/

# 确保存在以下文件
results/your_model/evaluation/evaluation_results.json
```

### Q2: 图表中文显示为方块

**问题**: 生成的图表中文字显示为 `□□□`

**解决方法**:
✅ **已自动解决！** 脚本现在会自动检测系统中可用的中文字体，优先使用：
- Linux: WenQuanYi Zen Hei, WenQuanYi Micro Hei, Noto Sans CJK
- Windows: SimHei, Microsoft YaHei, SimSun
- Mac: STHeiti, STSong

如仍有问题，可手动安装中文字体：
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-zenhei fonts-wqy-microhei

# CentOS/RHEL
sudo yum install wqy-zenhei-fonts wqy-microhei-fonts
```

### Q3: 如何修改表格中的指标？

**解决方法**: 编辑 `generate_results_table.py`，修改 `extract_metrics()` 函数中的列定义。

### Q4: 如何调整图表样式？

**解决方法**: 编辑 `generate_pruning_charts.py`，在 `plot_pruning_chart()` 函数中修改：
- `figsize`: 图表尺寸
- `colors`: 颜色方案
- `fontsize`: 字体大小
- 参考线位置和样式

### Q5: 内存不足错误

**问题**: 处理大量模型时内存溢出

**解决方法**:
```bash
# 分批处理模型
python generate_pruning_charts.py \
    --result_dir results \
    --models "model1,model2,model3"

# 然后处理下一批
python generate_pruning_charts.py \
    --result_dir results \
    --models "model4,model5,model6"
```

### Q6: 如何在论文中引用这些图表？

**LaTeX 示例**:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{pruning_charts/HGSP_2000/HGSP_2000_retention_ratio.png}
    \caption{Layer-wise retention ratio of HGSP-2000 model}
    \label{fig:hgsp_retention}
\end{figure}
```

**Markdown 示例**:
```markdown
![Layer-wise Pruning Ratio](pruning_charts/HGSP_2000/HGSP_2000_pruning_ratio.png)
*Figure 1: Layer-wise pruning distribution of HGSP-2000 model*
```

---

## 进阶技巧

### 批量生成不同格式的输出

```bash
# 同时生成多种格式
for format in csv markdown html excel; do
    python generate_results_table.py \
        --result_dir results \
        --output "summary_table.$format" \
        --format $format
done
```

### 自动化工作流

```bash
#!/bin/bash
# generate_paper_materials.sh

echo "生成结果表格..."
python core/visualization/generate_results_table.py \
    --result_dir results \
    --output paper_tables/table1.md \
    --format markdown

echo "生成剪枝图表..."
python core/visualization/generate_pruning_charts.py \
    --result_dir results \
    --output_dir paper_figures \
    --dpi 600 \
    --format pdf

echo "完成！论文材料已生成到 paper_tables/ 和 paper_figures/"
```

---

## 贡献与反馈

如有问题或改进建议，请提交 Issue 或 Pull Request。

**维护者**: LLaMA Pruning Research Team
**最后更新**: 2025-11-23
