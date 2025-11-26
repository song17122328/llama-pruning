# Baseline 方法复现

这个目录包含了各种剪枝baseline方法的复现脚本。

## 可用的 Baseline 方法

### 1. Magnitude（权重绝对值）

**论文**: 经典的权重剪枝方法
**类型**: 非结构化/结构化剪枝

最简单的剪枝方法，直接使用权重的绝对值作为重要性指标。

**特点：**
- ✓ 不需要计算梯度
- ✓ 不需要收集激活值
- ✓ 速度最快
- ✗ 不考虑数据分布信息

**使用方法：**
```bash
# 基础使用
python baselines/run_magnitude.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2

# 带评估和微调
python baselines/run_magnitude.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --run_evaluation \
    --finetune

# 自定义输出名称
python baselines/run_magnitude.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --output_name My_Magnitude_Baseline
```

**参数说明：**
- `--base_model`: 基础模型路径（必需）
- `--pruning_ratio`: 剪枝率，例如 0.2 表示剪枝20%（必需）
- `--output_name`: 输出目录名称（默认: Magnitude_{pruning_ratio}）
- `--run_evaluation`: 运行评估（默认: True）
- `--eval_metrics`: 评估指标（默认: ppl,zeroshot,speed,memory）
- `--finetune`: 剪枝后进行 LoRA 微调
- `--temperature`: H-GSP 温度参数（默认: 0.0，即纯 Magnitude）
- `--epsilon`: H-GSP 坍缩阈值（默认: 0.15）

---

### 2. ShortGPT（Block Influence 层移除）

**论文**: ShortGPT: Layers in Large Language Models are More Redundant Than You Expect
**类型**: 纯层级剪枝（深度剪枝）

基于 Block Influence (BI) 的层重要性计算方法，移除最不重要的层。

**核心思想：**
- 计算每层的输入和输出隐藏状态的相似度
- 相似度高 → 该层变换作用小 → 重要性低 → 可以被移除
- BI = 1 - cosine_similarity(input_hidden, output_hidden)

**特点：**
- ✓ 只需要前向传播
- ✓ 不需要梯度计算
- ✓ 专注于层级剪枝
- ✓ 剪枝后模型层数减少
- ✗ 无法进行宽度剪枝

**使用方法：**
```bash
# 基础使用（移除 4 层）
python baselines/run_shortgpt.py \
    --base_model /path/to/llama \
    --n_remove_layers 4

# 移除更多层（激进剪枝）
python baselines/run_shortgpt.py \
    --base_model /path/to/llama \
    --n_remove_layers 8 \
    --output_name ShortGPT_remove_8

# 带评估和微调
python baselines/run_shortgpt.py \
    --base_model /path/to/llama \
    --n_remove_layers 4 \
    --run_evaluation \
    --finetune

# 自定义 BI 计算参数
python baselines/run_shortgpt.py \
    --base_model /path/to/llama \
    --n_remove_layers 4 \
    --num_samples 100 \
    --seq_len 512 \
    --stride 256
```

**参数说明：**
- `--base_model`: 基础模型路径（必需）
- `--n_remove_layers`: 要移除的层数（必需）
- `--output_name`: 输出目录名称（默认: ShortGPT_remove_{n_remove_layers}）
- `--dataset`: 用于 BI 计算的数据集（默认: wikitext2）
- `--num_samples`: BI 计算样本数（默认: 50）
- `--seq_len`: 序列长度（默认: 512）
- `--stride`: 滑动窗口步长（默认: 256）
- `--run_evaluation`: 运行评估（默认: True）
- `--eval_metrics`: 评估指标（默认: ppl,zeroshot,speed,memory）
- `--finetune`: 剪枝后进行 LoRA 微调

**推荐配置：**
- 对于 32 层模型：移除 4-8 层
- 对于 24 层模型：移除 3-6 层
- 建议先从较少的层数开始测试

## 对比实验

推荐进行以下对比实验：

```bash
# 1. H-GSP (我们的方法 - 混合剪枝)
python run_global_pruning.py \
    --base_model /path/to/llama \
    --output_name HGSP_Taylor_20 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --run_evaluation ppl,zeroshot,speed,memory

# 2. Magnitude baseline (宽度剪枝)
python baselines/run_magnitude.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --output_name Magnitude_20

# 3. ShortGPT baseline (纯深度剪枝)
python baselines/run_shortgpt.py \
    --base_model /path/to/llama \
    --n_remove_layers 4 \
    --output_name ShortGPT_remove_4

# 4. Wanda baseline (可选 - 宽度剪枝)
python run_global_pruning.py \
    --base_model /path/to/llama \
    --output_name Wanda_20 \
    --pruning_ratio 0.2 \
    --importance_method wanda \
    --run_evaluation ppl,zeroshot,speed,memory
```

### 实验设计建议

**对于相同的参数剪枝率（例如 20%）：**

1. **Magnitude_20**: 使用 Magnitude 方法剪枝 20% 参数
2. **ShortGPT_remove_X**: 移除 X 层，使参数剪枝率接近 20%
   - 对于 LLaMA-3-8B (32层)：移除约 6-7 层 ≈ 20% 参数
   - 对于 LLaMA-2-7B (32层)：移除约 6-7 层 ≈ 20% 参数
3. **HGSP_Taylor_20**: 我们的方法，20% 参数剪枝（混合深度+宽度）

**完整对比流程：**
```bash
MODEL=/path/to/llama

# 我们的方法
python run_global_pruning.py \
    --base_model $MODEL \
    --output_name HGSP_20 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --finetune \
    --run_evaluation ppl,zeroshot,speed,memory

# Magnitude baseline
python baselines/run_magnitude.py \
    --base_model $MODEL \
    --pruning_ratio 0.2 \
    --finetune \
    --run_evaluation

# ShortGPT baseline (根据层数调整)
python baselines/run_shortgpt.py \
    --base_model $MODEL \
    --n_remove_layers 6 \
    --finetune \
    --run_evaluation
```

## 结果目录结构

所有结果会保存在 `results/` 目录下：

```
results/
├── HGSP_Taylor_20/          # 我们的方法
│   ├── pruned_model.bin
│   ├── analysis/
│   ├── evaluation/
│   └── logs/
├── Magnitude_20/            # Magnitude baseline
│   ├── pruned_model.bin
│   ├── analysis/
│   ├── evaluation/
│   └── logs/
└── ...
```

## 评估结果对比

评估完成后，可以使用评估脚本进行对比：

```bash
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/HGSP_Taylor_20/evaluation/evaluation_results.json,results/Magnitude_20/evaluation/evaluation_results.json \
    --output comparison_table.md
```

## 添加新的 Baseline

要添加新的 baseline 方法：

1. 在 `core/methods/global_pruning.py` 中添加重要性计算函数
2. 在 `build_global_group_table` 中添加对新方法的支持
3. 在 `run_global_pruning.py` 的 `--importance_method` 参数中添加新方法
4. 在 `baselines/` 目录下创建运行脚本

参考 Magnitude 的实现：
- `compute_attention_group_importance_magnitude()`
- `compute_mlp_group_importance_magnitude()`
- `baselines/run_magnitude.py`

## 注意事项

1. **Magnitude 方法的局限性**：
   - Magnitude 只考虑权重大小，不考虑数据分布
   - 对于某些模型可能效果较差
   - 建议作为最简单的 baseline 进行对比

2. **建议的剪枝率**：
   - 20%：适度剪枝，性能下降小
   - 50%：激进剪枝，性能下降较大
   - 建议先从 20% 开始测试

3. **微调的重要性**：
   - 对于 Magnitude 这样的简单方法，微调特别重要
   - 建议加上 `--finetune` 参数进行微调恢复

## 论文中的对比实验

建议在论文中包含以下对比：

| 方法 | 类型 | PPL (WikiText2) | Zero-shot Acc | 推理速度 | 显存占用 | 备注 |
|------|------|----------------|---------------|---------|---------|------|
| 原始模型 | - | - | - | - | - | Baseline |
| Magnitude | 宽度剪枝 | - | - | - | - | 简单 baseline |
| ShortGPT | 深度剪枝 | - | - | - | - | 纯层移除 |
| Wanda | 宽度剪枝 | - | - | - | - | 激活感知 |
| **H-GSP (Ours)** | 混合剪枝 | - | - | - | - | 深度+宽度 |

### 实验配置建议

**每个方法都应该测试：**
1. 剪枝后立即评估（展示剪枝方法的直接效果）
2. 微调后评估（展示恢复能力）

**对比维度：**
- **性能**：PPL、Zero-shot 准确率
- **效率**：推理速度、显存占用
- **剪枝率**：参数减少百分比、层数变化
- **鲁棒性**：不同剪枝率下的表现

### 方法对比分析

| 方法 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **Magnitude** | 简单快速，无需数据 | 不考虑数据分布，效果较差 | 快速原型、最简 baseline |
| **ShortGPT** | 专注层级，实现简单 | 无法宽度剪枝，灵活性差 | 纯层移除场景 |
| **Wanda** | 考虑激活值，较准确 | 需要收集激活，速度慢 | 宽度剪枝 |
| **H-GSP (Ours)** | 混合剪枝，自动深度+宽度 | 需要梯度计算 | 综合性能最优 |

这样可以全面展示我们方法的优势。
