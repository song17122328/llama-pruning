# Baseline 方法复现

这个目录包含了各种剪枝baseline方法的复现脚本。

## 可用的 Baseline 方法

### 1. Magnitude（权重绝对值）

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

## 对比实验

推荐进行以下对比实验：

```bash
# 1. Taylor 方法（我们的方法）
python run_global_pruning.py \
    --base_model /path/to/llama \
    --output_name HGSP_Taylor_20 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --run_evaluation ppl,zeroshot,speed,memory

# 2. Magnitude baseline
python baselines/run_magnitude.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --output_name Magnitude_20

# 3. Wanda baseline（如果想要对比）
python run_global_pruning.py \
    --base_model /path/to/llama \
    --output_name Wanda_20 \
    --pruning_ratio 0.2 \
    --importance_method wanda \
    --run_evaluation ppl,zeroshot,speed,memory
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

| 方法 | PPL (WikiText2) | Zero-shot Acc | 推理速度 | 显存占用 |
|------|----------------|---------------|---------|---------|
| 原始模型 | - | - | - | - |
| Magnitude | - | - | - | - |
| Wanda | - | - | - | - |
| **H-GSP (Ours)** | - | - | - | - |

每个方法都应该测试：
- 剪枝后立即评估
- 微调后评估

这样可以展示我们方法的优势。
