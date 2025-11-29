# H-GSP 使用文档

## 快速开始

H-GSP (Hybrid Global Structural Pruning) 是一个针对 LLaMA 模型的智能剪枝工具，结合了 Taylor 重要性、层级分析和块级分析，自动实现深度+宽度混合剪枝。

### 最简示例

```bash
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name llama3_pruned_50 \
  --pruning_ratio 0.5
```

## 核心参数

### 必需参数

- **`--base_model`**: 基础模型路径
- **`--output_name`**: 输出目录名称（所有结果保存在 `results/{output_name}/`）

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pruning_ratio` | 0.5 | 剪枝率（0.5 = 剪掉 50% 参数）|
| `--temperature` | 1.0 | H-GSP 温度参数 T，推荐范围 0.5-2.0 |
| `--dataset` | wikitext2 | 用于重要性计算的数据集 |
| `--importance_method` | taylor | 重要性计算方法: taylor / wanda / taylor_2nd |

**温度参数 T 说明：**
- `T=0`: 纯 Taylor 方法（不使用层级先验）
- `T=1`: **推荐**，平衡模式
- `T>1`: 激进模式，强化首尾层保护

### H-GSP 高级参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tau` | None | 门控阈值 τ（None=自动计算 25 分位数）|
| `--epsilon` | 0.15 | 坍缩阈值 ε，层保留率低于此值时自动整层移除 |

## 评估参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--run_evaluation` | 评估类型（逗号分隔）| `ppl,zeroshot` 或 `all` |
| `--eval_ppl_datasets` | PPL 数据集 | `wikitext2,ptb,c4` |
| `--eval_zeroshot_tasks` | Zero-shot 任务 | 默认包含 4 个任务 |

**评估类型：**
- `ppl`: 困惑度评估
- `zeroshot`: Zero-shot 任务准确率
- `efficiency`: 推理速度和显存占用
- `all`: 运行所有评估

## 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--finetune` | False | 是否在剪枝后微调 |
| `--finetune_method` | lora | 微调方法: lora / full |
| `--finetune_samples` | 500 | 微调样本数 |
| `--lora_r` | 8 | LoRA rank |

## 使用场景

### 1. 基础剪枝（50%）

```bash
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_50percent \
  --pruning_ratio 0.5
```

### 2. 剪枝 + PPL 评估

```bash
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_50_with_eval \
  --pruning_ratio 0.5 \
  --run_evaluation ppl
```

### 3. 剪枝 + 完整评估

```bash
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_50_full_eval \
  --pruning_ratio 0.5 \
  --run_evaluation all
```

### 4. 剪枝 + LoRA 微调 + 评估

```bash
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_50_finetuned \
  --pruning_ratio 0.5 \
  --finetune \
  --finetune_method lora \
  --finetune_samples 1000 \
  --run_evaluation ppl,zeroshot
```

### 5. 激进剪枝（70%）+ 温度调节

```bash
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_70_aggressive \
  --pruning_ratio 0.7 \
  --temperature 1.5 \
  --run_evaluation all
```

### 6. 使用不同重要性方法

```bash
# 二阶 Taylor（更精确但更慢）
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_taylor2nd \
  --pruning_ratio 0.5 \
  --importance_method taylor_2nd

# Wanda 方法（基于激活）
python run_global_pruning.py \
  --base_model /path/to/llama-3-8b \
  --output_name exp_wanda \
  --pruning_ratio 0.5 \
  --importance_method wanda
```

## 输出目录结构

运行后会在 `results/{output_name}/` 下生成：

```
results/exp_50percent/
├── models/
│   └── pruned_model.bin          # 剪枝后的模型
├── analysis/
│   ├── layer_removal_ppl.json    # 层级重要性分析
│   ├── block_removal_ppl.json    # 块级重要性分析
│   ├── global_group_table.csv    # Group 分析表
│   └── groups_to_prune.csv       # 剪枝决策表
├── evaluation/
│   └── evaluation_results.json   # 评估结果（如果运行了评估）
└── logs/
    └── ...                        # 日志文件
```

## 常见问题

### 1. 显存不足怎么办？

```bash
# 使用梯度检查点
python run_global_pruning.py \
  --base_model /path/to/model \
  --output_name exp \
  --pruning_ratio 0.5 \
  --use_gradient_checkpointing \
  --gradient_batch_size 2
```

### 2. 如何选择剪枝率？

- **30-50%**: 推荐范围，性能损失较小
- **50-70%**: 激进剪枝，建议配合微调
- **>70%**: 极限剪枝，性能会显著下降

### 3. 温度参数如何选择？

- 剪枝率 ≤ 50%: `T=1.0`（默认）
- 剪枝率 50-70%: `T=1.5`
- 剪枝率 > 70%: `T=2.0`

### 4. 何时需要微调？

- 剪枝率 < 30%: 通常不需要
- 剪枝率 30-50%: 可选，轻微提升
- 剪枝率 > 50%: **强烈推荐**

## 性能参考

以 LLaMA-3-8B 为例（单卡 A100）：

| 剪枝率 | 耗时 | 显存占用 |
|--------|------|----------|
| 30% | ~15 分钟 | ~25GB |
| 50% | ~20 分钟 | ~28GB |
| 70% | ~25 分钟 | ~30GB |

*注：包含 Taylor 重要性计算 + 层/块分析 + 剪枝执行*

## 技术细节

### H-GSP 算法流程

1. **Taylor 重要性计算**：基于一阶/二阶梯度
2. **层级重要性分析**：计算每层移除后的 PPL 增量
3. **块级重要性分析**：分别分析 Attention 和 MLP 块
4. **混合加权评分**：
   - 计算动态阈值 τ（25 分位数）
   - Layer-Dominant 模式：ppl_layer < τ
   - Block-Dominant 模式：ppl_layer ≥ τ
   - 温度调制：score = taylor × (ln(1+ppl))^T
5. **全局优化**：分数背包算法选择最优剪枝组合
6. **自动坍缩**：层保留率 < ε 时整层移除

### GQA-Aware 剪枝

自动识别 LLaMA 的 Grouped Query Attention 结构：
- 4:1 的 Q:KV 比例
- 保持 GQA 组完整性
- 避免破坏注意力机制

## 联系与反馈

有问题或建议请提交 Issue 到项目仓库。
