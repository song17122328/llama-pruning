# 多模型剪枝指南（LLaMA / Qwen / Mistral）

本文档提供 LLaMA-3-8B、Qwen2.5-7B 和 Mistral-7B-v0.3 模型的全局剪枝命令和使用指南。

## 📊 支持的模型架构

| 模型 | Q Heads | KV Heads | GQA Ratio | 参数量 | 特殊机制 |
|------|---------|----------|-----------|--------|----------|
| **LLaMA-3-8B** | 32 | 8 | **4:1** | 8B | 标准 Attention |
| **Mistral-7B-v0.3** | 32 | 8 | **4:1** | 7B | 标准 Attention（v0.2+ 已取消滑动窗口） |
| **Qwen2.5-7B** | 28 | 4 | **7:1** | 7B | 标准 Attention |

**关键特性**：
- ✅ 自动检测 GQA 配置（无需手动指定参数）
- ✅ 支持不同 GQA 比例（4:1 和 7:1）
- ✅ 统一的剪枝接口

---

## 📁 目录结构

```
results/
├── LLaMA-3-8B/           # LLaMA 全局剪枝结果
├── Qwen2.5-7B/           # Qwen 全局剪枝结果
└── Mistral-7B-v0.3/      # Mistral 全局剪枝结果
```

---

## 🔧 全局剪枝命令

### LLaMA-3-8B（基准模型）

#### 20% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --output_name LLaMA-3-8B/global_prune_20 \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --device cuda:0
```

#### 30% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --output_name LLaMA-3-8B/global_prune_30 \
  --pruning_ratio 0.3 \
  --temperature 0.0 \
  --device cuda:0
```

#### 50% 稀疏度（极端测试）
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --output_name LLaMA-3-8B/global_prune_50 \
  --pruning_ratio 0.5 \
  --temperature 0.0 \
  --device cuda:0
```

---

### Qwen2.5-7B

**架构特点**：
- GQA Ratio: **7:1**（不同于 LLaMA/Mistral 的 4:1）
- 总层数: 28
- 自动检测配置，无需手动指定

#### 20% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/global_prune_20 \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --device cuda:0
```

#### 30% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/global_prune_30 \
  --pruning_ratio 0.3 \
  --temperature 0.0 \
  --device cuda:0
```

#### 50% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/global_prune_50 \
  --pruning_ratio 0.5 \
  --temperature 0.0 \
  --device cuda:0
```

---

### Mistral-7B-v0.3

**架构特点**：
- GQA Ratio: **4:1**（与 LLaMA 相同）
- 总层数: 32
- v0.3 已移除滑动窗口，使用标准全注意力

#### 20% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/global_prune_20 \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --device cuda:0
```

#### 30% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/global_prune_30 \
  --pruning_ratio 0.3 \
  --temperature 0.0 \
  --device cuda:0
```

#### 50% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/global_prune_50 \
  --pruning_ratio 0.5 \
  --temperature 0.0 \
  --device cuda:0
```

---

## 🔄 微调命令（剪枝后性能恢复）

### 集成微调（推荐）

剪枝时直接启用微调：

```bash
# LLaMA 3 8B - 20% 剪枝 + 微调
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --output_name LLaMA-3-8B/prune_20_finetune \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --finetune \
  --finetune_data_path yahma/alpaca-cleaned \
  --finetune_epochs 3 \
  --finetune_lr 3e-4 \
  --finetune_batch_size 128 \
  --finetune_micro_batch_size 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --device cuda:0

# Qwen 2.5 7B - 20% 剪枝 + 微调
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/prune_20_finetune \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --finetune \
  --finetune_data_path yahma/alpaca-cleaned \
  --finetune_epochs 3 \
  --finetune_lr 3e-4 \
  --device cuda:0

# Mistral 7B v0.3 - 20% 剪枝 + 微调
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/prune_20_finetune \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --finetune \
  --finetune_data_path yahma/alpaca-cleaned \
  --finetune_epochs 3 \
  --finetune_lr 3e-4 \
  --device cuda:0
```

### 独立微调

如果已有剪枝模型，可以单独运行微调：

```bash
# 使用 finetune_lora.py 单独微调
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
  --pruned_model results/Qwen2.5-7B/global_prune_20/pruned_model.bin \
  --data_path yahma/alpaca-cleaned \
  --output_dir results/Qwen2.5-7B/prune_20_finetuned \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --batch_size 128 \
  --micro_batch_size 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --device cuda:0
```

---

## 📊 三模型对比实验设计

### 实验目标
- **相同 GQA 比例验证**：LLaMA-3 (4:1) vs Mistral (4:1)
- **不同 GQA 比例验证**：LLaMA-3 (4:1) vs Qwen (7:1)
- **算法泛化性验证**：三模型在不同稀疏度下的表现

### 测试矩阵

| 模型 | 稀疏度 | 目的 | 优先级 |
|------|--------|------|--------|
| LLaMA-3-8B | 20%, 30%, 50% | 基准对比 | ⭐⭐⭐⭐⭐ |
| Mistral-7B-v0.3 | 20%, 30% | 验证相同 GQA (4:1) | ⭐⭐⭐⭐⭐ |
| Qwen2.5-7B | 20%, 30% | 验证不同 GQA (7:1) | ⭐⭐⭐⭐⭐ |

### 批量运行脚本

创建 `scripts/run_all_experiments.sh`：

```bash
#!/bin/bash
# 批量运行三模型对比实验

# LLaMA-3-8B
for sparsity in 0.2 0.3 0.5; do
  CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
    --base_model meta-llama/Meta-Llama-3-8B \
    --output_name LLaMA-3-8B/prune_$(echo "$sparsity * 100" | bc | cut -d. -f1) \
    --pruning_ratio $sparsity \
    
    --device cuda:0
    
done

# Mistral-7B-v0.3
for sparsity in 0.2 0.3; do
  CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
    --base_model mistralai/Mistral-7B-v0.3 \
    --output_name Mistral-7B-v0.3/prune_$(echo "$sparsity * 100" | bc | cut -d. -f1) \
    --pruning_ratio $sparsity \
    
    --device cuda:0
    
done

# Qwen2.5-7B
for sparsity in 0.2 0.3; do
  CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
    --base_model Qwen/Qwen2.5-7B \
    --output_name Qwen2.5-7B/prune_$(echo "$sparsity * 100" | bc | cut -d. -f1) \
    --pruning_ratio $sparsity \
    
    --device cuda:0
    
done
```

---

## ⚙️ 核心参数说明

### 必需参数
- `--base_model`: 模型路径或 HuggingFace 模型 ID
- `--output_name`: 输出目录名（保存在 `results/{output_name}/`）
- `--pruning_ratio`: 目标稀疏度（0.2 = 20%）

### 剪枝参数
- `--importance_method`: 重要性度量
  - `taylor`: Taylor 一阶（默认，推荐）
  - `taylor_2nd`: Taylor 二阶（更精确，更慢）
  - `wanda`: Wanda 方法
  - `magnitude`: 权重大小
- `--dataset`: 校准数据集（wikitext2 / ptb / c4，默认 wikitext2）
- `--temperature`: H-GSP 温度参数（默认 1.0）
  - **设为 0.0**: 只使用全局 Taylor，跳过层级/块级重要性（推荐，避免模型兼容性问题）
  - **设为 1.0**: 使用完整 H-GSP 层次化剪枝策略
- `--epsilon`: H-GSP 坍缩阈值（默认 0.15）

### 微调参数
- `--finetune`: 启用微调
- `--finetune_data_path`: 微调数据集
- `--finetune_epochs`: 微调轮数（推荐 3-5）
- `--finetune_lr`: 学习率（推荐 3e-4）
- `--lora_r`: LoRA 秩（推荐 8-16）
- `--lora_alpha`: LoRA 缩放（推荐 2×r）

### 评估参数
- `--skip_evaluation`: 跳过自动评估（节省时间）
- `--eval_tasks`: 评估任务（默认：wikitext,c4）

---

## 🚀 快速开始

### 1. 快速验证（10% 稀疏度）

```bash
# 快速测试 Qwen（验证环境）
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/quick_test \
  --pruning_ratio 0.1 \
  --temperature 0.0 \
  --temperature 0.0 \
  --device cuda:0

# 快速测试 Mistral
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/quick_test \
  --pruning_ratio 0.1 \
  --temperature 0.0 \
  --temperature 0.0 \
  --device cuda:0
```

**推荐配置说明**：
- `--temperature 0.0`：只使用全局 Taylor 重要性，跳过层级/块级重要性测试
- ✅ **优势**：避免模型兼容性问题（如 Qwen/Mistral 的层恒等映射）
- ✅ **性能**：全局 Taylor 方法已被实验证明效果最好
- ✅ **速度**：跳过层级分析，加快剪枝速度

### 2. 标准剪枝（20% 稀疏度，推荐配置）

```bash
# Qwen（推荐：temperature=0）
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/prune_20 \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --temperature 0.0 \
  --device cuda:0

# Mistral（推荐：temperature=0）
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/prune_20 \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --temperature 0.0 \
  --device cuda:0

# LLaMA（可使用完整 H-GSP）
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --output_name LLaMA-3-8B/prune_20 \
  --pruning_ratio 0.2 \
  --temperature 0.0 \
  --temperature 1.0 \
  --device cuda:0
```

---

## 📈 评估剪枝模型

### 使用内置评估
```bash
python evaluation/run_evaluation.py \
  --model_path results/Qwen2.5-7B/prune_20/pruned_model.bin \
  --base_model Qwen/Qwen2.5-7B \
  --tasks wikitext,c4,lambada \
  --device cuda:0
```

### Python API
```python
from evaluation.metrics.ppl import PPLMetric
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载剪枝模型
model = AutoModelForCausalLM.from_pretrained('results/Qwen2.5-7B/prune_20')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')

# 评估
ppl = PPLMetric(model, tokenizer, datasets=['wikitext2'], device='cuda')
print(f"WikiText-2 PPL: {ppl['wikitext2']}")
```

---

## 📝 注意事项

### 显存要求
- **Qwen2.5-7B**: ~14GB（FP16）
- **Mistral-7B-v0.3**: ~14GB（FP16）
- **LLaMA-3-8B**: ~16GB（FP16）
- 推荐 GPU: A100 / V100 / 3090 / 4090

### 模型下载
```bash
# Qwen（国内用户推荐 ModelScope）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-7B')"

# 或使用 HuggingFace
huggingface-cli download Qwen/Qwen2.5-7B

# Mistral
huggingface-cli download mistralai/Mistral-7B-v0.3
```

### 输出文件
```
results/{output_name}/
├── pruned_model.bin              # 剪枝后的模型权重
├── config.json                   # 模型配置
├── pruning_analysis.json         # 剪枝分析报告
├── global_group_table.csv        # 全局分组表
└── logs/
    └── training.log              # 详细日志
```

---

## 🐛 常见问题

### Q1: 自动检测 GQA 配置失败？
**A**: 检查模型配置是否包含 `num_key_value_heads` 字段。代码会自动处理，如果检测失败会给出警告。

### Q2: OOM（显存不足）？
**A**: 减少 `--nsamples`（如 64 或 32），或使用 `--use_gradient_checkpointing`。

### Q3: 如何选择稀疏度？
**A**:
- 20%: 平衡性能和压缩率，推荐起点
- 30%: 需要微调恢复性能
- 50%: 极端测试，必须微调

### Q4: Qwen 和 Mistral 的剪枝效果差异？
**A**: Qwen 使用 7:1 GQA，理论上每个 KV head 承载更多信息，剪枝时需要更谨慎。建议从 20% 开始测试。

---

## 📚 相关文档

- [项目总览](../README.md)
- [使用指南](../USAGE.md)
- [微调文档](./FINETUNING.md)
- [项目结构](../PROJECT_STRUCTURE.md)

---

**最后更新**: 2024-11-29
