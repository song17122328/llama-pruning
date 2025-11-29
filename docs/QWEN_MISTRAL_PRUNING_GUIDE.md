# Qwen 2.5 & Mistral 7B 剪枝与微调指南

本文档提供 Qwen2.5-7B 和 Mistral-7B-v0.3 模型的剪枝与微调命令。

## 📁 目录结构

```
prune_log/
├── Qwen2.5-7B/           # Qwen 层级剪枝结果（layer_pruning.py）
└── Mistral-7B-v0.3/      # Mistral 层级剪枝结果（layer_pruning.py）

results/
├── Qwen2.5-7B/           # Qwen 全局剪枝结果（run_global_pruning.py）
└── Mistral-7B-v0.3/      # Mistral 全局剪枝结果（run_global_pruning.py）
```

---

## 🔧 Qwen 2.5 7B 剪枝命令

### 架构特点
- **Q Heads**: 28
- **KV Heads**: 4
- **GQA Ratio**: 7:1
- **Head Dim**: 128
- **总层数**: 28

### 1. 层级剪枝（Layer Pruning）- 推荐

#### 20% 剪枝率
```bash
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --save_ckpt_log_name Qwen2.5-7B/prune_20 \
  --pruning_ratio 0.2 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 1.0 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
```

#### 30% 剪枝率（更激进）
```bash
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --save_ckpt_log_name Qwen2.5-7B/prune_30 \
  --pruning_ratio 0.3 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 1.5 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
```

#### 50% 剪枝率（极端测试）
```bash
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --save_ckpt_log_name Qwen2.5-7B/prune_50 \
  --pruning_ratio 0.5 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 2.0 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
```

### 2. 全局剪枝（Global Pruning）

#### 20% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --output_name Qwen2.5-7B/global_prune_20 \
  --target_sparsity 0.2 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model \
  --skip_evaluation
```

### 3. 微调命令

#### LoRA 微调（剪枝后恢复性能）
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
  --pruned_model prune_log/Qwen2.5-7B/prune_20/best_model.bin \
  --data_path yahma/alpaca-cleaned \
  --output_dir prune_log/Qwen2.5-7B/prune_20_finetuned \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --batch_size 128 \
  --micro_batch_size 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --device cuda:0
```

---

## 🔧 Mistral 7B v0.3 剪枝命令

### 架构特点
- **Q Heads**: 32
- **KV Heads**: 8
- **GQA Ratio**: 4:1
- **Head Dim**: 128
- **总层数**: 32
- **注意**: v0.3 已移除滑动窗口，使用标准全注意力机制

### 1. 层级剪枝（Layer Pruning）- 推荐

#### 20% 剪枝率
```bash
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/prune_20 \
  --pruning_ratio 0.2 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 1.0 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
```

#### 30% 剪枝率
```bash
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/prune_30 \
  --pruning_ratio 0.3 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 1.5 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
```

#### 50% 剪枝率
```bash
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/prune_50 \
  --pruning_ratio 0.5 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 2.0 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
```

### 2. 全局剪枝（Global Pruning）

#### 20% 稀疏度
```bash
CUDA_VISIBLE_DEVICES=0 python run_global_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --output_name Mistral-7B-v0.3/global_prune_20 \
  --target_sparsity 0.2 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model \
  --skip_evaluation
```

### 3. 微调命令

#### LoRA 微调
```bash
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
  --pruned_model prune_log/Mistral-7B-v0.3/prune_20/best_model.bin \
  --data_path yahma/alpaca-cleaned \
  --output_dir prune_log/Mistral-7B-v0.3/prune_20_finetuned \
  --num_epochs 3 \
  --learning_rate 3e-4 \
  --batch_size 128 \
  --micro_batch_size 4 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --device cuda:0
```

---

## 📊 三模型对比测试建议

### 相同剪枝率对比（验证算法泛化性）

```bash
# LLaMA 3 8B (baseline)
python layer_pruning.py \
  --base_model meta-llama/Meta-Llama-3-8B \
  --save_ckpt_log_name LLaMA-3-8B/prune_20 \
  --pruning_ratio 0.2 ...

# Mistral 7B v0.3 (相同 GQA 4:1)
python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/prune_20 \
  --pruning_ratio 0.2 ...

# Qwen 2.5 7B (不同 GQA 7:1)
python layer_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --save_ckpt_log_name Qwen2.5-7B/prune_20 \
  --pruning_ratio 0.2 ...
```

### 论文实验建议

**测试矩阵：**

| 模型 | 剪枝率 | 目的 |
|------|--------|------|
| LLaMA 3 8B | 20%, 30%, 50% | 基准对比 |
| Mistral 7B v0.3 | 20%, 30% | 验证相同 GQA 比例（4:1）|
| Qwen 2.5 7B | 20%, 30% | 验证不同 GQA 比例（7:1）|

---

## ⚙️ 参数说明

### 核心参数
- `--pruning_ratio`: 总剪枝率（0.2 = 20%）
- `--pruning_distribution`: Attention:MLP 剪枝比例（5:5 表示各一半）
- `--pruning_strategy`: 剪枝策略
  - `inverse`: 重要层剪少，不重要层剪多（推荐）
  - `proportional`: 重要层剪多
  - `uniform`: 均匀剪枝

### 层重要度参数
- `--layer_importance_method`:
  - `removal`: 通过移除层评估重要度（推荐）
  - `activation`: 通过激活值评估
- `--layer_importance_weight`: 层间差异系数（1.0-2.0）

### 通道/头重要度参数
- `--channel_importance_samples`: Taylor 重要性评估样本数（10-50）
- `--taylor_seq_len`: 序列长度（128）
- `--nsamples`: 校准样本数（128）

---

## 🚀 快速开始

### 最小测试（快速验证）
```bash
# Qwen 快速测试（10% 剪枝）
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model Qwen/Qwen2.5-7B \
  --save_ckpt_log_name Qwen2.5-7B/test \
  --pruning_ratio 0.1 \
  --nsamples 32 \
  --layer_importance_samples 10 \
  --channel_importance_samples 5 \
  --device cuda:0

# Mistral 快速测试
CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/test \
  --pruning_ratio 0.1 \
  --nsamples 32 \
  --layer_importance_samples 10 \
  --channel_importance_samples 5 \
  --device cuda:0
```

---

## 📝 注意事项

1. **自动配置检测**：代码会自动检测 GQA 配置，无需手动指定 `--gqa_ratio`
2. **显存要求**：
   - Qwen 2.5 7B: ~14GB
   - Mistral 7B v0.3: ~14GB
   - 建议使用 A100/V100/3090 以上
3. **下载模型**：
   - Qwen：可使用 ModelScope 镜像加速
   - Mistral：直接从 HuggingFace 下载
4. **结果保存**：
   - 模型权重：`prune_log/{model}/prune_{ratio}/best_model.bin`
   - 日志文件：`prune_log/{model}/prune_{ratio}/log.txt`
   - 配置文件：`prune_log/{model}/prune_{ratio}/config.json`

---

## 📈 评估命令

### 评估剪枝后的模型
```bash
# 评估 Qwen 剪枝模型
python evaluation/run_evaluation.py \
  --model_path prune_log/Qwen2.5-7B/prune_20/best_model.bin \
  --tasks wikitext,c4,lambada \
  --device cuda:0

# 评估 Mistral 剪枝模型
python evaluation/run_evaluation.py \
  --model_path prune_log/Mistral-7B-v0.3/prune_20/best_model.bin \
  --tasks wikitext,c4,lambada \
  --device cuda:0
```
