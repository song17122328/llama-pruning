# LoRA 微调指南

本文档说明如何使用 LoRA 对剪枝后的模型进行微调，以恢复模型性能。

## 📋 目录

- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [使用方法](#使用方法)
- [参数说明](#参数说明)
- [输出说明](#输出说明)
- [常见问题](#常见问题)

---

## 快速开始

```bash
# 1. 安装依赖
pip install peft datasets transformers

# 2. 微调剪枝后的模型
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4

# 3. 查看结果
# 微调后的模型保存在: results/HGSP_2000_finetuned/
# 评估结果: results/HGSP_2000_finetuned/evaluation/evaluation_results.json
```

---

## 环境要求

### 必需依赖

```bash
pip install peft>=0.5.0 datasets>=2.14.0 transformers>=4.33.0
```

### 可选依赖

```bash
# WandB (用于训练监控)
pip install wandb

# 评估工具
pip install lm-eval>=0.4.0
```

### 硬件要求

| 模型大小 | 推荐显存 | 推荐配置 |
|---------|---------|---------|
| 2B 参数 | 16GB | 1x RTX 3090 |
| 4B 参数 | 24GB | 1x RTX 3090 Ti |
| 7B 参数 | 40GB | 1x A100 40GB |

**注意**: LoRA 微调比全参数微调节省约 70% 显存

---

## 使用方法

### 1. 基本用法

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned
```

### 2. 自定义 LoRA 参数

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 16 \              # 增大 LoRA 秩
    --lora_alpha 32 \          # 调整缩放系数
    --lora_dropout 0.1         # 增加 dropout
```

### 3. 调整训练参数

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --num_epochs 3 \           # 增加训练轮数
    --learning_rate 2e-4 \     # 提高学习率
    --batch_size 128 \         # 增大batch size
    --micro_batch_size 8       # 增大micro batch size
```

### 4. 使用自定义数据集

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path /path/to/your/dataset \
    --prompt_template_name custom
```

### 5. 跳过自动评估

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --skip_evaluation
```

### 6. 使用 WandB 监控

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --wandb_project "llama-pruning-finetune"
```

---

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--pruned_model` | 剪枝模型路径 | `results/HGSP_2000/pruned_model.bin` |

### 数据相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `yahma/alpaca-cleaned` | 训练数据集路径 |
| `--val_set_size` | `2000` | 验证集大小 |
| `--cutoff_len` | `256` | 最大序列长度 |
| `--prompt_template_name` | `alpaca` | 提示词模板 |

### 训练超参数

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `--batch_size` | `64` | 总batch size | 32-128 |
| `--micro_batch_size` | `4` | 每GPU的batch size | 1-8 |
| `--num_epochs` | `2` | 训练轮数 | 1-5 |
| `--learning_rate` | `1e-4` | 学习率 | 5e-5 ~ 3e-4 |

**梯度累积步数** = `batch_size / micro_batch_size`

### LoRA 配置

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `--lora_r` | `8` | LoRA 秩 | 4-32 |
| `--lora_alpha` | `16` | LoRA 缩放系数 | 8-64 |
| `--lora_dropout` | `0.05` | LoRA dropout | 0.0-0.1 |
| `--lora_target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj` | 目标模块 | - |

**LoRA 参数建议**:
- 更大的模型 → 更大的 `lora_r` (16-32)
- 更小的模型 → 更小的 `lora_r` (4-8)
- `lora_alpha` 通常设为 `lora_r * 2`

### 输出相关

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `results/<model_name>_finetuned` | 输出目录 |
| `--skip_evaluation` | `False` | 跳过自动评估 |

### 其他选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_on_inputs` | `False` | 在输入部分计算loss |
| `--add_eos_token` | `False` | 添加EOS token |
| `--group_by_length` | `False` | 按长度分组 |
| `--wandb_project` | `""` | WandB 项目名称 |
| `--resume_from_checkpoint` | `None` | 从检查点恢复 |

---

## 输出说明

### 目录结构

微调完成后，输出目录结构如下：

```
results/HGSP_2000_finetuned/
├── pruned_model.bin           # 微调后的完整模型
├── lora_adapter/              # LoRA adapter (可单独使用)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer 相关文件
├── evaluation/                # 评估结果
│   └── evaluation_results.json
└── checkpoint-*/              # 训练检查点 (可选)
```

### 使用微调后的模型

```python
import torch

# 加载微调后的模型
model_dict = torch.load('results/HGSP_2000_finetuned/pruned_model.bin')
model = model_dict['model']
tokenizer = model_dict['tokenizer']

# 使用模型
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

### 评估结果

`evaluation/evaluation_results.json` 包含:
- PPL (WikiText-2, PTB)
- Zero-shot 准确率 (BoolQ, PIQA, HellaSwag, 等)
- 推理速度和延迟
- 显存占用

---

## 常见问题

### Q1: 显存不足

**问题**: CUDA out of memory

**解决方法**:
```bash
# 1. 减小 micro_batch_size
python finetune_lora.py ... --micro_batch_size 2

# 2. 减小 cutoff_len
python finetune_lora.py ... --cutoff_len 128

# 3. 减小 LoRA 秩
python finetune_lora.py ... --lora_r 4
```

### Q2: 训练速度慢

**问题**: 训练速度很慢

**解决方法**:
```bash
# 1. 增大 micro_batch_size (如果显存允许)
python finetune_lora.py ... --micro_batch_size 8

# 2. 启用 group_by_length
python finetune_lora.py ... --group_by_length

# 3. 减小验证集大小
python finetune_lora.py ... --val_set_size 1000
```

### Q3: Loss 不下降

**问题**: 训练时 loss 不下降

**解决方法**:
```bash
# 1. 提高学习率
python finetune_lora.py ... --learning_rate 2e-4

# 2. 增大 LoRA 秩
python finetune_lora.py ... --lora_r 16

# 3. 增加训练轮数
python finetune_lora.py ... --num_epochs 5
```

### Q4: 如何只使用 LoRA adapter

**问题**: 不想保存完整模型，只想要 LoRA adapter

**解决方法**:

LoRA adapter 已经保存在 `<output_dir>/lora_adapter/` 目录下，可以这样使用：

```python
from peft import PeftModel

# 加载原始剪枝模型
pruned_dict = torch.load('results/HGSP_2000/pruned_model.bin')
base_model = pruned_dict['model']

# 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, 'results/HGSP_2000_finetuned/lora_adapter')
```

### Q5: 评估失败

**问题**: 自动评估失败

**解决方法**:

微调完成后手动运行评估：

```bash
python evaluation/run_evaluation.py \
    --model_path results/HGSP_2000_finetuned/pruned_model.bin \
    --metrics all \
    --output results/HGSP_2000_finetuned/evaluation/evaluation_results.json
```

### Q6: 使用自己的数据集

**问题**: 如何使用自定义数据集

**解决方法**:

数据集需要 Alpaca 格式的 JSON，包含以下字段：
- `instruction`: 指令
- `input`: 输入（可选）
- `output`: 期望输出

示例:
```json
[
  {
    "instruction": "总结以下文本",
    "input": "这是一段很长的文本...",
    "output": "这是总结..."
  }
]
```

然后使用 Hugging Face datasets 加载或本地路径。

---

## 高级用法

### 多GPU训练

```bash
# 使用 torchrun (推荐)
torchrun --nproc_per_node=4 finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --batch_size 256 \
    --micro_batch_size 4

# 或使用 accelerate
accelerate launch finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned
```

### 从检查点恢复

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --resume_from_checkpoint results/HGSP_2000_finetuned/checkpoint-200
```

### 完整的生产级命令

```bash
python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir results/HGSP_2000_finetuned_production \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --batch_size 128 \
    --micro_batch_size 8 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --group_by_length \
    --wandb_project "llama-pruning-production"
```

---

## 参考资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 文档](https://huggingface.co/docs/peft)
- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)
- [LLM-Pruner](https://github.com/horseee/LLM-Pruner)

---

**维护者**: LLaMA Pruning Research Team
**最后更新**: 2025-11-23
