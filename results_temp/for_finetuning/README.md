# LoRA微调工作流程文档

本目录包含为LoRA微调准备的12个剪枝模型配置（6个模型 × 2种配置）。

## 📊 选择的模型

每个模型选择了2个配置：
- **best_acc**: ACC最高的配置（zero-shot性能最优）
- **best_ppl**: PPL最低的配置（语言建模能力最优）

总共12个配置需要微调和评估。

## 📁 目录结构

```
results/
├── for_finetuning/              # 剪枝模型（微调前）
│   ├── Llama/
│   │   ├── best_acc/            # ACC: 0.5946, PPL: 13.39
│   │   └── best_ppl/            # PPL: 11.82, ACC: 0.5665
│   ├── Llama-Instruct/
│   │   ├── best_acc/            # ACC: 0.6318, PPL: 13.29
│   │   └── best_ppl/            # PPL: 13.20, ACC: 0.6247
│   ├── Qwen/
│   │   ├── best_acc/            # ACC: 0.6094, PPL: 14.46
│   │   └── best_ppl/            # PPL: 11.45, ACC: 0.5633
│   ├── Qwen-Instruct/
│   │   ├── best_acc/            # ACC: 0.6202, PPL: 13.42
│   │   └── best_ppl/            # PPL: 12.69, ACC: 0.6116
│   ├── Mistral/
│   │   ├── best_acc/            # ACC: 0.5861, PPL: 15.22
│   │   └── best_ppl/            # PPL: 9.69, ACC: 0.5332
│   └── Mistral-Instruct/
│       ├── best_acc/            # ACC: 0.6552, PPL: 24.33
│       └── best_ppl/            # PPL: 9.09, ACC: 0.6023
│
├── finetuned/                   # 微调后的模型
│   ├── Llama/
│   │   ├── best_acc_finetuned/
│   │   │   ├── adapter_model.bin      # LoRA权重
│   │   │   ├── adapter_config.json    # LoRA配置
│   │   │   ├── finetuning_config.json # 微调参数
│   │   │   └── training_log.txt       # 训练日志
│   │   └── best_ppl_finetuned/
│   └── ... (其他5个模型)
│
└── finetuned_evaluation/         # 微调后的评估结果
    ├── Llama/
    │   ├── best_acc_finetuned/
    │   │   ├── evaluation_results.json  # 评估结果
    │   │   ├── ppl_results.json        # PPL详细结果
    │   │   └── comparison_report.txt   # 微调前后对比
    │   └── best_ppl_finetuned/
    └── ... (其他5个模型)
```

## 🚀 工作流程

### 步骤1: 选择模型（已完成）

```bash
python param_search/select_best_for_finetuning.py
```

这会为每个模型选择ACC最高和PPL最低的配置，并复制到 `results/for_finetuning/`。

### 步骤2: LoRA微调

#### 微调单个模型配置

```bash
# 微调 Llama 的 best_acc 配置
python run_finetuning_workflow.py --model Llama --config best_acc --stage finetune

# 微调 Llama 的 best_ppl 配置
python run_finetuning_workflow.py --model Llama --config best_ppl --stage finetune
```

#### 批量微调所有12个配置

```bash
python run_finetuning_workflow.py --batch-all --stage finetune
```

### 步骤3: 评估微调后的模型

#### 评估单个模型

```bash
python run_finetuning_workflow.py --model Llama --config best_acc --stage evaluate
```

#### 批量评估所有12个配置

```bash
python run_finetuning_workflow.py --batch-all --stage evaluate
```

### 步骤4: 对比分析

#### 对比单个模型微调前后性能

```bash
python run_finetuning_workflow.py --model Llama --config best_acc --stage compare
```

#### 批量对比所有配置

```bash
python run_finetuning_workflow.py --batch-all --stage compare
```

### 完整流程（推荐）

```bash
# 对单个配置执行：微调 → 评估 → 对比
python run_finetuning_workflow.py --model Llama --config best_acc --stage all

# 对所有12个配置执行完整流程
python run_finetuning_workflow.py --batch-all --stage all
```

## ⚙️ LoRA微调参数

默认配置（可在脚本中修改）：

```json
{
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "num_epochs": 2,
  "learning_rate": 1e-4,
  "batch_size": 64,
  "micro_batch_size": 4
}
```

## 📈 评估指标

微调前后会评估以下指标：

### PPL (困惑度)
- WikiText2
- PTB

### Zero-shot ACC
- BoolQ
- PIQA
- HellaSwag
- WinoGrande
- ARC-Easy
- ARC-Challenge
- OpenBookQA

## 📊 预期结果分析

### best_acc配置
- **目标**: 验证高ACC配置通过微调能否进一步提升
- **关注指标**: Zero-shot ACC的变化
- **预期**: ACC应该有所提升，PPL可能略有改善

### best_ppl配置
- **目标**: 验证低PPL配置通过微调能否平衡性能
- **关注指标**: PPL保持优势同时ACC的提升
- **预期**: PPL保持较低，ACC显著提升

## 🔍 对比维度

微调后会生成对比报告，包含：

1. **PPL对比**
   - 微调前 vs 微调后
   - 变化量和变化百分比

2. **Zero-shot ACC对比**
   - 每个任务的ACC变化
   - 平均ACC变化
   - 最大提升和最小提升的任务

3. **配置间对比**
   - best_acc vs best_ppl微调后表现
   - 哪种配置更适合微调恢复

## 📝 注意事项

1. **模型权重**: 确保剪枝后的模型权重文件存在
2. **微调脚本**: 需要有compatible的LoRA微调脚本（`finetune_lora.py`）
3. **评估脚本**: 需要评估脚本支持LoRA模型（`run_evaluation.py`）
4. **GPU内存**: 微调需要足够的GPU内存（建议40GB+）
5. **时间估算**: 每个配置微调约需要2-4小时（取决于硬件和数据集）

## 🎯 研究价值

这个实验设计可以回答以下研究问题：

1. **最佳剪枝配置选择**: ACC最优 vs PPL最优，哪个更适合后续微调？
2. **性能恢复能力**: LoRA微调能在多大程度上恢复剪枝损失的性能？
3. **指标trade-off**: 微调后ACC和PPL的平衡关系如何变化？
4. **架构差异**: 不同模型架构（Llama/Qwen/Mistral）在微调恢复上的差异
5. **Instruct vs Base**: 指令微调模型和基础模型在剪枝+LoRA场景下的表现差异

## 📚 相关文件

- `SUMMARY.md`: 模型选择摘要
- `selection_info.json`: 每个配置的详细选择信息（位于各配置目录）
- `finetuning_config.json`: 微调参数（微调后生成）
- `comparison_report.txt`: 微调前后对比报告（评估后生成）

## 🔗 下一步

完成微调和评估后，可以：

1. 生成综合分析报告
2. 可视化微调前后的性能变化
3. 撰写论文，对比不同配置的微调效果
4. 选择最佳的"剪枝+微调"pipeline

---

**开始时间**: 运行脚本时记录
**预计完成时间**: 约24-48小时（取决于硬件）
**GPU需求**: 1-2块A100/H100（40GB+）
