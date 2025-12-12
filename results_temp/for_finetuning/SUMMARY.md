# 微调前的模型选择摘要

本目录包含为LoRA微调准备的剪枝模型。

## 选择标准

- **best_acc**: ACC最高的配置（评估zero-shot任务性能恢复）
- **best_ppl**: PPL最低的配置（评估语言建模能力恢复）

## 模型列表

| 模型 | 类型 | best_acc | best_ppl | 是否相同 |
|------|------|----------|----------|----------|
| Llama | Base | ACC:0.5946 | PPL:11.82 |  |
| Llama-Instruct | Instruct | ACC:0.6318 | PPL:13.20 |  |
| Qwen | Base | ACC:0.6094 | PPL:11.45 |  |
| Qwen-Instruct | Instruct | ACC:0.6202 | PPL:12.69 |  |
| Mistral | Base | ACC:0.5861 | PPL:9.69 |  |
| Mistral-Instruct | Instruct | ACC:0.6552 | PPL:9.09 |  |

## 下一步

1. 对每个配置运行LoRA微调
2. 评估微调后的模型
3. 对比微调前后的性能

详细信息参见各模型目录下的 `selection_info.json` 文件。
