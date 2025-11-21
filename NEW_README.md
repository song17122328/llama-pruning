# LLaMA Pruning Toolkit

高效的LLaMA模型结构化剪枝工具，支持全局剪枝和层级剪枝两种方法。

## ✨ 特性

- 🎯 **GQA架构感知**：自动维护4:1 Q:KV head比例
- 🔬 **多种重要性度量**：Taylor一阶/二阶、Wanda
- 🚀 **全局优化**：基于性价比的分数背包剪枝
- 🔧 **层级控制**：非均衡剪枝策略，保护重要层
- 💪 **微调恢复**：支持全参数和LoRA微调

## 📦 安装

```bash
# 克隆仓库
git clone <your-repo-url>
cd llama-pruning

# 安装依赖
pip install -r requirements.txt
```

**依赖**：torch, transformers, datasets, peft, pandas

## 🚀 快速开始

### 方法1：全局剪枝（推荐）

基于性价比得分（Importance/Cost）全局选择最优剪枝策略。

```bash
python global_pruning.py \
    --base_model /path/to/llama-3-8b \
    --save_ckpt_log_name my_experiment \
    --pruning_ratio 0.25 \
    --importance_method taylor \
    --num_samples 128 \
    --test_after_prune \
    --save_model
```

**核心参数**：
- `--pruning_ratio`: 剪枝率（0.25 = 25%）
- `--importance_method`: taylor（一阶）/ taylor_2nd（二阶）/ wanda
- `--num_samples`: 重要性评估样本数
- `--remove_empty_layers`: 自动移除剪空的层（深度剪枝）

### 方法2：层级剪枝（传统）

先评估层重要性，再为每层分配剪枝率。

```bash
python layer_pruning.py \
    --base_model /path/to/llama-3-8b \
    --save_ckpt_log_name my_experiment \
    --pruning_ratio 0.25 \
    --pruning_distribution 2:8 \
    --pruning_strategy inverse \
    --test_after_prune \
    --save_model
```

**核心参数**：
- `--pruning_distribution`: Attention:MLP剪枝比例（如2:8）
- `--pruning_strategy`: inverse（重要层少剪）/ uniform（均匀）
- `--freeze_top_n_layers`: 冻结最重要的N层

## 📊 两种方法对比

| 特性 | 全局剪枝 | 层级剪枝 |
|------|---------|---------|
| **优化目标** | 全局最优 | 层级最优 |
| **Attn:MLP** | 自动平衡 | 需手动指定 |
| **深度剪枝** | ✅ 自动 | ❌ |
| **计算时间** | 较慢 | 较快 |
| **PPL** | 最优 | 良好 |
| **推荐场景** | 追求极致性能 | 快速原型 |

**典型结果**（LLaMA-3-8B，剪枝25%）：
- 原始模型：PPL 12.3
- 全局剪枝（taylor_2nd）：PPL 58.9
- 层级剪枝（2:8, inverse）：PPL 83.8
- + LoRA微调：PPL 18.5

## 🔧 微调恢复

剪枝后使用LoRA微调恢复性能：

```bash
# 全局剪枝 + LoRA微调
python global_pruning.py \
    --base_model /path/to/llama-3-8b \
    --pruning_ratio 0.25 \
    --finetune \
    --finetune_method lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --finetune_samples 1000 \
    --finetune_lr 1e-4 \
    --test_after_prune \
    --save_model
```

**微调参数**：
- `--finetune_method`: full（全参数）/ lora（推荐）
- `--lora_r`: LoRA秩（4-16）
- `--lora_alpha`: 缩放系数（通常=2×r）
- `--finetune_lr`: 学习率（LoRA建议1e-4，全参数建议1e-5）

## 📈 评估

```python
from evaluation.metrics.ppl import PPLMetric
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained('/path/to/pruned_model')
tokenizer = AutoTokenizer.from_pretrained('/path/to/llama-3-8b')

# 评估PPL
ppl = PPLMetric(model, tokenizer, datasets=['wikitext2'], seq_len=128, device='cuda')
print(ppl)  # {'wikitext2 (wikitext-2-raw-v1)': 58.9}
```

## 📂 输出文件

运行后生成：

```
prune_log/my_experiment/
├── description.txt              # 实验配置
├── global_group_table.csv       # 全局分析表（仅全局剪枝）
├── layer_importance_config.json # 层重要性（仅层级剪枝）
├── pruning_strategy.png         # 剪枝策略可视化
├── pytorch_model.bin            # 剪枝后模型
└── YYYYMMDD_HHMMSS/
    └── training.log             # 详细日志
```

## 💡 使用建议

### 剪枝率选择

| 剪枝率 | 推荐方法 | 是否微调 | PPL退化 |
|--------|---------|---------|---------|
| 15-20% | 全局/层级均可 | 可选 | < 10% |
| 20-30% | 全局剪枝 | **推荐** | 10-30% |
| 30-40% | 全局剪枝 | **必须** | > 30% |

### 重要性方法选择

- **taylor**：平衡精度和速度，大多数场景推荐
- **taylor_2nd**：最高精度，愿意牺牲计算时间时使用
- **wanda**：快速原型验证，无需梯度计算

### 层级剪枝分布推荐

对于LLaMA-3-8B（Attention占19.2%，MLP占80.8%）：
- **2:8**：均衡剪枝率（推荐）
- **0:10**：只剪MLP，保护Attention
- **5:5**：等量剪枝参数

## 🛠️ 高级用法

### 仅剪枝Attention或MLP

```bash
# 只剪MLP
python layer_pruning.py \
    --pruning_distribution 0:10 \
    --pruning_ratio 0.25

# 只剪Attention
python layer_pruning.py \
    --pruning_distribution 10:0 \
    --pruning_ratio 0.25
```

### 保护关键层

```bash
# 冻结最重要的3层
python layer_pruning.py \
    --freeze_top_n_layers 3 \
    --pruning_ratio 0.25
```

### 深度剪枝（自动移除空层）

```bash
python global_pruning.py \
    --pruning_ratio 0.30 \
    --remove_empty_layers
```

## 🐛 故障排除

**CUDA OOM**：
```bash
--num_samples 50             # 减少样本数
--gradient_batch_size 2      # 减小批次大小
--seq_len 64                 # 减小序列长度
```

**PPL过高**：
- 降低剪枝率（0.15-0.20）
- 使用全局剪枝而非层级剪枝
- 启用微调恢复
- 尝试二阶Taylor重要性

## 📚 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{llama_pruning_toolkit,
  title={LLaMA Pruning Toolkit: GQA-Aware Structured Pruning},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/llama-pruning}}
}
```

## 📄 License

MIT License

---

**核心文件**：
- `global_pruning.py` - 全局剪枝主脚本
- `layer_pruning.py` - 层级剪枝主脚本
- `core/methods/global_pruning.py` - 全局剪枝算法
- `core/methods/gqa_aware.py` - GQA感知剪枝
- `core/importance/layer_analyzer.py` - 层重要性分析
- `core/trainer/finetuner.py` - LoRA微调
- `evaluation/metrics/ppl.py` - PPL评估