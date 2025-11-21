# LLaMA Pruning Toolkit

高效的LLaMA模型结构化剪枝工具，支持全局剪枝和层级剪枝两种方法。

## ✨ 特性

- 🎯 **结构化分组剪枝**：基于通道分组的端到端剪枝策略
- 🔬 **多种重要性度量**：Taylor一阶/二阶、Wanda
- 🚀 **全局优化**：基于性价比的分数背包剪枝
- 🔧 **层级控制**：非均衡剪枝策略，保护重要层
- 💪 **微调恢复**：支持全参数和LoRA微调

## 🧠 核心设计：分组剪枝逻辑

本工具采用**结构化分组剪枝**策略，确保剪枝后模型的维度一致性和语义完整性。

### 1️⃣ Attention 分组（GQA-Aware）

在 Grouped Query Attention (GQA) 架构中，将相关的 Q/K/V/O heads 作为一个整体进行剪枝：

```
┌─────────────────────────────────────────────────┐
│          第 i 个 GQA 剪枝组                      │
├─────────────────────────────────────────────────┤
│  • 1 个 KV head (包含 K head + V head)          │
│  • 对应的 4 个 Q heads                          │
│  • 对应的 4 个 O heads                          │
└─────────────────────────────────────────────────┘

保持 4:1 的 Q:KV 比例不变
```

**实现细节**：
- `q_proj`: 剪枝输出通道 `[4×head_dim]`
- `k_proj`: 剪枝输出通道 `[head_dim]`
- `v_proj`: 剪枝输出通道 `[head_dim]`
- `o_proj`: 剪枝输入通道 `[4×head_dim]`（对应 Q heads concat 的结果）

**为什么这样设计**？
- 保持 GQA 的 4:1 结构约束
- 确保 Q heads 和 KV heads 的语义对应关系
- 避免维度不匹配导致的推理错误

### 2️⃣ MLP 分组（通道级）

在 SwiGLU MLP 结构中，将 gate/up/down 的对应通道作为一组剪枝：

```
┌─────────────────────────────────────────────────┐
│          第 i 个 MLP 剪枝组                      │
├─────────────────────────────────────────────────┤
│  gate_proj[i, :]  hidden_dim → 第i个输出        │
│  up_proj[i, :]    hidden_dim → 第i个输出        │
│  down_proj[:, i]  第i个输入 → hidden_dim        │
└─────────────────────────────────────────────────┘

前向传播: x → SwiGLU(gate[i], up[i]) → down[:, i] → out
```

**实现细节**：
- `gate_proj.weight[i, :]`: 保留/删除第 i 行（输出通道）
- `up_proj.weight[i, :]`: 保留/删除第 i 行（输出通道）
- `down_proj.weight[:, i]`: 保留/删除第 i 列（输入通道）

**为什么这样设计**？
- 确保 `gate` 和 `up` 的对应通道一起参与 SwiGLU 激活
- 保证 `down` 的输入维度与前面的输出对齐
- 维持完整的端到端计算路径

### 📐 数学形式

**Attention 组重要性**：
```
I_attention(group_i) = I(Q_heads[4i:4i+4]) + I(K_head[i]) + I(V_head[i]) + I(O_heads[4i:4i+4])
```

**MLP 组重要性**：
```
I_mlp(channel_i) = I(gate[i, :]) + I(up[i, :]) + I(down[:, i])
```

**全局评分（分数背包）**：
```
Score(group) = Importance(group) / Cost(group)
剪枝策略: 选择 Score 最低的 groups 进行剪枝
```

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
python run_global_pruning.py \
    --base_model /path/to/llama-3-8b \
    --save_ckpt_log_name my_experiment \
    --pruning_ratio 0.25 \
    --importance_method taylor \
    --num_samples 128 \
    --test_after_prune \
    --output_model pruned_model.bin
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
python run_global_pruning.py \
    --base_model /path/to/llama-3-8b \
    --pruning_ratio 0.25 \
    --finetune \
    --finetune_method lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --finetune_samples 1000 \
    --finetune_lr 1e-4 \
    --test_after_prune \
    --output_model finetuned_model.bin
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
python run_global_pruning.py \
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

## 🔬 技术实现细节

### 分组剪枝代码实现

#### Attention 分组剪枝 (`core/methods/gqa_aware.py`)

```python
def prune_attention_by_gqa_groups(layer, keep_kv_indices, head_dim=128, gqa_ratio=4):
    """
    根据保留的 KV head 索引剪枝整个 GQA 组

    Args:
        keep_kv_indices: 要保留的 KV head 索引列表 [0, 2, 5, ...]
    """
    # 1. 计算对应的 Q head 索引
    keep_q_indices = []
    for kv_idx in keep_kv_indices:
        q_start = kv_idx * gqa_ratio  # 例如 KV[1] 对应 Q[4:8]
        keep_q_indices.extend(range(q_start, q_start + gqa_ratio))

    # 2. 转换为通道索引（head → channel）
    keep_q_channels = [range(q*head_dim, (q+1)*head_dim) for q in keep_q_indices]
    keep_kv_channels = [range(kv*head_dim, (kv+1)*head_dim) for kv in keep_kv_indices]

    # 3. 剪枝权重矩阵
    layer.self_attn.q_proj.weight = layer.self_attn.q_proj.weight[keep_q_channels, :]
    layer.self_attn.k_proj.weight = layer.self_attn.k_proj.weight[keep_kv_channels, :]
    layer.self_attn.v_proj.weight = layer.self_attn.v_proj.weight[keep_kv_channels, :]
    layer.self_attn.o_proj.weight = layer.self_attn.o_proj.weight[:, keep_q_channels]

    # 4. 更新配置
    layer.self_attn.num_heads = len(keep_q_indices)
    layer.self_attn.num_key_value_heads = len(keep_kv_indices)
```

#### MLP 分组剪枝 (`run_global_pruning.py`)

```python
def prune_mlp_by_channels(layer, keep_channel_indices):
    """
    根据保留的通道索引剪枝 MLP

    Args:
        keep_channel_indices: 要保留的中间层通道索引 [0, 5, 10, ...]
    """
    # 1. 剪枝 gate_proj 和 up_proj 的输出通道（行）
    layer.mlp.gate_proj.weight = layer.mlp.gate_proj.weight[keep_channel_indices, :]
    layer.mlp.up_proj.weight = layer.mlp.up_proj.weight[keep_channel_indices, :]

    # 2. 剪枝 down_proj 的输入通道（列）
    layer.mlp.down_proj.weight = layer.mlp.down_proj.weight[:, keep_channel_indices]

    # 3. 更新配置
    new_intermediate_size = len(keep_channel_indices)
    layer.mlp.gate_proj.out_features = new_intermediate_size
    layer.mlp.up_proj.out_features = new_intermediate_size
    layer.mlp.down_proj.in_features = new_intermediate_size
```

### 重要性计算方法

#### Taylor Expansion (一阶)

```python
# 对于每个权重参数
importance = |weight × gradient|

# Attention 组: 累加所有相关的 projection 层
I_group = |W_q × ∇W_q| + |W_k × ∇W_k| + |W_v × ∇W_v| + |W_o × ∇W_o|

# MLP 组: 累加三个 projection 层
I_channel = |W_gate[i] × ∇W_gate[i]| + |W_up[i] × ∇W_up[i]| + |W_down[:,i] × ∇W_down[:,i]|
```

#### Taylor Expansion (二阶)

```python
# 增加 Hessian 对角线项
importance = |weight × gradient| + 0.5 × |weight² × hessian_diag|

# Hessian 对角线近似: ∇²L ≈ (∇L)²
```

#### Wanda (Weight × Activation)

```python
# 使用激活值代替梯度
importance = |weight × activation|

# 无需反向传播，计算更快
```

### 全局剪枝算法（分数背包）

```python
# 1. 构建全局分析表
for layer in model.layers:
    for group in [attention_groups, mlp_groups]:
        importance = compute_importance(group)
        cost = count_parameters(group)
        score = importance / cost
        table.append((layer_id, group_id, score, cost))

# 2. 按 score 排序（从小到大）
table.sort(key=lambda x: x['score'])

# 3. 累加成本，直到达到剪枝目标
pruned_params = 0
for group in table:
    if pruned_params + group.cost <= target_pruned_params:
        prune_group(group)
        pruned_params += group.cost
    else:
        break  # 达到目标，停止剪枝
```

### 关键超参数对照表

| 参数 | Attention 分组 | MLP 分组 |
|------|---------------|---------|
| **组的大小** | 6个矩阵块 (Q/K/V各1个 + O各1个) | 3个向量 (gate/up/down各1个) |
| **head_dim** | 128 (LLaMA-3) | N/A |
| **gqa_ratio** | 4:1 (Q:KV) | N/A |
| **num_groups** | num_kv_heads (通常8) | intermediate_size (通常14336) |
| **cost/group** | ~1.6M 参数 | ~12K 参数 |

### 维度变化示例

**剪枝前** (LLaMA-3-8B):
```
Attention:
- num_q_heads = 32, num_kv_heads = 8, head_dim = 128
- q_proj: [4096, 4096]  (32 * 128 = 4096)
- k_proj: [1024, 4096]  (8 * 128 = 1024)
- v_proj: [1024, 4096]
- o_proj: [4096, 4096]

MLP:
- gate_proj: [14336, 4096]
- up_proj:   [14336, 4096]
- down_proj: [4096, 14336]
```

**剪枝后** (假设剪掉 50% Attention 和 30% MLP):
```
Attention:
- num_q_heads = 16, num_kv_heads = 4, head_dim = 128
- q_proj: [2048, 4096]  (16 * 128 = 2048)
- k_proj: [512, 4096]   (4 * 128 = 512)
- v_proj: [512, 4096]
- o_proj: [4096, 2048]  ← 注意这里是输入通道变化

MLP:
- gate_proj: [10035, 4096]  (14336 * 0.7 ≈ 10035)
- up_proj:   [10035, 4096]
- down_proj: [4096, 10035]  ← 注意这里是输入通道变化
```

---

**核心文件**：
- `run_global_pruning.py` - 全局剪枝主脚本（推荐使用）
- `layer_pruning.py` - 层级剪枝主脚本
- `core/methods/global_pruning.py` - 全局剪枝算法实现
- `core/methods/gqa_aware.py` - GQA 感知的 Attention 分组剪枝
- `core/importance/layer_analyzer.py` - 层重要性分析
- `core/trainer/finetuner.py` - LoRA/全参数微调
- `evaluation/metrics/ppl.py` - 困惑度评估

**相关论文**：
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Wanda: A Simple and Effective Pruning Approach](https://arxiv.org/abs/2306.11695)
- [The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning](https://arxiv.org/abs/2203.07259)