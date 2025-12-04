# LLaMA Pruning Toolkit

高效的 LLaMA / Qwen / Mistral 模型结构化剪枝工具，基于全局性价比优化的剪枝策略。

## ✨ 特性

- 🎯 **结构化分组剪枝**：GQA-aware 端到端剪枝策略
- 🌐 **多模型支持**：LLaMA-3-8B、Qwen2.5-7B、Mistral-7B-v0.3
- 🔬 **多种重要性度量**：Taylor 一阶/二阶、Magnitude
- 🚀 **全局优化**：基于性价比的分数背包剪枝算法
- 🔧 **自动配置检测**：自动识别不同模型的 GQA 架构（4:1 / 7:1）
- 💪 **微调恢复**：支持 LoRA 微调恢复性能

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

### 全局剪枝（Global Structural Pruning）

基于性价比得分（Importance/Cost）全局选择最优剪枝策略。

```bash
    
# Llama-3-8B-Instruct
python run_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --output_name Llama-3-8B-Instruct/Taylor_only_20 \
    --pruning_ratio 0.2 \
    --temperature 0.0 


# LLaMA-3-8B
python run_global_pruning.py \
    --base_model /newdata/LLMs/Llama-3-8B \
    --output_name LLaMA-3-8B/Taylor_only_20 \
    --pruning_ratio 0.2 \
    --temperature 0.0 


# Qwen2.5-7B（自动检测 GQA 7:1）
python run_global_pruning.py \
    --base_model /newdata/LLMs/Qwen2.5-7B \
    --output_name Qwen2.5-7B/Taylor_only_20 \
    --pruning_ratio 0.2 \
    --temperature 1.0 \


# Mistral-7B-v0.3（自动检测 GQA 4:1）
python run_global_pruning.py \
    --base_model /newdata/LLMs/Mistral-7B-v0.3 \
    --output_name Mistral-7B-v0.3/blockwise_20_c4 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --temperature 1.0 --tau 0 --dataset c4 

python run_global_pruning.py \
    --base_model /newdata/LLMs/Mistral-7B-v0.3 \
    --output_name Mistral-7B-v0.3/layerwise_20_c4_loss \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --temperature 1.0 --tau inf --dataset c4 

python run_global_pruning.py \
    --base_model /newdata/LLMs/Mistral-7B-v0.3 \
    --output_name Mistral-7B-v0.3/Taylor_2nd_20 \
    --pruning_ratio 0.2 \
    --importance_method taylor_2nd \
    --temperature 0 --dataset c4 


python run_global_pruning.py \
    --base_model /newdata/LLMs/Mistral-7B-v0.3 \
    --output_name Mistral-7B-v0.3/Taylor_only_20_c4 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --dataset c4 \
    --temperature 0.0 
```

**核心参数**：
- `--pruning_ratio`: 目标剪枝率（0.2 = 20%）
- `--importance_method`: taylor（一阶，默认）/ taylor_2nd（二阶）/ wanda / magnitude
- `--dataset`: 校准数据集（wikitext2 / ptb / c4，默认 wikitext2）
- `--temperature`: H-GSP 温度参数 T（默认 1.0）
  - `T=0`: 纯 Taylor 模式（跳过层/块重要性分析，最快）
  - `T=1`: 推荐模式（平衡基础方法与层级先验）
  - `T>1`: 激进模式（强化首尾保护）
- `--tau`: H-GSP 门控阈值 τ（默认 None 自动计算）
  - `tau=0`: 纯 Block-wise 模式（只使用块级重要性）
  - `tau=None`: 自动模式（计算25分位数，推荐）
  - `tau=inf`: 纯 Layer-wise 模式（只使用层级重要性）
- `--epsilon`: H-GSP 坍缩阈值 ε（默认 0）
- `--freeze_first_n_layers`: 冻结前N层不剪枝（默认 0）
- `--freeze_last_n_layers`: 冻结后N层不剪枝（默认 0）

**典型结果**（LLaMA-3-8B）：
- 原始模型：WikiText-2 PPL ~12.3
- 20% 剪枝：PPL ~58.9
- 30% 剪枝：PPL ~83.8
- + LoRA 微调：PPL ~18.5

## 🔧 微调恢复

剪枝后使用 LoRA 微调恢复性能：

```bash
# 剪枝 + 微调（集成）
python run_global_pruning.py \
    --base_model Qwen/Qwen2.5-7B \
    --output_name Qwen2.5-7B/prune_20_finetune \
    --pruning_ratio 0.2 \
    --finetune \
    --finetune_data_path yahma/alpaca-cleaned \
    --finetune_epochs 3 \
    --finetune_lr 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --device cuda:0

# 或使用独立微调脚本
python finetune_lora.py \
    --pruned_model results/Qwen2.5-7B/prune_20/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir results/Qwen2.5-7B/prune_20_finetuned \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --lora_r 8 \
    --lora_alpha 16 \
    --device cuda:0
```

**微调参数**：
- `--lora_r`: LoRA 秩（推荐 8-16）
- `--lora_alpha`: 缩放系数（通常 = 2×r）
- `--finetune_lr`: 学习率（推荐 3e-4）
- `--finetune_epochs`: 微调轮数（推荐 3-5）

## 🎯 H-GSP 方法详解

### 核心思想

H-GSP (Hierarchical Global Structural Pruning) 是一种分层次的全局结构化剪枝方法，结合了**全局 Taylor 重要性**和**层级/块级先验知识**。

### 评分公式

```
基础评分: S_base = Importance / Cost

混合加权: S_final = S_base × M

其中: M = B^T
      B = ln(1 + importance_prior)
      T = temperature (温度参数)
```

### 参数详解

#### 1. Temperature (温度 T)

控制层级先验的影响强度：

- **T=0** (纯 Taylor 模式)
  ```bash
  python run_global_pruning.py \
    --base_model /path/to/model \
    --pruning_ratio 0.2 \
    --temperature 0.0  # 最快，跳过层/块重要性分析
  ```
  - ✅ 只使用全局 Taylor 重要性
  - ✅ 最快（跳过 Step 3.5-3.6）
  - ✅ 适用于所有模型（无兼容性问题）
  - ⚠️ 不考虑层级结构

- **T=1** (推荐模式)
  ```bash
  python run_global_pruning.py \
    --base_model /path/to/model \
    --pruning_ratio 0.2 \
    --temperature 1.0  # 推荐，平衡性能
  ```
  - ✅ 平衡基础方法与层级先验
  - ✅ 自动保护重要层的首尾
  - ✅ 使用相似度方法（ShortGPT）

- **T>1** (激进模式)
  - 强化首尾保护，更激进地剪枝中间层

#### 2. Tau (门控阈值 τ)

控制 Layer-wise 和 Block-wise 模式的切换：

- **tau=0** (纯 Block-wise)
  ```bash
  python run_global_pruning.py \
    --base_model /path/to/model \
    --pruning_ratio 0.2 \
    --temperature 1.0 \
    --tau 0  # 强制使用块级重要性
  ```
  - 所有层都使用 Attention/MLP 块级重要性
  - 精细化剪枝策略

- **tau=None** (自动模式，推荐)
  ```bash
  python run_global_pruning.py \
    --base_model /path/to/model \
    --pruning_ratio 0.2 \
    --temperature 1.0
    # tau 默认 None，自动计算
  ```
  - 自动计算 τ = 25分位数(层重要性)
  - 低于 τ 的层 → Layer-Dominant 模式
  - 高于 τ 的层 → Block-Dominant 模式

- **tau=inf** (纯 Layer-wise)
  ```bash
  python run_global_pruning.py \
    --base_model /path/to/model \
    --pruning_ratio 0.2 \
    --temperature 1.0 \
    --tau inf  # 强制使用层级重要性
  ```
  - 所有层都使用层级重要性
  - 鼓励整层移除

#### 3. 层冻结参数

保护模型的首尾层不被剪枝：

```bash
python run_global_pruning.py \
  --base_model /path/to/model \
  --pruning_ratio 0.2 \
  --freeze_first_n_layers 2  # 冻结前2层
  --freeze_last_n_layers 2   # 冻结后2层
```

### 重要性计算方法

**相似度方法（ShortGPT，默认）**：
- 层重要性 = 1 - cosine_similarity(层输入, 层输出)
- 块重要性 = 1 - cosine_similarity(块输入, 块输出)
- ✅ 对所有模型通用（Qwen、Mistral等）
- ✅ 无需移除层，避免兼容性问题
- ✅ 计算高效

### 使用建议

1. **快速实验**：使用 `--temperature 0.0`（纯 Taylor）
2. **最佳性能**：使用 `--temperature 1.0`（H-GSP，推荐）
3. **保护首尾**：使用 `--freeze_first_n_layers` 和 `--freeze_last_n_layers`

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
results/{output_name}/
├── pruned_model.bin             # 剪枝后模型权重
├── config.json                  # 模型配置
├── pruning_analysis.json        # 剪枝分析报告
├── global_group_table.csv       # 全局分组表
└── logs/
    └── training.log             # 详细日志
```

## 💡 使用建议

### 稀疏度选择

| 稀疏度 | 是否微调 | PPL 退化 | 适用场景 |
|--------|---------|----------|----------|
| 10-20% | 可选 | < 10% | 快速压缩 |
| 20-30% | **推荐** | 10-30% | 平衡性能 |
| 30-50% | **必须** | > 30% | 极限压缩 |

### 重要性度量选择

- **taylor_fo**：Taylor 一阶，平衡精度和速度（推荐）
- **taylor_so**：Taylor 二阶，最高精度，计算较慢
- **magnitude**：权重大小，快速原型验证

### 多模型测试建议

- **LLaMA-3-8B**：基准模型，GQA 4:1
- **Mistral-7B-v0.3**：验证相同 GQA 比例（4:1）的泛化性
- **Qwen2.5-7B**：验证不同 GQA 比例（7:1）的适应性

## 🛠️ 高级用法

### 使用梯度检查点（节省显存）

```bash
python run_global_pruning.py \
    --base_model Qwen/Qwen2.5-7B \
    --output_name Qwen2.5-7B/prune_20 \
    --pruning_ratio 0.2 \
    --use_gradient_checkpointing \
    --device cuda:0
```

### 使用 Taylor 二阶（更精确）

```bash
python run_global_pruning.py \
    --base_model Qwen/Qwen2.5-7B \
    --output_name Qwen2.5-7B/prune_20_taylor2nd \
    --pruning_ratio 0.2 \
    --importance_method taylor_2nd \
    --device cuda:0
```


```bash
python run_global_pruning.py \
    --base_model Qwen/Qwen2.5-7B \
    --output_name Qwen2.5-7B/prune_30 \
    --pruning_ratio 0.3 \
    --temperature 1.5 \
    --epsilon 0.2 \
    --device cuda:0
```

## 🐛 故障排除

**CUDA OOM**：
```bash
--use_gradient_checkpointing     # 启用梯度检查点
--gradient_batch_size 2          # 减小批次大小
```

**PPL 过高**：
- 降低剪枝率（10-20%）
- 使用 Taylor 二阶（`--importance_method taylor_2nd`）
- 启用微调恢复（`--finetune`）
- 调整温度参数（`--temperature 1.5`）

**自动配置检测失败**：
- 检查模型 config 中是否有 `num_key_value_heads` 字段
- 代码会自动回退到 MHA 模式（Q heads = KV heads）

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