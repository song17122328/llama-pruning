# Hessian 参数命名和二阶泰勒实现说明

## 📋 你提出的问题

你在 `core/methods/global_pruning.py` 中发现：

```python
# 二阶项（如果提供了 Hessian）
if hessian_diag is not None and layer_idx is not None:
    full_name = f'model.layers.{layer_idx}.self_attn.{name}.weight'
    if full_name in hessian_diag:
        # ...
    else:
        print("⚠️ Warning: Hessian info missing for", full_name)
```

**你的疑问：**
1. `full_name` 是什么意思，可能不存在？
2. 为什么只有 `self_attn`，没有 `mlp`？

---

## ✅ 问题解答

### 1. `full_name` 的含义

**`full_name` 是 PyTorch 模型中参数的完整路径名称。**

#### 在 `run_global_pruning.py` 中（第 889-892 行）：

```python
# 初始化 Hessian 字典
for name, param in model.named_parameters():
    if param.requires_grad:
        hessian_diag[name] = torch.zeros_like(param.data, device='cpu')
```

这里的 `name` 是 PyTorch 自动生成的完整参数路径，例如：
- `model.layers.0.self_attn.q_proj.weight`
- `model.layers.0.self_attn.k_proj.weight`
- `model.layers.0.mlp.gate_proj.weight`
- `model.layers.0.mlp.up_proj.weight`

#### 在 `core/methods/global_pruning.py` 中构造 `full_name`：

```python
# Attention 层（第 59 行）
full_name = f'model.layers.{layer_idx}.self_attn.{name}.weight'
# 例如：model.layers.0.self_attn.q_proj.weight

# MLP 层（第 222 行）
full_name = f'model.layers.{layer_idx}.mlp.{name}.weight'
# 例如：model.layers.0.mlp.gate_proj.weight
```

**目的：** 通过构造的 `full_name` 从 `hessian_diag` 字典中查找对应参数的 Hessian 对角线值。

---

### 2. MLP 层确实有二阶处理！

**你可能误解了，MLP 层其实也有完整的二阶泰勒实现。**

#### Attention 层处理（`compute_attention_group_importance_taylor`，第 32-98 行）：

```python
def compute_attention_group_importance_taylor(layer, head_dim=128, gqa_ratio=4,
                                             hessian_diag=None, layer_idx=None):
    salience = {}
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        sub_layer = getattr(layer.self_attn, name)
        # 一阶项
        first_order = (sub_layer.weight * sub_layer.weight.grad).abs()

        # 二阶项（如果提供了 Hessian）
        if hessian_diag is not None and layer_idx is not None:
            full_name = f'model.layers.{layer_idx}.self_attn.{name}.weight'  # ← 这里
            if full_name in hessian_diag:
                hess = hessian_diag[full_name].to(sub_layer.weight.device)
                second_order = 0.5 * (sub_layer.weight ** 2 * hess).abs()
                salience[name] = first_order + second_order
            else:
                salience[name] = first_order
    # ...
```

#### MLP 层处理（`compute_mlp_group_importance_taylor`，第 189-246 行）：

```python
def compute_mlp_group_importance_taylor(layer, hessian_diag=None, layer_idx=None):
    # 一阶项
    gate_salience = (layer.mlp.gate_proj.weight * layer.mlp.gate_proj.weight.grad).abs().sum(1)
    up_salience = (layer.mlp.up_proj.weight * layer.mlp.up_proj.weight.grad).abs().sum(1)
    down_salience = (layer.mlp.down_proj.weight * layer.mlp.down_proj.weight.grad).abs().sum(0)

    # 二阶项（如果提供了 Hessian）
    if hessian_diag is not None and layer_idx is not None:
        for name in ['gate_proj', 'up_proj', 'down_proj']:
            full_name = f'model.layers.{layer_idx}.mlp.{name}.weight'  # ← 这里！
            if full_name in hessian_diag:
                sub_layer = getattr(layer.mlp, name)
                hess = hessian_diag[full_name].to(sub_layer.weight.device)
                second_order = 0.5 * (sub_layer.weight ** 2 * hess).abs()

                # 累加二阶项
                if name == 'gate_proj':
                    gate_salience = gate_salience + second_order.sum(1)
                elif name == 'up_proj':
                    up_salience = up_salience + second_order.sum(1)
                else:  # down_proj
                    down_salience = down_salience + second_order.sum(0)

    channel_importance = gate_salience + up_salience + down_salience
    return channel_importance
```

**结论：Attention 和 MLP 都有完整的二阶泰勒实现！**

---

## 🔍 可能的问题：参数命名不匹配

### 潜在问题 1：模型类型不同，参数前缀可能不同

不同模型的参数路径可能有差异：

| 模型 | 参数路径示例 |
|------|-------------|
| **LLaMA-3** | `model.layers.0.self_attn.q_proj.weight` |
| **Qwen2.5** | `model.layers.0.self_attn.q_proj.weight` |
| **某些模型** | `transformer.h.0.attn.q_proj.weight` |
| **GPT-2** | `transformer.h.0.attn.c_attn.weight` |

如果模型结构不是标准的 `model.layers.X.self_attn`，那么构造的 `full_name` 会找不到对应的键。

### 潜在问题 2：Bias 参数

某些模型（如 Qwen2.5）有 bias 参数：
- `model.layers.0.self_attn.q_proj.weight` ✅
- `model.layers.0.self_attn.q_proj.bias` ← Hessian 字典里也会有这个

代码目前只查找 `.weight`，不处理 `.bias`（这是合理的，因为剪枝主要针对 weight）。

---

## 🛠️ 我添加的调试功能

为了帮助你诊断问题，我在代码中添加了详细的调试信息：

### 1. 在 `run_global_pruning.py` 中（第 963-978 行）：

```python
if args.importance_method == 'taylor_2nd':
    logger.log(f"  ✓ Hessian 对角线近似计算完成")
    logger.log(f"  Hessian 字典包含 {len(hessian_diag)} 个参数")

    # 打印一些示例键名，用于调试
    sample_keys = list(hessian_diag.keys())[:10]
    logger.log(f"  示例 Hessian 键名（前10个）：")
    for key in sample_keys:
        logger.log(f"    - {key}")

    # 检查是否包含预期的键名
    layer_0_keys = [k for k in hessian_diag.keys() if 'layers.0.' in k]
    if layer_0_keys:
        logger.log(f"  Layer 0 的参数示例：")
        for key in layer_0_keys[:5]:
            logger.log(f"    - {key}")
```

**作用：** 运行二阶泰勒时，会打印 Hessian 字典中的实际键名，你可以直接看到参数路径格式。

### 2. 在 `core/methods/global_pruning.py` 中（第 68-73 行和 238-243 行）：

```python
# Attention 层
if layer_idx == 0:
    print(f"⚠️ Warning: Hessian key not found: '{full_name}'")
    # 尝试查找相似的键
    similar_keys = [k for k in hessian_diag.keys() if name in k and 'attn' in k][:3]
    if similar_keys:
        print(f"   可能的匹配键: {similar_keys}")

# MLP 层
if layer_idx == 0:
    print(f"⚠️ Warning: Hessian key not found: '{full_name}'")
    similar_keys = [k for k in hessian_diag.keys() if name in k and 'mlp' in k][:3]
    if similar_keys:
        print(f"   可能的匹配键: {similar_keys}")
```

**作用：**
- 如果找不到 Hessian 键，会打印详细的警告
- 只在第一层（layer 0）打印，避免刷屏
- 自动搜索并显示可能的匹配键，帮助你发现命名差异

---

## 🧪 如何验证

### 方法 1：运行二阶泰勒剪枝（推荐）

```bash
python run_global_pruning.py \
  --base_model /newdata/LLMs/Llama-3-8B-Instruct \
  --importance_method taylor_2nd \
  --pruning_ratio 0.2 \
  --output_name test_hessian_naming
```

**查看输出：**
1. 在梯度计算完成后，会打印 Hessian 字典的示例键名
2. 在剪枝过程中，如果有键名不匹配，会打印警告和可能的匹配

**预期情况：**

#### ✅ 正常情况（LLaMA-3）：
```
✓ Hessian 对角线近似计算完成
  Hessian 字典包含 291 个参数
  示例 Hessian 键名（前10个）：
    - model.embed_tokens.weight
    - model.layers.0.self_attn.q_proj.weight
    - model.layers.0.self_attn.k_proj.weight
    - model.layers.0.self_attn.v_proj.weight
    - model.layers.0.self_attn.o_proj.weight
    - model.layers.0.mlp.gate_proj.weight
    - model.layers.0.mlp.up_proj.weight
    - model.layers.0.mlp.down_proj.weight
    - ...
  Layer 0 的参数示例：
    - model.layers.0.self_attn.q_proj.weight
    - model.layers.0.self_attn.k_proj.weight
    - model.layers.0.mlp.gate_proj.weight
```

剪枝过程中**不会出现警告**，说明键名完全匹配。

#### ❌ 异常情况（非标准模型）：
```
⚠️ Warning: Hessian key not found: 'model.layers.0.self_attn.q_proj.weight'
   可能的匹配键: ['transformer.h.0.attn.q_proj.weight', ...]
```

说明模型结构不是标准的 `model.layers` 格式，需要修改代码中的 `full_name` 构造逻辑。

### 方法 2：使用测试脚本（需要 torch 环境）

我已经创建了 `test_hessian_naming.py`：

```bash
python test_hessian_naming.py --model /newdata/LLMs/Llama-3-8B-Instruct
```

这会直接打印模型的参数命名格式。

---

## 📊 参数命名规则总结

### 标准 Transformer 模型（LLaMA、Qwen、Mistral）：

```
模型前缀: model.
  ├─ embed_tokens.weight                              # Embedding
  ├─ layers.{i}.                                      # 第 i 层
  │    ├─ self_attn.                                  # Attention 块
  │    │    ├─ q_proj.weight (可能有 .bias)
  │    │    ├─ k_proj.weight (可能有 .bias)
  │    │    ├─ v_proj.weight (可能有 .bias)
  │    │    └─ o_proj.weight (可能有 .bias)
  │    ├─ mlp.                                        # MLP 块
  │    │    ├─ gate_proj.weight (可能有 .bias)
  │    │    ├─ up_proj.weight (可能有 .bias)
  │    │    └─ down_proj.weight (可能有 .bias)
  │    ├─ input_layernorm.weight
  │    └─ post_attention_layernorm.weight
  └─ norm.weight                                      # 最后的 LayerNorm
```

### 代码中构造的键名：

| 层类型 | 构造格式 | 示例 |
|--------|---------|------|
| Attention | `f'model.layers.{layer_idx}.self_attn.{name}.weight'` | `model.layers.5.self_attn.q_proj.weight` |
| MLP | `f'model.layers.{layer_idx}.mlp.{name}.weight'` | `model.layers.5.mlp.gate_proj.weight` |

**这与 PyTorch 的 `model.named_parameters()` 返回的名称一致！**

---

## 🎯 结论

### 回答你的问题：

1. **`full_name` 是什么意思？**
   - 是 PyTorch 模型中参数的完整路径名称
   - 用于从 `hessian_diag` 字典中查找对应的 Hessian 值
   - 对于标准模型（LLaMA、Qwen），命名应该完全匹配，不会"不存在"

2. **为什么只有 `self_attn`，没有 `mlp`？**
   - **这是误解！** MLP 层也有完整的二阶泰勒实现
   - 在 `compute_mlp_group_importance_taylor` 函数中（第 222 行）
   - 处理 `gate_proj`、`up_proj`、`down_proj` 三个层

### 如果确实出现"键名不存在"的问题：

**可能原因：**
- 使用了非标准结构的模型（不是 `model.layers` 格式）
- 模型参数路径有特殊前缀或后缀

**解决方法：**
1. 运行剪枝，查看调试输出中的实际键名
2. 根据实际键名格式，修改 `full_name` 的构造逻辑
3. 或者提供具体的模型路径，我可以帮你适配

---

## 📝 下一步建议

1. **运行一次二阶泰勒剪枝**，查看调试输出：
   ```bash
   python run_global_pruning.py \
     --base_model /newdata/LLMs/Llama-3-8B-Instruct \
     --importance_method taylor_2nd \
     --pruning_ratio 0.2 \
     --output_name debug_hessian
   ```

2. **检查日志输出**：
   - 查看 "示例 Hessian 键名" 部分
   - 查看是否有 "⚠️ Warning: Hessian key not found" 警告

3. **如果有警告**：
   - 记录实际的键名格式
   - 告诉我，我可以帮你修改代码适配该模型

4. **如果没有警告**：
   - 说明命名完全匹配，二阶泰勒应该正常工作
   - 可以对比一阶和二阶的剪枝结果，验证二阶确实不同

---

## 🔗 相关文件

- `run_global_pruning.py` (第 889-978 行)：Hessian 字典初始化和梯度计算
- `core/methods/global_pruning.py` (第 32-246 行)：Attention 和 MLP 的二阶泰勒实现
- `TAYLOR_2ND_FIX.md`：二阶泰勒梯度计算修复说明
