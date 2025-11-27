# SliceGPT 模型转换说明

## 问题背景

SliceGPT 模型使用了特殊的剪枝技术，包括：
- **旋转矩阵**（rotation matrices）：用于变换表示空间
- **融合的 LayerNorm**：将 LayerNorm 融合到其他层中
- **结构化切片**：通过切片操作实现剪枝

这些特殊结构使得 SliceGPT 模型无法直接转换为标准的 Llama 模型结构。

## 为什么不能转换为标准 Llama？

尝试将 SliceGPT 模型转换为标准 LlamaForCausalLM 会遇到以下问题：

### 1. 缺失的 LayerNorm 层
```
Missing keys:
- model.layers.0.input_layernorm.weight
- model.layers.0.post_attention_layernorm.weight
```
SliceGPT 将这些层融合到了其他模块中。

### 2. 额外的旋转矩阵
```
Unexpected keys:
- model.layers.0.mlp_shortcut_Q
- model.layers.0.attn_shortcut_Q
```
这些是 SliceGPT 特有的旋转矩阵，标准 Llama 没有。

### 3. 维度不匹配
```
Size mismatch:
- SliceGPT: torch.Size([128256, 3272])
- Standard: torch.Size([128256, 4096])
```
权重维度已改变，但 config 可能未更新。

### 4. 结构完整性
移除旋转矩阵或尝试重建 LayerNorm 会破坏模型的前向传播逻辑。

## 解决方案：直接保存

`convert_slicegpt_model.py` 采用直接保存的策略：

```python
# 1. 在 slicegpt 环境中使用官方加载器加载模型
model_adapter, tokenizer = slicegpt_hf_utils.load_sliced_model(
    base_model, model_dir, sparsity=0.2
)

# 2. 直接保存加载后的模型（不做结构转换）
save_dict = {
    'model': model_adapter.model,  # 保持原始结构
    'tokenizer': tokenizer,
    'method': 'SliceGPT',
    'pruning_ratio': sparsity,
    'config': {...}
}

torch.save(save_dict, output_path)
```

## 优点

✅ **保持完整性**：保留所有 SliceGPT 组件，确保模型可用
✅ **简单可靠**：不涉及复杂的结构转换，不会引入错误
✅ **跨环境使用**：生成的 .bin 文件可在 base 环境中使用 `torch.load()` 直接加载
✅ **兼容评估框架**：与其他剪枝方法的 .bin 格式保持一致

## 权衡

⚠️ **文件大小**：可能比其他剪枝方法略大（包含额外的旋转矩阵）
⚠️ **非标准结构**：不是标准 Llama 结构，但这是 SliceGPT 的固有特性
⚠️ **依赖 SliceGPT 加载**：转换时需要 slicegpt 环境，但使用时不需要

## 使用流程

### 前置要求

转换脚本需要 `dill` 包来序列化 SliceGPT 的动态类：

```bash
conda activate slicegpt
pip install dill
```

### 步骤 1：转换模型（在 slicegpt 环境）

```bash
conda activate slicegpt

python evaluation/convert_slicegpt_model.py \
    --slicegpt_model results/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
    --base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --output results/SliceGPT_2000/pruned_model.bin
```

**注意**：脚本会自动使用 `dill` 序列化（如果已安装）。如果未安装 dill，会回退到标准 pickle，但可能会失败。

### 步骤 2：评估模型（在 base 环境）

base 环境也需要安装 `dill` 来加载转换后的模型：

```bash
conda activate base
pip install dill

python evaluation/run_evaluation.py \
    --model_path results/SliceGPT_2000/pruned_model.bin \
    --metrics all \
    --output results/SliceGPT_2000/evaluation/evaluation_results.json
```

## 技术细节

### 模型结构对比

| 组件 | 标准 Llama | SliceGPT |
|------|-----------|----------|
| Embedding | ✓ | ✓ (维度已剪枝) |
| LayerNorm | ✓ 独立层 | ✗ 已融合 |
| Attention | ✓ | ✓ (维度已剪枝) |
| MLP | ✓ | ✓ (维度已剪枝) |
| 旋转矩阵 | ✗ | ✓ `attn_shortcut_Q`, `mlp_shortcut_Q` |

### 保存格式

```python
{
    'model': <SliceGPT model object>,  # 完整的 PyTorch nn.Module
    'tokenizer': <tokenizer>,
    'method': 'SliceGPT',
    'pruning_ratio': 0.2,
    'actual_ratio': 0.2,
    'config': {
        'base_model': '/path/to/base/model',
        'slicegpt_model': '/path/to/slicegpt.pt',
        'sparsity': 0.2,
        'round_interval': 8,
        'note': 'Model contains SliceGPT-specific components'
    }
}
```

## 常见问题

**Q: 为什么需要 dill？**

A: SliceGPT 在加载时动态创建模型类（`UninitializedLlamaForCausalLM`），这是一个局部类，无法被 Python 标准的 pickle 序列化。`dill` 是 pickle 的扩展，支持序列化局部定义的类和更复杂的 Python 对象。

**Q: 为什么不像 Wanda/Magnitude 那样真正"转换"模型？**

A: Wanda 和 Magnitude 是简单的权重剪枝（置零或删除），可以转换为标准结构。SliceGPT 使用旋转和融合，改变了模型的计算图，无法还原为标准结构。

**Q: .bin 文件能在没有 slicegpt 包的环境中使用吗？**

A: 可以！生成的 .bin 文件使用 dill 序列化，可以在任何安装了 `dill` 的 Python 环境中加载。加载后的模型对象是标准的 `nn.Module`，不需要 slicegpt 包来执行推理。

**Q: 如何验证转换后的模型是否正确？**

A: 可以使用 `test_model_info.py` 验证参数数量和结构：

```bash
# 在 base 环境中
python evaluation/test_model_info.py results/SliceGPT_2000/pruned_model.bin
```

## 总结

这个转换脚本采用了**最简单和最可靠**的策略：不做结构转换，直接保存 SliceGPT 加载的模型。这确保了模型的完整性和可用性，同时实现了跨环境使用的目标。
