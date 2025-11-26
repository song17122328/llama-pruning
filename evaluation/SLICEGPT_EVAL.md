# SliceGPT 模型评估指南

## 概述

本项目现已支持评估 SliceGPT 官方实现的剪枝模型（`.pt` 格式）。

## 前提条件

确保已安装 SliceGPT：

```bash
git clone https://github.com/microsoft/TransformerCompression
cd TransformerCompression
pip install -e .
```

## 使用方法

### 1. 自动推断（推荐）

如果你的 SliceGPT 模型文件遵循标准命名格式 `<model_name>_<sparsity>.pt`（例如 `Llama-3-8B-Instruct_0.2.pt`），可以直接使用：

```bash
python evaluation/run_evaluation.py \
    --model_path results/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
    --slicegpt_base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics ppl,zeroshot,memory,speed \
    --output results/SliceGPT_2000/evaluation/evaluation_results.json
```

自动推断逻辑：
- **Sparsity**: 从文件名提取（`_0.2.pt` → `0.2`）
- **Base Model**:
  - 如果目录下有 `config.json`，使用该目录作为本地模型
  - 否则，使用文件名主体部分从 HuggingFace 下载（`Llama-3-8B-Instruct_0.2` → `Llama-3-8B-Instruct`）

### 2. 手动指定参数

如果文件名不标准，或需要指定特定的基础模型：

```bash
python evaluation/run_evaluation.py \
    --model_path path/to/sliced_model.pt \
    --slicegpt_base_model /newdata/LLMs/Llama-3-8B-Instruct \
    --slicegpt_sparsity 0.2 \
    --metrics all \
    --output results/slicegpt_custom.json
```

### 3. 只评估特定指标

```bash
# 只评估 PPL
python evaluation/run_evaluation.py \
    --model_path outputs/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
    --metrics ppl \
    --output results/slicegpt_ppl.json

# 评估 PPL + Zero-shot
python evaluation/run_evaluation.py \
    --model_path outputs/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
    --metrics ppl,zeroshot \
    --zeroshot_tasks boolq,piqa,hellaswag \
    --output results/slicegpt_ppl_zeroshot.json
```

## 支持的评估指标

所有标准评估指标都支持 SliceGPT 模型：

- **ppl**: 困惑度（wikitext2, ptb）
- **zeroshot**: Zero-shot 准确率
- **speed**: 推理速度
- **memory**: 显存占用
- **efficiency**: 综合效率指标

## 文件结构要求

SliceGPT 模型目录应包含：

```
outputs/SliceGPT_2000/
├── Llama-3-8B-Instruct_0.2.pt      # 模型权重（必需）
├── Llama-3-8B-Instruct_0.2.json    # 切片配置（必需）
├── config.json                      # 模型配置（如果是本地模型）
├── tokenizer_config.json           # Tokenizer 配置（如果是本地模型）
└── ...                             # 其他 tokenizer 文件
```

## 加载流程

1. **检测 `.pt` 文件** → 调用 `load_slicegpt_model()`
2. **推断参数** → 从文件名和目录结构提取 `sparsity` 和 `base_model`
3. **导入 SliceGPT** → 使用官方 `slicegpt.hf_utils.load_sliced_model()`
4. **特殊处理** → 应用 `replace_layers`, `fuse_modules`, `slice_rotated_model`
5. **返回模型** → 提取 `model_adapter.model` 进行评估

## 常见问题

### Q1: 提示找不到 slicegpt 模块

**A**: 确保已安装 SliceGPT：

```bash
pip install git+https://github.com/microsoft/TransformerCompression.git
# 或
git clone https://github.com/microsoft/TransformerCompression
cd TransformerCompression
pip install -e .
```

### Q2: 无法推断 sparsity

**A**: 手动指定 `--slicegpt_sparsity` 参数：

```bash
python evaluation/run_evaluation.py \
    --model_path my_model.pt \
    --slicegpt_sparsity 0.2 \
    --output results.json
```

### Q3: 找不到 base model

**A**: 两种解决方案：

1. 将 tokenizer 和 config 文件复制到 SliceGPT 模型目录
2. 手动指定 `--slicegpt_base_model`：

```bash
python evaluation/run_evaluation.py \
    --model_path sliced.pt \
    --slicegpt_base_model /path/to/original/model \
    --output results.json
```

### Q4: 为什么 SliceGPT 模型比其他方法的文件大？

**A**: SliceGPT 使用了不同的剪枝方式：
- **我们的方法**（Wanda, Magnitude, SlimGPT）：真正删除权重矩阵的行/列
- **SliceGPT**: 通过旋转和切片保持部分结构，但 state_dict 可能包含额外信息

这是正常的，不影响评估结果的准确性。

## 示例：完整评估流程

```bash
# 1. 评估 SliceGPT 20% 剪枝模型
python evaluation/run_evaluation.py \
    --model_path outputs/SliceGPT_2000/Llama-3-8B-Instruct_0.2.pt \
    --metrics all \
    --output results/SliceGPT_2000/evaluation.json

# 2. 评估原始模型（用于对比）
python evaluation/run_evaluation.py \
    --model_path /newdata/LLMs/Llama-3-8B-Instruct \
    --metrics all \
    --output results/original/evaluation.json

# 3. 评估我们的方法（Wanda）
python evaluation/run_evaluation.py \
    --model_path results/Wanda_2000/pruned_model.bin \
    --metrics all \
    --output results/Wanda_2000/evaluation.json

# 4. 生成对比表格
python evaluation/run_evaluation.py \
    --compare \
    --model_paths results/original/evaluation.json,results/SliceGPT_2000/evaluation.json,results/Wanda_2000/evaluation.json \
    --output comparison.md
```

## 技术细节

### 与其他模型格式的区别

| 特性 | SliceGPT (.pt) | 我们的方法 (.bin) | HF 模型 |
|------|----------------|-------------------|---------|
| 格式 | `torch.save(model.state_dict())` | `torch.save({'model': model, ...})` | HF safetensors |
| 加载器 | `slicegpt.hf_utils.load_sliced_model()` | `torch.load()` | `AutoModelForCausalLM` |
| 特殊处理 | 需要 replace/fuse/slice | 包含 IdentityDecoderLayer | 标准加载 |
| 配置文件 | `.json` (slicing config) | 嵌入在 .bin 中 | HF config.json |

### 内部实现

`evaluation/utils/model_loader.py` 中的 `load_slicegpt_model()` 函数处理：

```python
def load_slicegpt_model(model_path, device, torch_dtype, base_model, sparsity):
    # 1. 从文件名推断参数
    if sparsity is None:
        sparsity = float(Path(model_path).stem.split('_')[-1])

    if base_model is None:
        model_name = Path(model_path).stem.rsplit('_', 1)[0]
        if (Path(model_path).parent / "config.json").exists():
            base_model = str(Path(model_path).parent)
        else:
            base_model = model_name

    # 2. 设置 SliceGPT config
    slicegpt_config.device = torch.device(device)
    slicegpt_config.dtype = torch_dtype

    # 3. 使用官方加载器
    model_adapter, tokenizer = slicegpt_hf_utils.load_sliced_model(
        base_model, str(Path(model_path).parent),
        sparsity=sparsity, round_interval=8
    )

    # 4. 返回模型
    return model_adapter.model, tokenizer
```

## 更新日志

### 2024-01-XX
- 初始支持 SliceGPT 模型评估
- 添加自动参数推断
- 集成到统一评估框架
