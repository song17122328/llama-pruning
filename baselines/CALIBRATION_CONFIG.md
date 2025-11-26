# 校准数据集配置对比

本文档记录所有剪枝方法的校准数据集配置，确保公平对比实验。

## ⚠️ 重要性

**为了确保公平对比，所有需要校准数据的方法必须使用相同的配置：**
- 相同的数据集
- 相同的样本数量
- 相同的序列长度
- 相同的batch_size

## 📊 当前配置总结

| 方法 | 数据集 | 样本数 | seq_len | batch_size | 备注 |
|------|--------|--------|---------|------------|------|
| **H-GSP (Taylor)** | wikitext2 | 128 | 128 | 4 | 内部固定 |
| **H-GSP (Wanda)** | wikitext2 | 128 | 128 | 4 | 内部固定 |
| **Wanda baseline** | wikitext2 | 128 | - | - | 通过 --calibration_samples |
| **SlimGPT** | wikitext2 | 64 | 128 | 1 | ⚠️ 不一致 |
| **ShortGPT** | wikitext2 | 50 | 512 | - | ⚠️ 不一致 |
| **Magnitude** | - | - | - | - | ✓ 无需校准数据 |

## 🔍 详细配置

### 1. H-GSP (我们的方法)

**文件**: `run_global_pruning.py`

**内部固定参数** (不对外暴露):
```python
# Taylor importance 计算 (用于宽度剪枝)
TAYLOR_NUM_SAMPLES = 128          # 梯度计算样本数
TAYLOR_SEQ_LEN = 128              # 序列长度
gradient_batch_size = 4            # 批次大小 (可通过 --gradient_batch_size 修改)

# Layer importance 计算 (用于深度剪枝)
LAYER_IMPORTANCE_NUM_SAMPLES = 50  # 层级重要度样本数
LAYER_IMPORTANCE_SEQ_LEN = 128     # 序列长度

# Block importance 计算 (用于混合剪枝)
BLOCK_IMPORTANCE_NUM_SAMPLES = 50  # 块级重要度样本数
BLOCK_IMPORTANCE_SEQ_LEN = 128     # 序列长度
```

**数据集**: `wikitext2` (默认，可通过 `--dataset` 修改)

**关键代码位置**:
- run_global_pruning.py:586-592 (参数定义)
- run_global_pruning.py:594-679 (Taylor 计算)
- run_global_pruning.py:685-730 (Wanda 激活收集)

**使用方法**:
```bash
python run_global_pruning.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --dataset wikitext2 \
    --gradient_batch_size 4
```

---

### 2. Wanda Baseline

**文件**: `baselines/run_wanda.py`

**配置参数**:
```python
--calibration_samples 128    # 默认: 128
--dataset wikitext2          # 默认: wikitext2
```

**内部使用**: 最终调用 `run_global_pruning.py`，使用相同的内部固定参数

**使用方法**:
```bash
python baselines/run_wanda.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --calibration_samples 128 \
    --dataset wikitext2
```

---

### 3. SlimGPT Baseline

**文件**: `baselines/run_slimgpt.py`

**配置参数**:
```python
--dataset wikitext2          # 默认: wikitext2
--num_samples 64             # 默认: 64 ⚠️
--seq_len 128                # 默认: 128
--max_samples 128            # Hessian 最大token数 (128k)
batch_size = 1               # 固定在代码中
```

**⚠️ 问题**:
- `num_samples=64` 与 H-GSP 的 128 不一致
- `batch_size=1` 固定，无法修改

**关键代码位置**:
- baselines/run_slimgpt.py:263-268 (参数定义)
- baselines/run_slimgpt.py:62 (dataloader 创建，batch_size=1)

**使用方法**:
```bash
python baselines/run_slimgpt.py \
    --base_model /path/to/llama \
    --pruning_ratio 0.2 \
    --num_samples 128 \
    --seq_len 128 \
    --dataset wikitext2
```

---

### 4. ShortGPT Baseline

**文件**: `baselines/run_shortgpt.py`

**配置参数**:
```python
--dataset wikitext2          # 默认: wikitext2
--num_samples 50             # 默认: 50 ⚠️
--seq_len 512                # 默认: 512 ⚠️
--stride 256                 # 滑动窗口步长
```

**⚠️ 问题**:
- `num_samples=50` 与 H-GSP 的 128 不一致
- `seq_len=512` 与 H-GSP 的 128 不一致

**关键代码位置**:
- baselines/run_shortgpt.py:70-77 (参数定义)

**使用方法**:
```bash
python baselines/run_shortgpt.py \
    --base_model /path/to/llama \
    --n_remove_layers 6 \
    --num_samples 128 \
    --seq_len 128 \
    --dataset wikitext2
```

---

### 5. Magnitude Baseline

**文件**: `baselines/run_magnitude.py`

**配置**: ✓ **无需校准数据**

只使用权重绝对值，不依赖数据集。

---

## ✅ 推荐的统一配置

为确保公平对比，建议所有方法使用以下统一配置：

```bash
# 统一的校准配置
DATASET=wikitext2
NUM_SAMPLES=128
SEQ_LEN=128
BATCH_SIZE=4  # SlimGPT 除外（固定为1）
```

### 完整实验脚本 (统一配置)

```bash
MODEL=/path/to/llama
DATASET=wikitext2
SAMPLES=128
SEQ_LEN=128

# 1. H-GSP (Taylor) - 默认已使用 128 samples
python run_global_pruning.py \
    --base_model $MODEL \
    --output_name HGSP_Taylor_20 \
    --pruning_ratio 0.2 \
    --importance_method taylor \
    --dataset $DATASET \
    --gradient_batch_size 4

# 2. H-GSP (Wanda) - 默认已使用 128 samples
python run_global_pruning.py \
    --base_model $MODEL \
    --output_name HGSP_Wanda_20 \
    --pruning_ratio 0.2 \
    --importance_method wanda \
    --dataset $DATASET \
    --gradient_batch_size 4

# 3. Wanda baseline - 显式指定 128 samples
python baselines/run_wanda.py \
    --base_model $MODEL \
    --pruning_ratio 0.2 \
    --calibration_samples $SAMPLES \
    --dataset $DATASET

# 4. SlimGPT - 修改为 128 samples (原默认 64)
python baselines/run_slimgpt.py \
    --base_model $MODEL \
    --pruning_ratio 0.2 \
    --num_samples $SAMPLES \
    --seq_len $SEQ_LEN \
    --dataset $DATASET

# 5. ShortGPT - 修改为 128 samples, seq_len 128 (原默认 50, 512)
python baselines/run_shortgpt.py \
    --base_model $MODEL \
    --n_remove_layers 6 \
    --num_samples $SAMPLES \
    --seq_len $SEQ_LEN \
    --dataset $DATASET

# 6. Magnitude - 无需校准数据
python baselines/run_magnitude.py \
    --base_model $MODEL \
    --pruning_ratio 0.2
```

---

## 🔧 需要修改的地方

### ⚠️ SlimGPT

**问题**:
1. 默认 `num_samples=64`，需改为 128
2. `batch_size=1` 硬编码在 `create_dataloader()` 中

**修改建议**:
```python
# baselines/run_slimgpt.py:263
parser.add_argument('--num_samples', type=int, default=128,  # 改为 128
                   help='Hessian 计算样本数（默认: 128）')

# baselines/run_slimgpt.py:62
def create_dataloader(dataset_manager, num_samples, seq_len, batch_size=4):  # 改为 4
```

### ⚠️ ShortGPT

**问题**:
1. 默认 `num_samples=50`，需改为 128
2. 默认 `seq_len=512`，需改为 128（与其他方法一致）

**修改建议**:
```python
# baselines/run_shortgpt.py:73
parser.add_argument('--num_samples', type=int, default=128,  # 改为 128
                   help='BI 计算样本数（默认: 128）')

# baselines/run_shortgpt.py:75
parser.add_argument('--seq_len', type=int, default=128,  # 改为 128
                   help='序列长度（默认: 128）')
```

**注意**: ShortGPT 使用较长的 `seq_len=512` 可能有其理论依据（BI 计算需要更多上下文），如果改为 128 可能影响效果。建议：
- **选项1**: 保持 512，但在论文中说明差异
- **选项2**: 改为 128，确保完全公平对比
- **选项3**: 两种配置都测试，分别报告结果

---

## 📝 数据集说明

### WikiText-2

- **类型**: 英文维基百科文本
- **用途**: LLM 校准和评估的标准数据集
- **优势**: 干净、结构化、代表性强

### 其他可选数据集

- **PTB** (Penn Treebank): 较小，新闻文本
- **C4** (Colossal Clean Crawled Corpus): 更大更多样

**建议**: 使用 WikiText-2，与大多数论文一致。

---

## 🎯 检查清单

在运行对比实验前，请确认：

- [ ] 所有方法使用相同的数据集 (wikitext2)
- [ ] 所有方法使用相同的样本数 (128)
- [ ] 所有方法使用相同的序列长度 (128)
- [ ] SlimGPT 和 ShortGPT 已更新默认参数
- [ ] 记录所有实验配置到日志文件
- [ ] 在论文中明确说明校准数据配置

---

## 📚 参考信息

### H-GSP 内部参数来源

**为什么选择 128 samples?**
- 平衡计算成本和准确性
- 与 Wanda 论文一致 (128 samples)
- 足够统计显著性

**为什么选择 seq_len=128?**
- 标准 LLM 评估长度
- 快速迭代测试
- 与大多数剪枝论文一致

**为什么 gradient_batch_size=4?**
- 适配大多数单卡 GPU (24GB)
- 避免 OOM
- 可根据显存调整 (1-8)

### Batch Size 说明

- **Taylor/Wanda**: batch_size=4，分批计算节省显存
- **SlimGPT**: batch_size=1，Hessian 计算逐样本处理
- **ShortGPT**: 无 batch_size，逐文本计算 BI

---

## 🚨 常见问题

### Q1: 为什么不同方法的 batch_size 不同？

**A**: batch_size 只影响计算效率和显存使用，不影响最终结果（假设正确归一化）。关键是样本数和序列长度一致。

### Q2: ShortGPT 能否使用 seq_len=128 而非 512？

**A**: 可以，但可能影响 BI 计算准确性（更短的序列可能无法充分体现层的变换作用）。建议两种配置都测试。

### Q3: SlimGPT 的 max_samples 是什么？

**A**: 限制 Hessian 计算的总 token 数（128k = 128000 tokens），防止内存溢出。不影响校准样本数。

### Q4: 我应该如何报告实验配置？

**A**: 在论文 Methods 或 Appendix 中明确说明：
- 数据集名称和版本
- 样本数和序列长度
- 任何方法特定的参数（如 max_samples）
- 硬件配置（GPU 型号、显存）

---

最后更新: 2025-11-26
