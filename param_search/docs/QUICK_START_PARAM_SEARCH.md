# 参数搜索快速开始

## 🎯 为什么需要参数搜索？

您发现修改 `TAYLOR_SEQ_LEN` 可以提高 Mistral 的剪枝后 ACC，这是因为：

- ✅ **短序列 (16-64)**: 梯度更稳定，适合 Mistral 等梯度不稳定的模型
- ⚠️ **长序列 (256-512)**: 可能导致梯度爆炸/消失，造成极端剪枝

## 🚀 快速开始（3 步）

### 步骤 1: 修改配置文件

编辑 `configs/mistral_param_search.json`，修改模型路径：

```bash
vim configs/mistral_param_search.json
```

修改第 2 行：

```json
"base_model": "/data/models/Mistral-7B-v0.3",  // ← 改成您的路径
```

### 步骤 2: 运行搜索

```bash
# 快速测试（2 个配置，约 2 小时）
python param_search/search_best_params.py --config configs/quick_param_search.json

# 完整搜索（15 个配置，约 8 小时）
python param_search/search_best_params.py --config configs/mistral_param_search.json
```

### 步骤 3: 查看最佳配置

```bash
cat results/param_search_mistral_20/best_config.json
```

输出示例：

```json
{
  "params": {
    "taylor_seq_len": 64,      // ← 最佳序列长度
    "taylor_num_samples": 256   // ← 最佳样本数
  },
  "metrics": {
    "acc": 0.6234,
    "ppl": 12.45
  }
}
```

## 📊 查看所有结果

```bash
# 方法 1: 使用 Python
python -c "
import pandas as pd
df = pd.read_csv('results/param_search_mistral_20/search_results.csv')
df = df[df['success'] == True].sort_values('acc', ascending=False)
print(df[['taylor_seq_len', 'taylor_num_samples', 'acc', 'ppl']].head(10))
"

# 方法 2: 使用 Excel/LibreOffice 打开
# results/param_search_mistral_20/search_results.csv
```

## 🎨 可视化结果（可选）

创建 `plot_results.py`：

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取结果
df = pd.read_csv('results/param_search_mistral_20/search_results.csv')
df = df[df['success'] == True]

# 创建热力图
pivot = df.pivot_table(
    values='acc',
    index='taylor_seq_len',
    columns='taylor_num_samples',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu')
plt.title('ACC vs. Seq Length & Num Samples')
plt.xlabel('Num Samples')
plt.ylabel('Seq Length')
plt.savefig('param_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ 热力图已保存到 param_heatmap.png")
```

运行：

```bash
python plot_results.py
```

## 🔧 使用最佳配置

找到最佳配置后，使用它运行完整实验：

```bash
# 假设最佳配置是 seq_len=64, num_samples=256
python run_global_pruning.py \
    --base_model /path/to/Mistral-7B-v0.3 \
    --output_name mistral_final_best \
    --pruning_ratio 0.2 \
    --taylor_seq_len 64 \
    --taylor_num_samples 256 \
    --dataset c4 \
    --importance_method taylor \
    --run_evaluation all
```

## 💡 搜索策略建议

### 快速验证（2 小时）

测试您的假设是否正确：

```json
{
  "search_params": {
    "taylor_seq_len": [32, 64, 128],
    "taylor_num_samples": [256]
  }
}
```

### 标准搜索（6-8 小时）

找到较好的配置：

```json
{
  "search_params": {
    "taylor_seq_len": [16, 32, 64, 128, 256],
    "taylor_num_samples": [128, 256, 512]
  }
}
```

### 精细优化（12-16 小时）

在最佳范围附近细化：

```json
{
  "search_params": {
    "taylor_seq_len": [48, 56, 64, 72, 80],
    "taylor_num_samples": [200, 256, 300]
  }
}
```

## ⚙️ 常用命令

```bash
# 1. 快速测试（只运行 2 个实验）
python param_search/search_best_params.py \
    --config configs/quick_param_search.json \
    --max_experiments 2

# 2. 从中断处继续
python param_search/search_best_params.py \
    --config configs/mistral_param_search.json \
    --resume

# 3. 手动测试单个配置
python run_global_pruning.py \
    --base_model /path/to/model \
    --output_name test_seq64 \
    --pruning_ratio 0.2 \
    --taylor_seq_len 64 \
    --taylor_num_samples 256

# 4. 查看最佳配置
cat results/param_search_mistral_20/best_config.json

# 5. 排序查看所有结果
python -c "
import pandas as pd
df = pd.read_csv('results/param_search_mistral_20/search_results.csv')
df = df[df['success']==True].sort_values('acc', ascending=False)
print(df[['taylor_seq_len', 'taylor_num_samples', 'acc', 'ppl']])
"
```

## 📈 预期结果

基于您的发现，预期：

| 配置 | 预期 ACC | 说明 |
|------|---------|------|
| seq_len=32, samples=256 | ⭐⭐⭐⭐⭐ | 最可能是最佳配置 |
| seq_len=64, samples=256 | ⭐⭐⭐⭐ | 也可能很好 |
| seq_len=128, samples=256 | ⭐⭐⭐ | 可能开始下降 |
| seq_len=256, samples=256 | ⭐⭐ | 可能较差（梯度不稳定） |

## 🛠️ 故障排除

### OOM (显存不足)

```bash
# 降低批次大小
--gradient_batch_size 2

# 使用梯度检查点
--use_gradient_checkpointing
```

### 搜索中断了

```bash
# 使用 --resume 继续
python param_search/search_best_params.py \
    --config configs/mistral_param_search.json \
    --resume
```

### 想修改评估任务

编辑配置文件的 `other_args`:

```json
{
  "other_args": {
    "eval_zeroshot_tasks": "boolq,piqa,hellaswag",
    "eval_ppl_datasets": "wikitext2,ptb"
  }
}
```

## 📚 更多文档

- **完整指南**: `PARAM_SEARCH_GUIDE.md`
- **示例脚本**: `examples/quick_param_search.sh`
- **配置文件**: `configs/mistral_param_search.json`

## 🎓 关键要点

1. ✅ **短序列通常更好**（对 Mistral）：16-64 比 256-512 更稳定
2. ✅ **256 样本数是个好起点**：平衡速度和准确性
3. ✅ **使用自动搜索**：比手动尝试效率高 10 倍
4. ✅ **查看热力图**：直观了解参数影响
5. ✅ **验证稳定性**：多次运行确保结果可靠

**开始搜索吧！找到最佳配置后，您的 Mistral 剪枝效果会显著提升！** 🚀
