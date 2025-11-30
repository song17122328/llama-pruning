# 参数搜索指南

## 📋 概述

本指南帮助您找到最佳的 Taylor 重要性计算参数（序列长度和样本数），以获得最高的剪枝后准确率。

## 🎯 背景

您发现 `TAYLOR_SEQ_LEN` 长度变短可以提高 Mistral 的剪枝后 ACC，可能是因为：
- **梯度爆炸/消失减轻**：短序列减少了梯度累积带来的数值不稳定
- **更准确的局部重要性**：短序列更能捕捉参数的局部重要性
- **计算稳定性**：避免长序列导致的浮点精度问题

## 🔧 新增的命令行参数

### 基础参数

```bash
--taylor_num_samples 256         # Taylor 重要性计算的样本数（默认: 256）
--taylor_seq_len 32              # Taylor 重要性计算的序列长度（默认: 32）
```

### 高级参数（可选）

```bash
--layer_importance_num_samples 50   # 层重要性分析的样本数（默认: 50）
--layer_importance_seq_len 32       # 层重要性分析的序列长度（默认: 32）
--block_importance_num_samples 50   # 块重要性分析的样本数（默认: 50）
--block_importance_seq_len 32       # 块重要性分析的序列长度（默认: 32）
```

## 📝 使用方法

### 方法 1: 手动测试单个配置

```bash
python run_global_pruning.py \
    --base_model /path/to/Mistral-7B-v0.3 \
    --output_name mistral_test_seq64 \
    --pruning_ratio 0.2 \
    --taylor_seq_len 64 \
    --taylor_num_samples 256 \
    --dataset c4 \
    --importance_method taylor
```

### 方法 2: 自动网格搜索（推荐）

#### 步骤 1: 修改配置文件

编辑 `configs/mistral_param_search.json`：

```json
{
  "base_model": "/data/models/Mistral-7B-v0.3",  // ← 修改为您的模型路径
  "pruning_ratio": 0.2,
  "output_base": "param_search_mistral_20",
  "search_params": {
    "taylor_seq_len": [16, 32, 64, 128, 256],     // ← 要测试的序列长度
    "taylor_num_samples": [128, 256, 512]          // ← 要测试的样本数
  },
  "other_args": {
    "dataset": "c4",
    "temperature": 0.0,
    "importance_method": "taylor",
    "run_evaluation": "ppl,zeroshot"
  }
}
```

#### 步骤 2: 运行搜索

```bash
# 完整搜索（5 × 3 = 15 个实验）
python search_best_params.py --config configs/mistral_param_search.json

# 快速测试（仅测试 2 个配置）
python search_best_params.py --config configs/quick_param_search.json --max_experiments 2

# 从中断处继续
python search_best_params.py --config configs/mistral_param_search.json --resume
```

#### 步骤 3: 查看结果

搜索完成后，会生成以下文件：

1. **`results/param_search_mistral_20/search_results.csv`** - 所有实验结果
2. **`results/param_search_mistral_20/best_config.json`** - 最佳配置

**查看最佳配置**：

```bash
cat results/param_search_mistral_20/best_config.json
```

输出示例：

```json
{
  "params": {
    "taylor_seq_len": 64,
    "taylor_num_samples": 256
  },
  "metrics": {
    "acc": 0.6234,
    "ppl": 12.45,
    "pruning_ratio": 0.201
  },
  "output_dir": "results/param_search_mistral_20/exp_008_taylor_seq_len64_taylor_num_samples256"
}
```

**查看所有结果排名**：

```bash
# 按 ACC 排序
python -c "import pandas as pd; df = pd.read_csv('results/param_search_mistral_20/search_results.csv'); print(df.sort_values('acc', ascending=False)[['taylor_seq_len', 'taylor_num_samples', 'acc', 'ppl']].to_string())"
```

## 📊 配置建议

### 快速测试（2-3 小时）

测试少量关键配置，快速验证假设：

```json
{
  "search_params": {
    "taylor_seq_len": [32, 64, 128],
    "taylor_num_samples": [256]
  }
}
```

### 标准搜索（6-8 小时）

平衡搜索空间和时间：

```json
{
  "search_params": {
    "taylor_seq_len": [16, 32, 64, 128, 256],
    "taylor_num_samples": [128, 256, 512]
  }
}
```

### 精细搜索（12-16 小时）

找到更精确的最佳值：

```json
{
  "search_params": {
    "taylor_seq_len": [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    "taylor_num_samples": [64, 128, 256, 384, 512, 768]
  }
}
```

## 🧪 推荐搜索策略

### 阶段 1: 粗粒度搜索

先用大步长找到大致范围：

```json
{
  "search_params": {
    "taylor_seq_len": [16, 64, 256],
    "taylor_num_samples": [128, 512]
  }
}
```

### 阶段 2: 精细搜索

在最佳范围附近细化：

假设阶段 1 发现 `seq_len=64` 最好，则：

```json
{
  "search_params": {
    "taylor_seq_len": [48, 56, 64, 72, 80],
    "taylor_num_samples": [256]
  }
}
```

## 💡 参数选择建议

### `taylor_seq_len` (序列长度)

| 值 | 优点 | 缺点 | 适用场景 |
|----|------|------|---------|
| **16-32** | 梯度稳定，计算快 | 可能丢失长距离依赖 | Mistral 等梯度不稳定模型 |
| **64-128** | 平衡性能和稳定性 | - | 通用推荐 |
| **256-512** | 捕捉长距离依赖 | 可能梯度爆炸/消失 | 稳定模型（如 LLaMA） |

### `taylor_num_samples` (样本数)

| 值 | 优点 | 缺点 | 适用场景 |
|----|------|------|---------|
| **64-128** | 计算快 | 统计不稳定 | 快速原型 |
| **256** | 平衡速度和准确性 | - | **推荐** |
| **512-1024** | 统计更稳定 | 计算慢 | 最终优化 |

## 🔍 结果分析

### 查看 CSV 结果

```python
import pandas as pd

# 读取结果
df = pd.read_csv('results/param_search_mistral_20/search_results.csv')

# 只看成功的实验
df_valid = df[df['success'] == True]

# 按 ACC 降序排列
df_sorted = df_valid.sort_values('acc', ascending=False)

# 显示 Top 10
print(df_sorted[['taylor_seq_len', 'taylor_num_samples', 'acc', 'ppl']].head(10))

# 绘制热力图
import matplotlib.pyplot as plt
import seaborn as sns

pivot = df_valid.pivot_table(
    values='acc',
    index='taylor_seq_len',
    columns='taylor_num_samples',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu')
plt.title('ACC vs. Seq_Len & Num_Samples')
plt.xlabel('Num Samples')
plt.ylabel('Seq Length')
plt.savefig('param_heatmap.png', dpi=150, bbox_inches='tight')
print("热力图已保存到 param_heatmap.png")
```

### 关键指标

1. **ACC (准确率)**: 主要优化目标，越高越好
2. **PPL (困惑度)**: 次要指标，越低越好
3. **剪枝率**: 确保接近目标（如 20%）

## ⚙️ 高级用法

### 自定义评估任务

修改配置文件的 `other_args` 部分：

```json
{
  "other_args": {
    "eval_zeroshot_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,mmlu",
    "eval_ppl_datasets": "wikitext2,ptb,c4"
  }
}
```

### 并行运行多个搜索

如果有多个 GPU，可以同时运行多个搜索：

```bash
# GPU 0: 搜索 seq_len
CUDA_VISIBLE_DEVICES=0 python search_best_params.py \
    --config configs/search_seq_len.json &

# GPU 1: 搜索 num_samples
CUDA_VISIBLE_DEVICES=1 python search_best_params.py \
    --config configs/search_num_samples.json &

wait
```

## 📈 预期结果

基于您的发现（短序列提高 ACC），预期结果：

```
最佳配置可能是:
  taylor_seq_len: 32-64 (比默认的 256 短很多)
  taylor_num_samples: 256-512

预期提升:
  ACC: +2-5% (相比 seq_len=256)
  剪枝均衡性: Layer 2-4 不再极端剪枝
```

## 🛠️ 故障排除

### 问题 1: OOM (显存不足)

**解决方法**：
- 降低 `gradient_batch_size`
- 使用 `--use_gradient_checkpointing`
- 减少 `taylor_num_samples`

### 问题 2: 搜索中断

**解决方法**：
```bash
# 使用 --resume 从中断处继续
python search_best_params.py --config configs/mistral_param_search.json --resume
```

### 问题 3: 结果异常

**检查**：
- 查看 `results/*/logs/` 中的日志
- 确保所有实验都成功（`success=True`）
- 验证剪枝率是否接近目标

## 📚 示例工作流

### 完整示例: Mistral-7B 参数优化

```bash
# 1. 创建配置文件
cat > configs/my_mistral_search.json <<EOF
{
  "base_model": "/data/models/Mistral-7B-v0.3",
  "pruning_ratio": 0.2,
  "output_base": "mistral_optimal_params",
  "search_params": {
    "taylor_seq_len": [16, 32, 64, 128],
    "taylor_num_samples": [256, 512]
  },
  "other_args": {
    "dataset": "c4",
    "temperature": 0.0,
    "importance_method": "taylor",
    "run_evaluation": "ppl,zeroshot",
    "eval_zeroshot_tasks": "boolq,piqa,hellaswag,winogrande"
  }
}
EOF

# 2. 运行搜索
python search_best_params.py --config configs/my_mistral_search.json

# 3. 查看最佳配置
cat results/mistral_optimal_params/best_config.json

# 4. 使用最佳配置运行完整评估
python run_global_pruning.py \
    --base_model /data/models/Mistral-7B-v0.3 \
    --output_name mistral_final_best \
    --pruning_ratio 0.2 \
    --taylor_seq_len 64 \
    --taylor_num_samples 256 \
    --run_evaluation all \
    --dataset c4
```

## 🎓 总结

1. **使用自动搜索**：`search_best_params.py` 是最高效的方式
2. **从粗到细**：先粗粒度搜索，再精细优化
3. **关注 ACC**：这是最重要的指标
4. **验证稳定性**：最佳配置应该在多次运行中保持稳定
5. **记录最佳值**：将最佳配置保存为默认值

**下一步**：使用找到的最佳参数进行完整的剪枝+微调流程！
