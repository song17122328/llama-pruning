#!/bin/bash
# 快速参数搜索示例 - Mistral-7B

# 设置模型路径（修改为您的实际路径）
MODEL_PATH="/path/to/Mistral-7B-v0.3"

# 方法 1: 手动测试单个配置
echo "========================================"
echo "方法 1: 手动测试单个配置"
echo "========================================"

python run_global_pruning.py \
    --base_model "$MODEL_PATH" \
    --output_name mistral_seq32_samples256 \
    --pruning_ratio 0.2 \
    --taylor_seq_len 32 \
    --taylor_num_samples 256 \
    --dataset c4 \
    --importance_method taylor \
    --temperature 0.0 \
    --run_evaluation ppl,zeroshot

# 方法 2: 快速参数搜索（2-3 个配置）
echo ""
echo "========================================"
echo "方法 2: 快速参数搜索"
echo "========================================"

# 创建临时配置文件
cat > /tmp/quick_search.json <<EOF
{
  "base_model": "$MODEL_PATH",
  "pruning_ratio": 0.2,
  "output_base": "quick_search_mistral",
  "search_params": {
    "taylor_seq_len": [32, 64],
    "taylor_num_samples": [256]
  },
  "other_args": {
    "dataset": "c4",
    "temperature": 0.0,
    "importance_method": "taylor",
    "run_evaluation": "ppl,zeroshot",
    "eval_ppl_datasets": "wikitext2",
    "eval_zeroshot_tasks": "boolq,piqa"
  }
}
EOF

python search_best_params.py --config /tmp/quick_search.json

# 查看结果
echo ""
echo "========================================"
echo "查看最佳配置"
echo "========================================"
cat results/quick_search_mistral/best_config.json

echo ""
echo "完成！查看详细结果："
echo "  - CSV: results/quick_search_mistral/search_results.csv"
echo "  - 最佳配置: results/quick_search_mistral/best_config.json"
