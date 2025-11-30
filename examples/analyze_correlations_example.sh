#!/bin/bash
# 相关性分析完整示例

echo "======================================"
echo "步骤 1: 运行参数搜索（包含梯度统计）"
echo "======================================"

# 修改模型路径
MODEL_PATH="/path/to/Mistral-7B-v0.3"

# 创建配置文件
cat > /tmp/correlation_search.json <<EOF
{
  "base_model": "$MODEL_PATH",
  "pruning_ratio": 0.2,
  "output_base": "correlation_analysis_demo",
  "search_params": {
    "taylor_seq_len": [32, 64, 128],
    "taylor_num_samples": [256]
  },
  "other_args": {
    "dataset": "c4",
    "temperature": 0.0,
    "importance_method": "taylor",
    "run_evaluation": "ppl,zeroshot",
    "eval_ppl_datasets": "wikitext2",
    "eval_zeroshot_tasks": "boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa"
  }
}
EOF

# 运行搜索
python param_search/search_best_params.py --config /tmp/correlation_search.json

echo ""
echo "======================================"
echo "步骤 2: 分析相关性"
echo "======================================"

# 运行相关性分析
python param_search/analyze_param_correlations.py \
    --results results/correlation_analysis_demo/search_results.csv \
    --output_dir results/correlation_analysis_demo/

echo ""
echo "======================================"
echo "步骤 3: 查看结果"
echo "======================================"

# 显示最佳配置
echo "最佳配置:"
cat results/correlation_analysis_demo/best_config.json | python -m json.tool

echo ""
echo "相关性分析 Top 5:"
head -20 results/correlation_analysis_demo/correlation_report.txt

echo ""
echo "======================================"
echo "生成的文件:"
echo "======================================"
ls -lh results/correlation_analysis_demo/

echo ""
echo "查看可视化结果:"
echo "  - 相关性热力图: results/correlation_analysis_demo/correlation_heatmap.png"
echo "  - 散点图矩阵: results/correlation_analysis_demo/scatter_matrix.png"
echo "  - 详细报告: results/correlation_analysis_demo/correlation_report.txt"
echo "  - 预测模型: results/correlation_analysis_demo/prediction_model.json"
