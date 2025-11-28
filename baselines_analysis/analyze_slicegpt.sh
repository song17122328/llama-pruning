#!/bin/bash
#
# SliceGPT 模型结构分析脚本
# 自动生成于: 2025-11-28 00:54:23
#
# 使用方法：
#   conda activate slicegpt
#   bash baselines_analysis/analyze_slicegpt.sh
#

echo "======================================="
echo "SliceGPT 模型结构分析"
echo "======================================="
echo ""

# 检查是否在 slicegpt 环境中
if [[ "$CONDA_DEFAULT_ENV" != "slicegpt" ]]; then
    echo "错误: 请先激活 slicegpt 环境"
    echo "运行: conda activate slicegpt"
    exit 1
fi

# 确保安装了 dill
echo "检查依赖..."
pip show dill > /dev/null 2>&1 || pip install dill
echo ""

echo "======================================="
echo "[1/2] 分析模型: SliceGPT_2000"
echo "======================================="
echo "模型配置: baselines/SliceGPT_2000/Llama-3-8B-Instruct_0.2.json"
echo ""

# 创建输出目录
mkdir -p baselines/SliceGPT_2000/analysis

# 使用 SliceGPT 的特殊加载方式分析模型
# 注意：这里需要实现 SliceGPT 的分析逻辑
# 暂时跳过，因为 SliceGPT 需要特殊的加载方式
echo "⚠️  警告: SliceGPT 模型需要特殊处理，请手动分析"
echo "  配置文件: baselines/SliceGPT_2000/Llama-3-8B-Instruct_0.2.json"
echo "  输出目录: baselines/SliceGPT_2000/analysis"
echo ""

echo "======================================="
echo "[2/2] 分析模型: SliceGPT_PCA_2000"
echo "======================================="
echo "模型配置: baselines/SliceGPT_PCA_2000/Llama-3-8B-Instruct_0.2.json"
echo ""

# 创建输出目录
mkdir -p baselines/SliceGPT_PCA_2000/analysis

# 使用 SliceGPT 的特殊加载方式分析模型
# 注意：这里需要实现 SliceGPT 的分析逻辑
# 暂时跳过，因为 SliceGPT 需要特殊的加载方式
echo "⚠️  警告: SliceGPT 模型需要特殊处理，请手动分析"
echo "  配置文件: baselines/SliceGPT_PCA_2000/Llama-3-8B-Instruct_0.2.json"
echo "  输出目录: baselines/SliceGPT_PCA_2000/analysis"
echo ""

echo "======================================="
echo "完成！"
echo "======================================="