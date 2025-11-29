#!/bin/bash
# 项目清理脚本
# 自动整理项目文件，移动输出到 outputs/ 目录

set -e

echo "================================"
echo "项目清理脚本"
echo "================================"
echo ""

# 1. 创建 outputs 目录
echo "1. 创建 outputs 目录..."
mkdir -p outputs

# 2. 移动有用的输出文件到 outputs/
echo "2. 移动输出文件到 outputs/..."
if [ -f "models_structure_summary.json" ]; then
    mv models_structure_summary.json outputs/
    echo "  ✓ 移动 models_structure_summary.json"
fi

if [ -f "models_structure_summary.txt" ]; then
    mv models_structure_summary.txt outputs/
    echo "  ✓ 移动 models_structure_summary.txt"
fi

if [ -f "baselines_compare.xlsx" ]; then
    mv baselines_compare.xlsx outputs/
    echo "  ✓ 移动 baselines_compare.xlsx"
fi

# 3. 删除旧的汇总表文件
echo "3. 删除旧的汇总表文件..."
rm -f summary_table.csv summary_table.xlsx summary_table.md summary_table.html
echo "  ✓ 删除 summary_table.*"

# 4. 删除测试文件
echo "4. 删除测试文件..."
rm -f evaluation/test_model_info.py
rm -f evaluation/test_batch_scan.py
echo "  ✓ 删除测试文件"

# 5. 删除 Python 缓存
echo "5. 清理 Python 缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ 清理 Python 缓存"

# 6. 删除临时文件
echo "6. 删除临时文件..."
find . -type f -name "*~" -delete 2>/dev/null || true
find . -type f -name "*.swp" -delete 2>/dev/null || true
find . -type f -name ".DS_Store" -delete 2>/dev/null || true
echo "  ✓ 清理临时文件"

# 7. 可选：清理图表输出（会重新生成）
echo ""
read -p "是否清理 pruning_charts/ 目录？（图表会重新生成） [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf pruning_charts/
    echo "  ✓ 删除 pruning_charts/"
else
    echo "  ⊘ 保留 pruning_charts/"
fi

# 8. 报告结果
echo ""
echo "================================"
echo "清理完成！"
echo "================================"
echo ""
echo "输出文件已移动到: outputs/"
ls -lh outputs/ 2>/dev/null || echo "  (空目录)"
echo ""
echo "建议下一步："
echo "  1. 检查 outputs/ 目录内容"
echo "  2. 运行: git status 查看变更"
echo "  3. 更新 .gitignore 忽略 outputs/ 目录"
