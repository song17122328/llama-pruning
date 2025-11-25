#!/bin/bash
#
# 下载 Alpaca Cleaned 数据集到本地
#
# 用法:
#   bash scripts/download_alpaca_dataset.sh
#

set -e

echo "========================================"
echo "下载 Alpaca Cleaned 数据集"
echo "========================================"
echo ""

# 创建数据集目录
DATASET_DIR="datasets/alpaca-cleaned"
mkdir -p "$DATASET_DIR"

echo "数据集目录: $DATASET_DIR"
echo ""

# 数据集 URL
DATASET_URL="https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"
OUTPUT_FILE="$DATASET_DIR/alpaca_data_cleaned.json"

# 检查是否已经下载
if [ -f "$OUTPUT_FILE" ]; then
    echo "✓ 数据集已存在: $OUTPUT_FILE"
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "  文件大小: $FILE_SIZE"

    read -p "是否重新下载? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过下载"
        exit 0
    fi
fi

# 下载数据集
echo "正在下载数据集..."
echo "URL: $DATASET_URL"
echo ""

if command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$DATASET_URL" --show-progress
elif command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$DATASET_URL" --progress-bar
else
    echo "错误: 未找到 wget 或 curl"
    echo "请安装其中一个工具:"
    echo "  Ubuntu/Debian: sudo apt-get install wget"
    echo "  CentOS/RHEL: sudo yum install wget"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ 下载完成!"
echo "========================================"
echo ""

# 显示文件信息
if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")

    echo "文件信息:"
    echo "  路径: $OUTPUT_FILE"
    echo "  大小: $FILE_SIZE"
    echo "  行数: $LINE_COUNT"
    echo ""

    # 检查是否是有效的 JSON
    if command -v python3 &> /dev/null; then
        echo "验证 JSON 格式..."
        if python3 -c "import json; json.load(open('$OUTPUT_FILE'))" 2>/dev/null; then
            SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$OUTPUT_FILE'))))")
            echo "✓ JSON 格式有效"
            echo "  样本数量: $SAMPLE_COUNT"
        else
            echo "⚠ JSON 格式验证失败"
        fi
    fi

    echo ""
    echo "使用方法:"
    echo "  # 从本地文件加载"
    echo "  python finetune_lora.py \\"
    echo "      --pruned_model results/HGSP_2000/pruned_model.bin \\"
    echo "      --data_path $OUTPUT_FILE"
    echo ""
    echo "  # 或从目录加载"
    echo "  python finetune_lora.py \\"
    echo "      --pruned_model results/HGSP_2000/pruned_model.bin \\"
    echo "      --data_path $DATASET_DIR"
else
    echo "错误: 下载失败"
    exit 1
fi
