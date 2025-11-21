#!/usr/bin/env python3
"""
GPU选择函数 - 重导出自 core.utils.get_best_gpu

保持向后兼容性，实际实现在 core/utils/get_best_gpu.py
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.utils.get_best_gpu import get_best_gpu

__all__ = ['get_best_gpu']


# 使用示例
if __name__ == "__main__":
    gpu_id = get_best_gpu()
    print(f"使用 GPU {gpu_id}")
