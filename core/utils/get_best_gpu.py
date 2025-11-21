#!/usr/bin/env python3
"""
简单的GPU选择函数 - 返回剩余显存最大的GPU
"""

import subprocess
import torch

def get_best_gpu():
    """
    返回剩余显存最大的GPU编号
    
    Returns:
        int: GPU编号 (0-7)，如果失败返回0
    """
    try:
        # 使用nvidia-smi查询
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析结果
        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_id = int(parts[0].strip())
            free_mem = int(parts[1].strip())  # MB
            gpu_memory.append((gpu_id, free_mem))
        
        # 选择剩余显存最大的GPU
        best_gpu = max(gpu_memory, key=lambda x: x[1])
        
        print(f"✓ 选择 GPU {best_gpu[0]}, 剩余显存: {best_gpu[1]/1024:.2f} GB")
        
        return best_gpu[0]
        
    except Exception as e:
        print(f"警告: 无法获取GPU信息 ({e})，使用GPU 0")
        return 0


# 使用示例
if __name__ == "__main__":
    gpu_id = get_best_gpu()
    print(f"使用 GPU {gpu_id}")
    
    # 设置环境变量
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 之后加载模型时会使用选中的GPU
    # model = model.to('cuda:0')