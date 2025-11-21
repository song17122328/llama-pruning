#!/usr/bin/env python3
"""
效率指标评估

包括:
1. 参数量统计
2. 推理速度（吞吐量和延迟）
3. 内存/显存占用
"""

import torch
import time
import numpy as np
from typing import Dict, Tuple, Optional
import gc


def evaluate_efficiency(
    model,
    tokenizer,
    device: str = 'cuda',
    num_samples: int = 100,
    input_length: int = 512,
    output_length: int = 128,
    batch_sizes: list = [1, 4, 8]
) -> Dict:
    """
    全面评估模型效率

    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        num_samples: 测试样本数（用于速度测试）
        input_length: 输入序列长度
        output_length: 生成序列长度
        batch_sizes: 要测试的batch size列表

    Returns:
        包含所有效率指标的字典
    """
    print(f"\n{'='*60}")
    print(f"评估效率指标,当前的设备是{device}")
    print(f"{'='*60}")

    results = {}

    # 1. 参数量统计
    from evaluation.utils.model_loader import get_model_info
    model_info = get_model_info(model)
    results['model_info'] = model_info

    print(f"\n1. 参数量:")
    print(f"   总参数: {model_info['total_params_B']:.2f}B")
    if model_info['attention_params'] is not None:
        print(f"   Attention: {model_info['attention_params_M']:.1f}M ({model_info['attention_ratio']*100:.1f}%)")
        print(f"   MLP: {model_info['mlp_params_M']:.1f}M ({model_info['mlp_ratio']*100:.1f}%)")

    # 2. 推理速度
    print(f"\n2. 推理速度:")
    speed_results = {}
    for bs in batch_sizes:
        try:
            speed_res = measure_inference_speed(
                model, tokenizer,
                batch_size=bs,
                input_length=input_length,
                output_length=output_length,
                num_samples=num_samples // bs,  # 调整样本数以保持总token数大致相同
                device=device
            )
            speed_results[f'batch_size_{bs}'] = speed_res

            print(f"   Batch={bs}:")
            print(f"     吞吐量: {speed_res['throughput_tokens_per_sec']:.2f} tokens/s")
            print(f"     延迟: {speed_res['latency_ms_per_token']:.2f} ms/token")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"   Batch={bs}: OOM (跳过)")
                torch.cuda.empty_cache()
            else:
                raise

        # 每次速度测试后清理缓存（只清理当前使用的GPU）
        gc.collect()
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()

    results['speed'] = speed_results

    # 3. 显存占用
    if device.startswith('cuda'):
        print(f"\n3. 显存占用:")

        # 彻底清理之前测试的残留，确保内存测量准确
        gc.collect()
        device_id = int(device.split(':')[1]) if ':' in device else 0
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()

        memory_results = measure_memory_usage(model, tokenizer, device=device)
        results['memory'] = memory_results

        print(f"   模型加载: {memory_results['model_memory_mb']:.1f} MB")
        print(f"   推理峰值: {memory_results['inference_peak_mb']:.1f} MB")

    return results


def measure_inference_speed(
    model,
    tokenizer,
    batch_size: int = 1,
    input_length: int = 512,
    output_length: int = 128,
    num_samples: int = 100,
    device: str = 'cuda',
    warmup_steps: int = 10
) -> Dict[str, float]:
    """
    测量推理速度

    Args:
        model: 模型
        tokenizer: tokenizer
        batch_size: 批次大小
        input_length: 输入长度
        output_length: 输出长度
        num_samples: 测试样本数
        device: 设备
        warmup_steps: 预热步数

    Returns:
        {
            'throughput_tokens_per_sec': float,
            'latency_ms_per_token': float,
            'total_time_sec': float
        }
    """
    model.eval()

    # 准备dummy输入
    dummy_text = "This is a test sentence for measuring inference speed. " * 20
    inputs = tokenizer(
        [dummy_text] * batch_size,
        return_tensors='pt',
        max_length=input_length,
        truncation=True,
        padding='max_length'
    )

    # 移动输入到目标设备（支持cuda和cuda:N格式）
    if device.startswith('cuda'):
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Warmup
    print(f"   预热中... ({warmup_steps} steps)")
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    # 正式测速
    print(f"   测速中... ({num_samples} samples, batch_size={batch_size})")

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=output_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

    if device.startswith('cuda'):
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time

    # 计算指标
    total_tokens = num_samples * batch_size * output_length
    throughput = total_tokens / elapsed_time
    latency_ms = (elapsed_time * 1000) / total_tokens

    return {
        'throughput_tokens_per_sec': throughput,
        'latency_ms_per_token': latency_ms,
        'total_time_sec': elapsed_time,
        'total_tokens': total_tokens
    }


def measure_memory_usage(
    model,
    tokenizer,
    device: str = 'cuda',
    input_length: int = 512
) -> Dict[str, float]:
    """
    测量显存占用

    Args:
        model: 模型
        tokenizer: tokenizer
        device: 设备
        input_length: 测试输入长度

    Returns:
        {
            'model_memory_mb': float,  # 模型加载显存
            'inference_peak_mb': float  # 推理峰值显存
        }
    """
    if not device.startswith('cuda'):
        return {
            'model_memory_mb': 0,
            'inference_peak_mb': 0
        }

    # 提取GPU编号
    device_id = int(device.split(':')[1]) if ':' in device else 0

    # 清空缓存（empty_cache不接受参数，清空所有GPU）
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_id)

    # 测量模型加载显存
    model_memory = torch.cuda.memory_allocated(device_id) / (1024 ** 2)

    # 测量推理峰值显存
    dummy_text = "Test " * input_length
    inputs = tokenizer(
        dummy_text,
        return_tensors='pt',
        max_length=input_length,
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    torch.cuda.synchronize(device_id)
    peak_memory = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)

    return {
        'model_memory_mb': model_memory,
        'inference_peak_mb': peak_memory
    }


def compare_efficiency(
    results_dict: Dict[str, Dict],
    baseline_name: str = 'Original'
) -> Dict:
    """
    对比多个模型的效率

    Args:
        results_dict: {model_name: efficiency_results}
        baseline_name: baseline模型名称

    Returns:
        对比结果
    """
    if baseline_name not in results_dict:
        print(f"警告: baseline '{baseline_name}' 不在结果中")
        return None

    baseline = results_dict[baseline_name]
    comparison = {}

    for model_name, results in results_dict.items():
        if model_name == baseline_name:
            continue

        comp = {}

        # 参数量减少
        baseline_params = baseline['model_info']['total_params']
        model_params = results['model_info']['total_params']
        comp['param_reduction'] = (baseline_params - model_params) / baseline_params

        # 速度提升（使用batch_size=1的结果）
        if 'batch_size_1' in baseline['speed'] and 'batch_size_1' in results['speed']:
            baseline_throughput = baseline['speed']['batch_size_1']['throughput_tokens_per_sec']
            model_throughput = results['speed']['batch_size_1']['throughput_tokens_per_sec']
            comp['speedup'] = model_throughput / baseline_throughput

        # 内存减少
        if 'memory' in baseline and 'memory' in results:
            baseline_mem = baseline['memory']['model_memory_mb']
            model_mem = results['memory']['model_memory_mb']
            comp['memory_reduction'] = (baseline_mem - model_mem) / baseline_mem

        comparison[model_name] = comp

    return comparison


if __name__ == '__main__':
    # 测试
    import sys
    sys.path.insert(0, '../..')
    from evaluation.utils.model_loader import load_model_and_tokenizer

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)

    results = evaluate_efficiency(
        model, tokenizer,
        device=args.device,
        num_samples=50,
        batch_sizes=[1, 4]
    )

    print(f"\n完整结果: {results}")
