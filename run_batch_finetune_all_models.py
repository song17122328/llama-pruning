#!/usr/bin/env python3
"""
批量微调和评估 all_model_blockwise_128_128_8 下的所有剪枝模型

目录结构:
    results/all_model_blockwise_128_128_8/
    ├── Llama/
    │   ├── blockwise/pruned_model.bin
    │   ├── magnitude/pruned_model.bin
    │   ├── wanda/pruned_model.bin
    │   ├── LLM-Pruner/pruned_model.bin
    │   ├── ShortGPT_remove_7/pruned_model.bin
    │   └── ShortGPT_remove_8/pruned_model.bin
    ├── Llama-Instruct/
    │   └── ...
    └── ...

工作流程:
    1. 评估剪枝后模型（如果还没评估）
    2. LoRA 微调
    3. 评估微调后模型

用法:
    # 处理单个模型的单个配置
    python run_batch_finetune_all_models.py --model Llama --method blockwise

    # 处理单个模型的所有配置
    python run_batch_finetune_all_models.py --model Llama --all-methods

    # 处理所有模型的所有配置（顺序执行）
    python run_batch_finetune_all_models.py --all

    # 并行处理（使用4个GPU）
    python run_batch_finetune_all_models.py --all --num-gpus 4

    # 跳过已完成的任务
    python run_batch_finetune_all_models.py --all --skip-completed

    # 只评估（不微调）
    python run_batch_finetune_all_models.py --all --stage evaluate-before

    # 只微调（假设已评估）
    python run_batch_finetune_all_models.py --all --stage finetune

    # 只评估微调后的模型
    python run_batch_finetune_all_models.py --all --stage evaluate-after
"""

import argparse
import json
import subprocess
import os
from pathlib import Path
import sys
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def setup_logging(log_file=None):
    """设置日志系统"""
    logger = logging.getLogger('batch_finetune')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 终端输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"日志已保存到: {log_file}")

    return logger


def get_best_gpus(num_gpus):
    """获取N个剩余显存最大的GPU"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(',')
            gpu_id = int(parts[0].strip())
            free_mem = int(parts[1].strip())
            gpu_memory.append((gpu_id, free_mem))

        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        best_gpus = [gpu_id for gpu_id, _ in gpu_memory[:num_gpus]]

        print(f"✓ 选择了 {len(best_gpus)} 个GPU: {best_gpus}")
        for gpu_id, free_mem in gpu_memory[:num_gpus]:
            print(f"  GPU {gpu_id}: 剩余显存 {free_mem/1024:.2f} GB")

        return best_gpus

    except Exception as e:
        print(f"警告: 无法获取GPU信息 ({e})，使用默认GPU列表")
        return list(range(num_gpus))


def get_all_model_method_pairs():
    """
    获取所有模型和方法的配对列表

    Returns:
        list: [(model, method), ...] 列表
    """
    models = ['Llama', 'Llama-Instruct', 'Mistral', 'Mistral-Instruct', 'Qwen', 'Qwen-Instruct']

    # 每个模型的方法列表（根据实际文件）
    method_map = {
        'Llama': ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner', 'ShortGPT_remove_7', 'ShortGPT_remove_8'],
        'Llama-Instruct': ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner', 'ShortGPT_remove_7', 'ShortGPT_remove_8'],
        'Mistral': ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner', 'ShortGPT_remove_6', 'ShortGPT_remove_7'],
        'Mistral-Instruct': ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner', 'ShortGPT_remove_6', 'ShortGPT_remove_7'],
        'Qwen': ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner', 'ShortGPT_remove_6', 'ShortGPT_remove_7'],
        'Qwen-Instruct': ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner', 'ShortGPT_remove_6', 'ShortGPT_remove_7']
    }

    pairs = []
    for model in models:
        for method in method_map[model]:
            pairs.append((model, method))

    return pairs


def check_task_status(model, method):
    """
    检查任务完成状态

    Returns:
        dict: {
            'pruned_model_exists': bool,
            'before_eval_exists': bool,
            'finetuned_model_exists': bool,
            'after_eval_exists': bool
        }
    """
    base_dir = Path('results/all_model_blockwise_128_128_8') / model / method

    status = {
        'pruned_model_exists': (base_dir / 'pruned_model.bin').exists(),
        'before_eval_exists': (base_dir / 'evaluation' / 'evaluation_results.json').exists(),
        'finetuned_model_exists': False,
        'after_eval_exists': False
    }

    # 微调后的模型保存位置
    finetuned_dir = Path('results/all_model_blockwise_128_128_8_finetuned') / model / method
    status['finetuned_model_exists'] = (finetuned_dir / 'pruned_model.bin').exists() or \
                                       (finetuned_dir / 'adapter_model.safetensors').exists()
    status['after_eval_exists'] = (finetuned_dir / 'evaluation_after_finetune' / 'evaluation_results.json').exists()

    return status


def evaluate_pruned_model(model, method, gpu_id, logger=None):
    """
    评估剪枝后的模型

    Args:
        model: 模型名称
        method: 剪枝方法
        gpu_id: GPU ID
        logger: 日志对象
    """
    log = logger.info if logger else print

    pruned_model_path = Path('results/all_model_blockwise_128_128_8') / model / method / 'pruned_model.bin'
    eval_output_dir = Path('results/all_model_blockwise_128_128_8') / model / method / 'evaluation'
    eval_output_json = eval_output_dir / 'evaluation_results.json'

    if not pruned_model_path.exists():
        log(f"✗ 剪枝模型不存在: {pruned_model_path}")
        return False

    # 创建评估输出目录
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log(f"[GPU {gpu_id}] 评估剪枝模型: {model}/{method}")

    # 构建评估命令
    cmd = [
        'python', 'evaluation/run_evaluation.py',
        '--model_path', str(pruned_model_path),
        '--output', str(eval_output_json),
        '--metrics', 'ppl,zeroshot'
    ]

    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        log(f"[GPU {gpu_id}] ✓ 评估完成: {eval_output_json}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"[GPU {gpu_id}] ✗ 评估失败: {e}")
        if e.stderr:
            log(f"  错误信息: {e.stderr[:500]}")
        return False


def finetune_model(model, method, gpu_id, lora_config=None, logger=None):
    """
    微调剪枝后的模型

    Args:
        model: 模型名称
        method: 剪枝方法
        gpu_id: GPU ID
        lora_config: LoRA配置
        logger: 日志对象
    """
    log = logger.info if logger else print

    pruned_model_path = Path('results/all_model_blockwise_128_128_8') / model / method / 'pruned_model.bin'
    finetuned_dir = Path('results/all_model_blockwise_128_128_8_finetuned') / model / method

    if not pruned_model_path.exists():
        log(f"✗ 剪枝模型不存在: {pruned_model_path}")
        return False

    # 创建输出目录
    finetuned_dir.mkdir(parents=True, exist_ok=True)

    # 默认LoRA配置
    if lora_config is None:
        lora_config = {
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'num_epochs': 2,
            'learning_rate': 1e-4,
            'batch_size': 64,
            'micro_batch_size': 4
        }

    # 保存微调配置
    config_file = finetuned_dir / 'finetuning_config.json'
    with open(config_file, 'w') as f:
        json.dump({
            'model': model,
            'method': method,
            'lora_config': lora_config,
            'pruned_model_path': str(pruned_model_path)
        }, f, indent=2)

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log(f"[GPU {gpu_id}] 微调模型: {model}/{method}")

    # 构建微调命令
    cmd = [
        'python', 'finetune_lora.py',
        '--pruned_model', str(pruned_model_path),
        '--output_dir', str(finetuned_dir),
        '--lora_r', str(lora_config['lora_r']),
        '--lora_alpha', str(lora_config['lora_alpha']),
        '--lora_dropout', str(lora_config['lora_dropout']),
        '--num_epochs', str(lora_config['num_epochs']),
        '--learning_rate', str(lora_config['learning_rate']),
        '--batch_size', str(lora_config['batch_size']),
        '--micro_batch_size', str(lora_config['micro_batch_size']),
        '--gradient_checkpointing'
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
        log(f"[GPU {gpu_id}] ✓ 微调完成: {finetuned_dir}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"[GPU {gpu_id}] ✗ 微调失败: {e}")
        return False


def evaluate_finetuned_model(model, method, gpu_id, logger=None):
    """
    评估微调后的模型

    Args:
        model: 模型名称
        method: 剪枝方法
        gpu_id: GPU ID
        logger: 日志对象
    """
    log = logger.info if logger else print

    finetuned_model_path = Path('results/all_model_blockwise_128_128_8_finetuned') / model / method / 'pruned_model.bin'
    eval_output_dir = Path('results/all_model_blockwise_128_128_8_finetuned') / model / method / 'evaluation_after_finetune'
    eval_output_json = eval_output_dir / 'evaluation_results.json'

    if not finetuned_model_path.exists():
        log(f"✗ 微调模型不存在: {finetuned_model_path}")
        return False

    # 创建评估输出目录
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log(f"[GPU {gpu_id}] 评估微调后模型: {model}/{method}")

    # 构建评估命令
    cmd = [
        'python', 'evaluation/run_evaluation.py',
        '--model_path', str(finetuned_model_path),
        '--output', str(eval_output_json),
        '--metrics', 'ppl,zeroshot'
    ]

    try:
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        log(f"[GPU {gpu_id}] ✓ 评估完成: {eval_output_json}")
        return True
    except subprocess.CalledProcessError as e:
        log(f"[GPU {gpu_id}] ✗ 评估失败: {e}")
        if e.stderr:
            log(f"  错误信息: {e.stderr[:500]}")
        return False


def process_single_model(model, method, stage, gpu_id, skip_completed=False, logger=None):
    """
    处理单个模型的完整流程

    Args:
        model: 模型名称
        method: 剪枝方法
        stage: 执行阶段 ('all', 'evaluate-before', 'finetune', 'evaluate-after')
        gpu_id: GPU ID
        skip_completed: 是否跳过已完成的步骤
        logger: 日志对象

    Returns:
        tuple: (task_name, success, message)
    """
    task_name = f"{model}/{method}"
    log = logger.info if logger else print

    try:
        status = check_task_status(model, method)

        # 检查剪枝模型是否存在
        if not status['pruned_model_exists']:
            return (task_name, False, "剪枝模型不存在")

        # Step 1: 评估剪枝后模型
        if stage in ['all', 'evaluate-before']:
            if skip_completed and status['before_eval_exists']:
                log(f"[GPU {gpu_id}] ⊙ 跳过评估（剪枝后）: {task_name}")
            else:
                success = evaluate_pruned_model(model, method, gpu_id, logger)
                if not success and stage == 'evaluate-before':
                    return (task_name, False, "评估剪枝模型失败")

        # Step 2: 微调
        if stage in ['all', 'finetune']:
            if skip_completed and status['finetuned_model_exists']:
                log(f"[GPU {gpu_id}] ⊙ 跳过微调: {task_name}")
            else:
                success = finetune_model(model, method, gpu_id, logger=logger)
                if not success:
                    return (task_name, False, "微调失败")

        # Step 3: 评估微调后模型
        if stage in ['all', 'evaluate-after']:
            if skip_completed and status['after_eval_exists']:
                log(f"[GPU {gpu_id}] ⊙ 跳过评估（微调后）: {task_name}")
            else:
                success = evaluate_finetuned_model(model, method, gpu_id, logger)
                if not success and stage == 'evaluate-after':
                    return (task_name, False, "评估微调模型失败")

        log(f"[GPU {gpu_id}] ✓ 完成: {task_name}")
        return (task_name, True, "成功")

    except Exception as e:
        log(f"[GPU {gpu_id}] ✗ 处理 {task_name} 时出错: {e}")
        return (task_name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='批量微调和评估所有剪枝模型')
    parser.add_argument('--model', type=str,
                       choices=['Llama', 'Llama-Instruct', 'Mistral', 'Mistral-Instruct', 'Qwen', 'Qwen-Instruct'],
                       help='模型名称')
    parser.add_argument('--method', type=str,
                       choices=['blockwise', 'magnitude', 'wanda', 'LLM-Pruner',
                               'ShortGPT_remove_6', 'ShortGPT_remove_7', 'ShortGPT_remove_8'],
                       help='剪枝方法')
    parser.add_argument('--all-methods', action='store_true',
                       help='处理指定模型的所有方法（需配合 --model 使用）')
    parser.add_argument('--all', action='store_true',
                       help='处理所有模型的所有方法')
    parser.add_argument('--stage', type=str,
                       choices=['all', 'evaluate-before', 'finetune', 'evaluate-after'],
                       default='all',
                       help='执行阶段')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='并行使用的GPU数量（默认1）')
    parser.add_argument('--skip-completed', action='store_true',
                       help='跳过已完成的步骤')
    parser.add_argument('--log-file', type=str, default=None,
                       help='日志文件路径')

    args = parser.parse_args()

    # 设置日志
    if args.log_file is None and (args.all or args.all_methods):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.all:
            args.log_file = str(log_dir / f'batch_finetune_all_{timestamp}.log')
        elif args.all_methods:
            args.log_file = str(log_dir / f'batch_finetune_{args.model}_{timestamp}.log')

    logger = setup_logging(args.log_file)

    # 构建任务列表
    tasks = []
    if args.all:
        tasks = get_all_model_method_pairs()
        logger.info(f"\n{'='*80}")
        logger.info(f"处理所有模型的所有方法")
        logger.info(f"{'='*80}")
        logger.info(f"\n将处理 {len(tasks)} 个任务")
    elif args.all_methods:
        if not args.model:
            parser.error("使用 --all-methods 需要同时指定 --model")

        all_pairs = get_all_model_method_pairs()
        tasks = [(m, method) for m, method in all_pairs if m == args.model]

        logger.info(f"\n{'='*80}")
        logger.info(f"处理模型的所有方法: {args.model}")
        logger.info(f"{'='*80}")
        logger.info(f"\n将处理 {len(tasks)} 个方法")
    else:
        if not args.model or not args.method:
            parser.error("需要指定 --model 和 --method，或使用 --all 或 --all-methods")
        tasks = [(args.model, args.method)]

    if args.skip_completed:
        logger.info("启用跳过已完成任务功能")

    logger.info(f"执行阶段: {args.stage}")

    # 执行任务
    if args.num_gpus > 1 and len(tasks) > 1:
        # 并行模式
        logger.info(f"\n并行模式: 使用 {args.num_gpus} 个GPU")

        gpu_ids = get_best_gpus(args.num_gpus)

        results = []
        with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
            future_to_task = {}
            for i, (model, method) in enumerate(tasks):
                gpu_id = gpu_ids[i % len(gpu_ids)]
                future = executor.submit(
                    process_single_model, model, method, args.stage, gpu_id,
                    args.skip_completed, logger
                )
                future_to_task[future] = (model, method)

            for future in as_completed(future_to_task):
                task_name, success, msg = future.result()
                results.append((task_name, success, msg))

        # 打印结果摘要
        logger.info(f"\n{'='*80}")
        logger.info(f"处理完成")
        logger.info(f"{'='*80}")
        success_count = sum(1 for _, success, _ in results if success)
        logger.info(f"\n成功: {success_count}/{len(results)}")

        if success_count < len(results):
            logger.info("\n失败的任务:")
            for task_name, success, msg in results:
                if not success:
                    logger.info(f"  ✗ {task_name}: {msg}")
    else:
        # 顺序模式
        logger.info(f"\n顺序模式: 逐个处理")

        from core.utils.get_best_gpu import get_best_gpu

        success_count = 0
        for model, method in tasks:
            gpu_id = get_best_gpu()
            task_name, success, msg = process_single_model(
                model, method, args.stage, gpu_id,
                args.skip_completed, logger
            )
            if success:
                success_count += 1

        logger.info(f"\n✓ 处理完成")
        logger.info(f"成功: {success_count}/{len(tasks)}")


if __name__ == '__main__':
    main()
