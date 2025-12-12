#!/usr/bin/env python3
"""
完整的剪枝-评估-微调-评估流程脚本

支持一键运行一类模型的所有剪枝方法（6种），并自动执行：
1. 剪枝（使用统一的校准参数）
2. 评估剪枝后模型
3. LoRA 微调
4. 评估微调后模型
5. 导出结果到 Excel

使用方法：
    python run_complete_pipeline.py \
        --model Llama \
        --taylor_num_samples 128 \
        --taylor_seq_len 128 \
        --layer_importance_num_samples 128 \
        --layer_importance_seq_len 128 \
        --block_importance_num_samples 128 \
        --block_importance_seq_len 128 \
        --gradient_batch_size 8 \
        --output_prefix test_128_128_8

    简化参数版本（使用 x1,x2,y1,y2,z 表示法）：
    python run_complete_pipeline.py \
        --model Llama \
        --x1 128 --x2 128 \
        --y1 128 --y2 128 \
        --z 8 \
        --output_prefix test_128_128_8

支持的模型：
    - Llama
    - Llama-Instruct
    - Mistral
    - Mistral-Instruct
    - Qwen
    - Qwen-Instruct

每个模型会运行6种剪枝方法：
    - blockwise: 本文方法
    - magnitude: Magnitude baseline
    - wanda: Wanda baseline
    - LLM-Pruner: LLM-Pruner baseline
    - ShortGPT_remove_X: ShortGPT baseline（层数根据模型不同）
"""

import argparse
import json
import subprocess
import os
from pathlib import Path
import sys
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# 模型路径映射
MODEL_PATHS = {
    'Llama': '/newdata/LLMs/Llama-3-8B',
    'Llama-Instruct': '/newdata/LLMs/Llama-3-8B-Instruct',
    'Mistral': '/newdata/LLMs/Mistral-7B-v0.3',
    'Mistral-Instruct': '/newdata/LLMs/Mistral-7B-Instruct-v0.3',
    'Qwen': '/newdata/LLMs/Qwen2.5-7B',
    'Qwen-Instruct': '/newdata/LLMs/Qwen2.5-7B-Instruct'
}

# ShortGPT 移除层数映射
SHORTGPT_LAYERS = {
    'Llama': [7, 8],
    'Llama-Instruct': [7, 8],
    'Mistral': [6, 7],
    'Mistral-Instruct': [6, 7],
    'Qwen': [6, 7],
    'Qwen-Instruct': [6, 7]
}


def setup_logging(log_file=None):
    """设置日志系统"""
    logger = logging.getLogger('complete_pipeline')
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

    return logger


def get_best_gpu():
    """获取剩余显存最大的GPU"""
    try:
        from core.utils.get_best_gpu import get_best_gpu as get_gpu
        return get_gpu()
    except:
        return 0


def run_pruning(model, method, params, output_dir, gpu_id, logger):
    """
    运行剪枝

    Args:
        model: 模型名称
        method: 剪枝方法 (blockwise, magnitude, wanda, LLM-Pruner, ShortGPT_remove_X)
        params: 校准参数字典
        output_dir: 输出目录
        gpu_id: GPU ID
        logger: 日志对象
    """
    base_model_path = MODEL_PATHS[model]
    log = logger.info

    log(f"[GPU {gpu_id}] 开始剪枝: {model}/{method}")

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 特别说明：对于 LLM-Pruner，需要同时设置 CUDA_VISIBLE_DEVICES 和 --device cuda
    # CUDA_VISIBLE_DEVICES 限制可见的 GPU，--device cuda 让 LLM-Pruner 使用 cuda:0（即被限制后的第一个GPU）
    log(f"[GPU {gpu_id}] 设置环境变量: CUDA_VISIBLE_DEVICES={gpu_id}")

    try:
        if method == 'blockwise':
            # blockwise 方法
            cmd = [
                'python', 'run_global_pruning.py',
                '--base_model', base_model_path,
                '--output_name', str(output_dir),
                '--taylor_num_samples', str(params['x1']),
                '--taylor_seq_len', str(params['x2']),
                '--layer_importance_num_samples', str(params['y1']),
                '--layer_importance_seq_len', str(params['y2']),
                '--block_importance_num_samples', str(params['y1']),
                '--block_importance_seq_len', str(params['y2']),
                '--gradient_batch_size', str(params['z']),
                '--pruning_ratio', '0.2'
            ]

        elif method == 'magnitude':
            # Magnitude baseline
            cmd = [
                'python', 'run_global_pruning.py',
                '--base_model', base_model_path,
                '--output_name', str(output_dir),
                '--importance_method', 'magnitude',
                '--taylor_num_samples', str(params['x1']),
                '--taylor_seq_len', str(params['x2']),
                '--gradient_batch_size', str(params['z']),
                '--pruning_ratio', '0.2',
                '--temperature', '0.0'
            ]

        elif method == 'wanda':
            # Wanda baseline
            cmd = [
                'python', 'run_global_pruning.py',
                '--base_model', base_model_path,
                '--output_name', str(output_dir),
                '--importance_method', 'wanda',
                '--taylor_num_samples', str(params['x1']),
                '--taylor_seq_len', str(params['x2']),
                '--gradient_batch_size', str(params['z']),
                '--pruning_ratio', '0.2',
                '--temperature', '0.0'
            ]

        elif method == 'LLM-Pruner':
            # LLM-Pruner baseline
            # 注意：LLM-Pruner 需要 CUDA_VISIBLE_DEVICES 环境变量（已在上面设置）+ --device cuda 参数
            # 这样 LLM-Pruner 会使用 cuda:0，而 cuda:0 实际映射到 CUDA_VISIBLE_DEVICES 指定的 GPU
            log(f"[GPU {gpu_id}] LLM-Pruner 特殊设置: CUDA_VISIBLE_DEVICES={gpu_id}, --device cuda")
            cmd = [
                'python', '/data/home/yuanxiaosong/LLM-Pruner_baseline/llama3.py',
                '--pruning_ratio', '0.28',
                '--device', 'cuda',          # 使用 cuda（cuda:0）
                '--eval_device', 'cuda',     # 评估也使用 cuda
                '--base_model', base_model_path,
                '--block_wise',
                '--block_mlp_layer_start', '4',
                '--block_mlp_layer_end', '30',
                '--block_attention_layer_start', '4',
                '--block_attention_layer_end', '30',
                '--save_ckpt_log_name', str(output_dir),
                '--pruner_type', 'taylor',
                '--taylor', 'param_first',
                '--num_examples', str(params['x1']),
                '--calibration_seq_len', str(params['x2']),
                '--save_model',
                '--calibration_dataset', 'c4',
                '--calibration_batch_size', str(params['z'])
            ]

        elif method.startswith('ShortGPT_remove_'):
            # ShortGPT baseline
            n_layers = int(method.split('_')[-1])
            cmd = [
                'python', 'baselines/run_shortgpt.py',
                '--base_model', base_model_path,
                '--n_remove_layers', str(n_layers),
                '--num_samples', str(params['y1']),
                '--seq_len', str(params['y2']),
                '--stride', str(params['y2']),
                '--output_name', str(output_dir)
            ]
        else:
            log(f"[GPU {gpu_id}] ✗ 未知的剪枝方法: {method}")
            return False

        # 执行命令
        log(f"[GPU {gpu_id}] 执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=7200)  # 2小时超时

        if result.returncode == 0:
            log(f"[GPU {gpu_id}] ✓ 剪枝完成: {method}")
            return True
        else:
            log(f"[GPU {gpu_id}] ✗ 剪枝失败: {method}")
            if result.stderr:
                log(f"  错误信息: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        log(f"[GPU {gpu_id}] ✗ 剪枝超时: {method}")
        return False
    except Exception as e:
        log(f"[GPU {gpu_id}] ✗ 剪枝异常: {method} - {e}")
        return False


def evaluate_model(model_path, output_json, gpu_id, logger):
    """
    评估模型

    Args:
        model_path: 模型路径（.bin文件）
        output_json: 输出JSON文件路径
        gpu_id: GPU ID
        logger: 日志对象
    """
    log = logger.info

    # 创建输出目录
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log(f"[GPU {gpu_id}] 评估模型: {model_path}")

    cmd = [
        'python', 'evaluation/run_evaluation.py',
        '--model_path', str(model_path),
        '--output', str(output_json),
        '--metrics', 'ppl,zeroshot'
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            log(f"[GPU {gpu_id}] ✓ 评估完成")
            return True
        else:
            log(f"[GPU {gpu_id}] ✗ 评估失败")
            if result.stderr:
                log(f"  错误信息: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        log(f"[GPU {gpu_id}] ✗ 评估超时")
        return False
    except Exception as e:
        log(f"[GPU {gpu_id}] ✗ 评估异常: {e}")
        return False


def finetune_model(pruned_model_path, output_dir, gpu_id, logger):
    """
    微调模型

    Args:
        pruned_model_path: 剪枝后模型路径
        output_dir: 微调输出目录
        gpu_id: GPU ID
        logger: 日志对象
    """
    log = logger.info

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log(f"[GPU {gpu_id}] 微调模型: {pruned_model_path}")

    cmd = [
        'python', 'finetune_lora.py',
        '--pruned_model', str(pruned_model_path),
        '--output_dir', str(output_dir),
        '--lora_r', '8',
        '--lora_alpha', '16',
        '--lora_dropout', '0.05',
        '--num_epochs', '2',
        '--learning_rate', '1e-4',
        '--batch_size', '64',
        '--micro_batch_size', '4',
        '--gradient_checkpointing'
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=14400)  # 4小时超时

        if result.returncode == 0:
            log(f"[GPU {gpu_id}] ✓ 微调完成")
            return True
        else:
            log(f"[GPU {gpu_id}] ✗ 微调失败")
            if result.stderr:
                log(f"  错误信息: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        log(f"[GPU {gpu_id}] ✗ 微调超时")
        return False
    except Exception as e:
        log(f"[GPU {gpu_id}] ✗ 微调异常: {e}")
        return False


def process_single_method(model, method, params, output_prefix, gpu_id, skip_completed, logger):
    """
    处理单个剪枝方法的完整流程

    Returns:
        tuple: (method, success, results_dict)
    """
    log = logger.info

    # 输出目录
    pruned_dir = Path('results') / output_prefix / model / method
    finetuned_dir = Path('results') / f'{output_prefix}_finetuned' / model / method

    results = {
        'model': model,
        'method': method,
        'pruning_success': False,
        'eval_before_success': False,
        'finetune_success': False,
        'eval_after_success': False,
        'ppl_before': None,
        'acc_before': None,
        'ppl_after': None,
        'acc_after': None
    }

    try:
        # Step 1: 剪枝
        pruned_model_path = pruned_dir / 'pruned_model.bin'
        if skip_completed and pruned_model_path.exists():
            log(f"[GPU {gpu_id}] ⊙ 跳过剪枝（已存在）: {model}/{method}")
            results['pruning_success'] = True
        else:
            success = run_pruning(model, method, params, pruned_dir, gpu_id, logger)
            results['pruning_success'] = success
            if not success:
                return (method, False, results)

        # Step 2: 评估剪枝后模型
        eval_before_json = pruned_dir / 'evaluation' / 'evaluation_results.json'
        if skip_completed and eval_before_json.exists():
            log(f"[GPU {gpu_id}] ⊙ 跳过评估（已存在）: {model}/{method}")
            results['eval_before_success'] = True
        else:
            success = evaluate_model(pruned_model_path, eval_before_json, gpu_id, logger)
            results['eval_before_success'] = success
            if not success:
                return (method, False, results)

        # 读取评估结果
        if eval_before_json.exists():
            with open(eval_before_json, 'r') as f:
                eval_data = json.load(f)
                if 'metrics' in eval_data:
                    if 'ppl' in eval_data['metrics']:
                        results['ppl_before'] = eval_data['metrics']['ppl'].get('wikitext2 (wikitext-2-raw-v1)')
                    if 'avg_zeroshot_acc' in eval_data['metrics']:
                        results['acc_before'] = eval_data['metrics']['avg_zeroshot_acc']

        # Step 3: 微调
        finetuned_model_path = finetuned_dir / 'pruned_model.bin'
        if skip_completed and finetuned_model_path.exists():
            log(f"[GPU {gpu_id}] ⊙ 跳过微调（已存在）: {model}/{method}")
            results['finetune_success'] = True
        else:
            success = finetune_model(pruned_model_path, finetuned_dir, gpu_id, logger)
            results['finetune_success'] = success
            if not success:
                return (method, False, results)

        # Step 4: 评估微调后模型
        eval_after_json = finetuned_dir / 'evaluation_after_finetune' / 'evaluation_results.json'
        if skip_completed and eval_after_json.exists():
            log(f"[GPU {gpu_id}] ⊙ 跳过评估（已存在）: {model}/{method}")
            results['eval_after_success'] = True
        else:
            success = evaluate_model(finetuned_model_path, eval_after_json, gpu_id, logger)
            results['eval_after_success'] = success
            if not success:
                return (method, False, results)

        # 读取微调后评估结果
        if eval_after_json.exists():
            with open(eval_after_json, 'r') as f:
                eval_data = json.load(f)
                if 'metrics' in eval_data:
                    if 'ppl' in eval_data['metrics']:
                        results['ppl_after'] = eval_data['metrics']['ppl'].get('wikitext2 (wikitext-2-raw-v1)')
                    if 'avg_zeroshot_acc' in eval_data['metrics']:
                        results['acc_after'] = eval_data['metrics']['avg_zeroshot_acc']

        log(f"[GPU {gpu_id}] ✓ 完成: {model}/{method}")
        return (method, True, results)

    except Exception as e:
        log(f"[GPU {gpu_id}] ✗ 处理失败: {model}/{method} - {e}")
        return (method, False, results)


def export_to_excel(all_results, output_file, logger):
    """
    将结果导出到 Excel

    Args:
        all_results: 所有方法的结果列表
        output_file: 输出Excel文件路径
        logger: 日志对象
    """
    log = logger.info

    # 转换为 DataFrame
    df = pd.DataFrame(all_results)

    # 计算改进
    df['ppl_improvement'] = df.apply(
        lambda row: (row['ppl_before'] - row['ppl_after']) / row['ppl_before'] * 100
        if row['ppl_before'] and row['ppl_after'] else None,
        axis=1
    )
    df['acc_improvement'] = df.apply(
        lambda row: (row['acc_after'] - row['acc_before']) * 100
        if row['acc_before'] is not None and row['acc_after'] is not None else None,
        axis=1
    )

    # 重新排列列
    columns = [
        'model', 'method',
        'pruning_success', 'eval_before_success', 'finetune_success', 'eval_after_success',
        'ppl_before', 'ppl_after', 'ppl_improvement',
        'acc_before', 'acc_after', 'acc_improvement'
    ]
    df = df[columns]

    # 导出到 Excel
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)

        # 调整列宽
        worksheet = writer.sheets['Results']
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column].width = adjusted_width

    log(f"✓ 结果已导出到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='完整的剪枝-评估-微调-评估流程')

    # 必需参数
    parser.add_argument('--model', type=str, required=True,
                       choices=['Llama', 'Llama-Instruct', 'Mistral', 'Mistral-Instruct', 'Qwen', 'Qwen-Instruct'],
                       help='模型名称')
    parser.add_argument('--output_prefix', type=str, required=True,
                       help='输出目录前缀（例如: exp_128_128_8）')

    # 校准参数（完整版）
    parser.add_argument('--taylor_num_samples', type=int, help='Taylor 重要性计算的样本数')
    parser.add_argument('--taylor_seq_len', type=int, help='Taylor 重要性计算的序列长度')
    parser.add_argument('--layer_importance_num_samples', type=int, help='层重要性分析的样本数')
    parser.add_argument('--layer_importance_seq_len', type=int, help='层重要性分析的序列长度')
    parser.add_argument('--block_importance_num_samples', type=int, help='块重要性分析的样本数')
    parser.add_argument('--block_importance_seq_len', type=int, help='块重要性分析的序列长度')
    parser.add_argument('--gradient_batch_size', type=int, help='梯度计算的批次大小')

    # 校准参数（简化版 x1,x2,y1,y2,z）
    parser.add_argument('--x1', type=int, help='等同于 taylor_num_samples')
    parser.add_argument('--x2', type=int, help='等同于 taylor_seq_len')
    parser.add_argument('--y1', type=int, help='等同于 layer/block_importance_num_samples')
    parser.add_argument('--y2', type=int, help='等同于 layer/block_importance_seq_len')
    parser.add_argument('--z', type=int, help='等同于 gradient_batch_size')

    # 可选参数
    parser.add_argument('--methods', type=str, nargs='+',
                       help='指定要运行的方法（默认运行所有方法）')
    parser.add_argument('--skip-completed', action='store_true',
                       help='跳过已完成的步骤')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='并行使用的GPU数量')

    args = parser.parse_args()

    # 处理参数（简化版优先）
    params = {}
    if args.x1 is not None:
        params['x1'] = args.x1
    elif args.taylor_num_samples is not None:
        params['x1'] = args.taylor_num_samples
    else:
        parser.error("需要指定 --x1 或 --taylor_num_samples")

    if args.x2 is not None:
        params['x2'] = args.x2
    elif args.taylor_seq_len is not None:
        params['x2'] = args.taylor_seq_len
    else:
        parser.error("需要指定 --x2 或 --taylor_seq_len")

    if args.y1 is not None:
        params['y1'] = args.y1
    elif args.layer_importance_num_samples is not None:
        params['y1'] = args.layer_importance_num_samples
    else:
        parser.error("需要指定 --y1 或 --layer_importance_num_samples")

    if args.y2 is not None:
        params['y2'] = args.y2
    elif args.layer_importance_seq_len is not None:
        params['y2'] = args.layer_importance_seq_len
    else:
        parser.error("需要指定 --y2 或 --layer_importance_seq_len")

    if args.z is not None:
        params['z'] = args.z
    elif args.gradient_batch_size is not None:
        params['z'] = args.gradient_batch_size
    else:
        parser.error("需要指定 --z 或 --gradient_batch_size")

    # 设置日志
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'pipeline_{args.model}_{args.output_prefix}_{timestamp}.log'
    logger = setup_logging(log_file)

    logger.info(f"\n{'='*80}")
    logger.info(f"完整流程：剪枝 → 评估 → 微调 → 评估")
    logger.info(f"{'='*80}")
    logger.info(f"模型: {args.model}")
    logger.info(f"输出前缀: {args.output_prefix}")
    logger.info(f"校准参数: x1={params['x1']}, x2={params['x2']}, y1={params['y1']}, y2={params['y2']}, z={params['z']}")

    # 确定要运行的方法
    if args.methods:
        methods = args.methods
    else:
        methods = ['blockwise', 'magnitude', 'wanda', 'LLM-Pruner']
        for n_layers in SHORTGPT_LAYERS[args.model]:
            methods.append(f'ShortGPT_remove_{n_layers}')

    logger.info(f"将运行 {len(methods)} 个方法: {', '.join(methods)}")

    # 运行所有方法
    all_results = []

    if args.num_gpus > 1:
        # 并行模式（暂不实现，因为剪枝本身就占用大量资源）
        logger.warning("并行模式暂未实现，将使用顺序模式")
        args.num_gpus = 1

    # 顺序模式
    for method in methods:
        gpu_id = get_best_gpu()
        method_name, success, results = process_single_method(
            args.model, method, params, args.output_prefix,
            gpu_id, args.skip_completed, logger
        )
        all_results.append(results)

    # 导出到 Excel
    output_file = Path('results') / args.output_prefix / args.model / f'{args.model}_results.xlsx'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    export_to_excel(all_results, output_file, logger)

    # 打印摘要
    logger.info(f"\n{'='*80}")
    logger.info(f"处理完成")
    logger.info(f"{'='*80}")
    success_count = sum(1 for r in all_results if r['eval_after_success'])
    logger.info(f"成功: {success_count}/{len(methods)}")
    logger.info(f"结果已导出到: {output_file}")


if __name__ == '__main__':
    main()
