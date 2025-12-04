#!/usr/bin/env python3
"""
LoRA微调工作流管理脚本

用法:
    # 微调单个模型的某个配置（剪枝模型）
    python run_finetuning_workflow.py --model Llama --config best_acc --stage finetune

    # 微调单个模型的base配置（原始HF模型）
    python run_finetuning_workflow.py --model Llama --config base --stage finetune

    # 评估微调后的模型
    python run_finetuning_workflow.py --model Llama --config best_acc --stage evaluate

    # 完整流程（微调+评估）
    python run_finetuning_workflow.py --model Llama --config best_acc --stage all

    # 【新功能】处理单个模型的所有配置（自动跳过已完成）
    python run_finetuning_workflow.py --model Llama-Instruct --model-all --skip-completed --stage all

    # 【新功能】并行处理单个模型的所有配置
    python run_finetuning_workflow.py --model Qwen --model-all --skip-completed --num-gpus 4 --stage all

    # 批量处理所有模型和所有配置（顺序执行）
    # 注意：现在包含8-9个配置类型（根据模型不同），共约48个任务
    #      - 6个通用配置: best_acc, best_ppl, base, wanda, magnitude, LLMPruner
    #      - 2个ShortGPT配置 (Llama/Llama-Instruct: remove_7/8; 其他: remove_6/7)
    python run_finetuning_workflow.py --batch-all --stage all

    # 批量处理所有模型（并行使用4个GPU，跳过已完成）
    python run_finetuning_workflow.py --batch-all --skip-completed --stage finetune --num-gpus 4

    # 评估所有base模型（原始未剪枝模型，用于对比）
    python run_finetuning_workflow.py --evaluate-base --num-gpus 4

    # 说明：
    # - 配置类型:
    #   * best_acc: 准确率最优的剪枝配置
    #   * best_ppl: PPL最优的剪枝配置
    #   * base: 原始HF模型（未剪枝）
    #   * wanda: Wanda剪枝方法
    #   * magnitude: Magnitude剪枝方法
    #   * LLMPruner: LLM-Pruner baseline
    #   * ShortGPT_remove_6/7/8: ShortGPT baseline (不同层数)
    # - 每个模型在单张卡上训练（不是分布式训练）
    # - --num-gpus 指定同时运行的任务数，每个任务占用1张卡
    # - 自动选择显存最大的N张GPU
    # - 任务会轮询分配到不同的GPU上
    # - --evaluate-base 评估原始base模型，结果保存到 results/base_evaluation/
    # - base配置的微调结果保存到 results/finetuned/{model}/base_finetuned/
"""

import argparse
import json
import subprocess
import os
from pathlib import Path
import sys
import time
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.utils.get_best_gpu import get_best_gpu


# Base模型路径映射
BASE_MODEL_PATHS = {
    'Llama': '/newdata/LLMs/Llama-3-8B',
    'Llama-Instruct': '/newdata/LLMs/Llama-3-8B-Instruct',
    'Qwen': '/newdata/LLMs/Qwen2.5-7B',
    'Qwen-Instruct': '/newdata/LLMs/Qwen2.5-7B-Instruct',
    'Mistral': '/newdata/LLMs/Mistral-7B-v0.3',
    'Mistral-Instruct': '/newdata/LLMs/Mistral-7B-Instruct-v0.3'
}


def setup_logging(log_file=None):
    """
    设置日志系统，同时输出到终端和文件

    Args:
        log_file: 日志文件路径，如果为None则只输出到终端
    """
    # 创建logger
    logger = logging.getLogger('finetuning_workflow')
    logger.setLevel(logging.INFO)

    # 清除已有的handlers（避免重复）
    logger.handlers.clear()

    # 格式化
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


def check_task_completed(model, config_type, stage='all'):
    """
    检查任务是否已完成

    Args:
        model: 模型名称
        config_type: 配置类型
        stage: 执行阶段 ('finetune', 'evaluate', 'compare', 'all')

    Returns:
        bool: 是否已完成
    """
    finetuned_dir = Path('results') / 'finetuned' / model / f'{config_type}_finetuned'
    eval_dir = Path('results') / 'finetuned_evaluation' / model / f'{config_type}_finetuned'

    # 检查微调是否完成
    finetune_completed = False
    if stage in ['finetune', 'all']:
        # 检查是否存在 pruned_model.bin（微调完成的标志）
        model_file = finetuned_dir / 'pruned_model.bin'
        # 或者检查是否存在 adapter_model.safetensors（LoRA权重）
        lora_file = finetuned_dir / 'adapter_model.safetensors'
        finetune_completed = model_file.exists() or lora_file.exists()

    # 检查评估是否完成
    evaluate_completed = False
    if stage in ['evaluate', 'all']:
        eval_result = eval_dir / 'evaluation_results.json'
        evaluate_completed = eval_result.exists()

    # 检查对比是否完成
    compare_completed = False
    if stage in ['compare', 'all']:
        compare_report = eval_dir / 'comparison_report.txt'
        compare_completed = compare_report.exists()

    # 根据stage判断是否完成
    if stage == 'finetune':
        return finetune_completed
    elif stage == 'evaluate':
        return evaluate_completed
    elif stage == 'compare':
        return compare_completed
    elif stage == 'all':
        # all阶段需要所有步骤都完成
        return finetune_completed and evaluate_completed and compare_completed

    return False


def get_all_configs_for_model(model):
    """
    获取某个模型的所有配置列表

    Args:
        model: 模型名称

    Returns:
        list: 配置列表
    """
    # 基础配置
    configs = ['best_acc', 'best_ppl', 'base', 'wanda', 'magnitude', 'LLMPruner']

    # ShortGPT根据模型类型有不同版本
    shortgpt_configs = {
        'Llama': ['ShortGPT_remove_7', 'ShortGPT_remove_8'],
        'Llama-Instruct': ['ShortGPT_remove_7', 'ShortGPT_remove_8'],
        'Mistral': ['ShortGPT_remove_6', 'ShortGPT_remove_7'],
        'Mistral-Instruct': ['ShortGPT_remove_6', 'ShortGPT_remove_7'],
        'Qwen': ['ShortGPT_remove_6', 'ShortGPT_remove_7'],
        'Qwen-Instruct': ['ShortGPT_remove_6', 'ShortGPT_remove_7']
    }

    # 添加该模型的ShortGPT配置
    configs.extend(shortgpt_configs.get(model, []))

    return configs


def get_best_gpus(num_gpus):
    """
    获取N个剩余显存最大的GPU

    Args:
        num_gpus: 需要的GPU数量

    Returns:
        list: GPU ID列表
    """
    try:
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

        # 按剩余显存排序，选择前N个
        gpu_memory.sort(key=lambda x: x[1], reverse=True)
        best_gpus = [gpu_id for gpu_id, _ in gpu_memory[:num_gpus]]

        print(f"✓ 选择了 {len(best_gpus)} 个GPU: {best_gpus}")
        for gpu_id, free_mem in gpu_memory[:num_gpus]:
            print(f"  GPU {gpu_id}: 剩余显存 {free_mem/1024:.2f} GB")

        return best_gpus

    except Exception as e:
        print(f"警告: 无法获取GPU信息 ({e})，使用默认GPU列表")
        return list(range(num_gpus))


class FinetuningWorkflow:
    def __init__(self, model, config_type):
        self.model = model
        self.config_type = config_type  # 'best_acc', 'best_ppl', or 'base'

        # 路径设置
        if config_type == 'base':
            # Base模型使用HF格式路径
            self.base_model_path = BASE_MODEL_PATHS[model]
            self.pruned_dir = None  # Base模型没有pruned_dir
        else:
            # 剪枝模型路径
            self.pruned_dir = Path('results') / 'for_finetuning' / model / config_type
            self.base_model_path = None

        self.finetuned_dir = Path('results') / 'finetuned' / model / f'{config_type}_finetuned'
        self.eval_dir = Path('results') / 'finetuned_evaluation' / model / f'{config_type}_finetuned'

        # 加载选择信息（base模型没有）
        self.selection_info = self.load_selection_info() if config_type != 'base' else None

    def load_selection_info(self):
        """加载模型选择信息（仅用于best_acc/best_ppl）"""
        info_file = self.pruned_dir / 'selection_info.json'
        if not info_file.exists():
            # 对于baseline方法（wanda, magnitude, LLMPruner, ShortGPT），
            # 没有selection_info.json是正常的，返回基本信息
            return {
                'selection_criterion': self.config_type,
                'pruning_method': self.config_type,
                'model': self.model
            }

        with open(info_file, 'r') as f:
            return json.load(f)

    def finetune(self, lora_config=None, gpu_id=None):
        """运行LoRA微调

        Args:
            lora_config: LoRA配置字典
            gpu_id: 指定使用的GPU ID，如果为None则自动选择
        """
        print(f"\n{'='*80}")
        print(f"开始微调: {self.model} - {self.config_type}")
        print(f"{'='*80}")

        # 默认LoRA配置
        if lora_config is None:
            # 为了实验公平性，所有配置使用相同的micro_batch_size=1
            # 这样gradient_accumulation_steps对所有模型都是64步
            # 虽然剪枝模型可以用更大的batch，但为了可比性统一配置
            lora_config = {
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'num_epochs': 2,
                'learning_rate': 1e-4,
                'batch_size': 64,
                'micro_batch_size': 4  # 统一使用以保证实验公平性
            }

        print(f"\nLoRA配置:")
        for k, v in lora_config.items():
            print(f"  {k}: {v}")

        if self.config_type == 'base':
            print(f"\nBase模型路径: {self.base_model_path}")
        else:
            print(f"\n剪枝模型目录: {self.pruned_dir}")
        print(f"微调输出目录: {self.finetuned_dir}")

        # 创建输出目录
        self.finetuned_dir.mkdir(parents=True, exist_ok=True)

        # 保存微调配置
        config_file = self.finetuned_dir / 'finetuning_config.json'
        config_data = {
            'model': self.model,
            'config_type': self.config_type,
            'lora_config': lora_config
        }
        if self.config_type == 'base':
            config_data['base_model_path'] = self.base_model_path
        else:
            config_data['pruned_model_info'] = self.selection_info

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"\n微调配置已保存: {config_file}")

        # 获取GPU ID（如果未指定）
        if gpu_id is None:
            gpu_id = get_best_gpu()

        # 设置环境变量，确保LoRA微调只使用单GPU（避免数据分布在多GPU上出错）
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"\n使用GPU: {gpu_id}")

        # 检查是否存在检查点（自动恢复训练）
        checkpoint_dir = None
        if self.finetuned_dir.exists():
            # 查找最新的检查点目录 (checkpoint-xxx 格式)
            checkpoints = sorted([
                d for d in self.finetuned_dir.iterdir()
                if d.is_dir() and d.name.startswith('checkpoint-')
            ], key=lambda x: int(x.name.split('-')[1]))

            if checkpoints:
                checkpoint_dir = checkpoints[-1]  # 使用最新的检查点
                print(f"\n✓ 发现检查点: {checkpoint_dir}")
                print(f"  将从检查点恢复训练...")

        # 构建微调命令
        cmd = ['python', 'finetune_lora.py']

        # 根据config_type选择模型参数
        if self.config_type == 'base':
            # Base模型使用--base_model参数
            cmd.extend(['--base_model', self.base_model_path])
        else:
            # 剪枝模型使用--pruned_model参数
            cmd.extend(['--pruned_model', str(self.pruned_dir / 'pruned_model.bin')])

        # 添加其他参数
        cmd.extend([
            '--output_dir', str(self.finetuned_dir),
            '--lora_r', str(lora_config['lora_r']),
            '--lora_alpha', str(lora_config['lora_alpha']),
            '--lora_dropout', str(lora_config['lora_dropout']),
            '--num_epochs', str(lora_config['num_epochs']),
            '--learning_rate', str(lora_config['learning_rate']),
            '--batch_size', str(lora_config['batch_size']),
            '--micro_batch_size', str(lora_config['micro_batch_size'])
        ])

        # 为了实验公平性，所有配置都启用梯度检查点
        # 虽然剪枝模型可以不用，但为了保证训练过程一致性，统一启用
        # gradient_checkpointing在数学上等价，只影响计算方式（节省显存但略慢）
        cmd.append('--gradient_checkpointing')

        # 如果存在检查点，添加恢复参数
        if checkpoint_dir:
            cmd.extend(['--resume_from_checkpoint', str(checkpoint_dir)])

        print(f"\n执行命令: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")
        print(f"\n⚠️  注意：请确保 finetune_lora.py 存在并且参数正确")
        print(f"如果需要，请修改此脚本中的命令构建逻辑")

        # 运行微调
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"\n✓ 微调完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ 微调失败: {e}")
            return False


    def evaluate(self, gpu_id=None):
        """评估微调后的模型

        Args:
            gpu_id: 指定使用的GPU ID，如果为None则自动选择
        """
        print(f"\n{'='*80}")
        print(f"评估微调后模型: {self.model} - {self.config_type}")
        print(f"{'='*80}")

        print(f"\n微调模型目录: {self.finetuned_dir}")
        print(f"评估输出目录: {self.eval_dir}")

        # 创建评估输出目录
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # 获取GPU ID（如果未指定）
        if gpu_id is None:
            gpu_id = get_best_gpu()

        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"\n使用GPU: {gpu_id}")

        # 微调后的模型bin文件路径
        finetuned_model_bin = self.finetuned_dir / 'pruned_model.bin'

        # 评估结果JSON文件路径
        eval_output_json = self.eval_dir / 'evaluation_results.json'

        # 构建评估命令
        # 注意：
        # 1. CUDA_VISIBLE_DEVICES 限制评估脚本只能看到指定的GPU
        # 2. 评估脚本的 --auto_select_gpu 默认为 True，会自动选择剩余显存最大的GPU
        # 3. 由于只有1个GPU可见，auto_select_gpu 会自动选择 cuda:0
        # 4. 不需要传递 --device 参数，因为会被 auto_select_gpu 覆盖
        cmd = [
            'python', 'evaluation/run_evaluation.py',
            '--model_path', str(finetuned_model_bin),  # 指定bin文件路径
            '--output', str(eval_output_json),  # 指定输出JSON文件
            '--metrics', 'ppl,zeroshot'  # 只评估PPL和zero-shot任务
            # 不传递 --device，让 auto_select_gpu 在受限的GPU列表中自动选择
        ]

        print(f"\n执行命令: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")

        # 运行评估
        try:
            subprocess.run(cmd, env=env, check=True)
            print(f"\n✓ 评估完成")
            print(f"✓ 评估结果已保存: {eval_output_json}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ 评估失败: {e}")
            return False


    def compare_results(self):
        """对比微调前后的结果"""
        print(f"\n{'='*80}")
        print(f"对比微调前后结果: {self.model} - {self.config_type}")
        print(f"{'='*80}")

        # 读取微调前的结果
        if self.config_type == 'base':
            # Base模型的微调前结果在base_evaluation目录
            before_eval = Path('results') / 'base_evaluation' / self.model / 'evaluation_results.json'
        else:
            # 剪枝模型的微调前结果在pruned_dir
            before_eval = self.pruned_dir / 'evaluation' / 'evaluation_results.json'

        if not before_eval.exists():
            print(f"⚠️  找不到微调前的评估结果: {before_eval}")
            if self.config_type == 'base':
                print(f"提示: 请先运行 python {sys.argv[0]} --evaluate-base 评估base模型")
            return

        # 读取微调后的结果
        after_eval = self.eval_dir / 'evaluation_results.json'
        if not after_eval.exists():
            print(f"⚠️  找不到微调后的评估结果")
            print(f"请先运行评估: python {sys.argv[0]} --model {self.model} --config {self.config_type} --stage evaluate")
            return

        with open(before_eval, 'r') as f:
            before_data = json.load(f)

        with open(after_eval, 'r') as f:
            after_data = json.load(f)

        # 生成对比报告
        report = self.generate_comparison_report(before_data, after_data)

        # 保存报告
        report_file = self.eval_dir / 'comparison_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)

        print(report)
        print(f"\n✓ 对比报告已保存: {report_file}")

    def generate_comparison_report(self, before, after):
        """生成对比报告"""
        report = []
        report.append(f"="*80)
        report.append(f"微调前后性能对比")
        report.append(f"="*80)
        report.append(f"\n模型: {self.model}")

        if self.config_type == 'base':
            report.append(f"配置: {self.config_type} (原始HF模型)")
        else:
            report.append(f"配置: {self.config_type} ({self.selection_info['selection_criterion']})")
            report.append(f"剪枝方法: {self.selection_info['pruning_method']}")

        # PPL对比
        if 'metrics' in before and 'ppl' in before['metrics']:
            before_ppl = before['metrics']['ppl']
            after_ppl = after['metrics']['ppl'] if 'metrics' in after and 'ppl' in after['metrics'] else None

            report.append(f"\n{'-'*80}")
            report.append(f"PPL (WikiText2):")
            report.append(f"  微调前: {before_ppl.get('wikitext2 (wikitext-2-raw-v1)', 'N/A'):.2f}")
            if after_ppl:
                report.append(f"  微调后: {after_ppl.get('wikitext2 (wikitext-2-raw-v1)', 'N/A'):.2f}")
                before_val = before_ppl.get('wikitext2 (wikitext-2-raw-v1)', 0)
                after_val = after_ppl.get('wikitext2 (wikitext-2-raw-v1)', 0)
                if before_val and after_val:
                    diff = after_val - before_val
                    pct = (diff / before_val) * 100
                    report.append(f"  变化: {diff:+.2f} ({pct:+.2f}%)")

        # Zero-shot ACC对比
        if 'metrics' in before and 'zeroshot' in before['metrics']:
            report.append(f"\n{'-'*80}")
            report.append(f"Zero-shot ACC:")

            tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
            before_zs = before['metrics']['zeroshot']
            after_zs = after['metrics'].get('zeroshot', {}) if 'metrics' in after else {}

            before_accs = []
            after_accs = []

            for task in tasks:
                if task in before_zs:
                    before_acc = before_zs[task].get('accuracy', 0)
                    before_accs.append(before_acc)

                    after_acc = after_zs.get(task, {}).get('accuracy', 0) if after_zs else 0
                    after_accs.append(after_acc)

                    diff = after_acc - before_acc if after_acc else 0

                    report.append(f"  {task:15s}: {before_acc:.4f} → {after_acc:.4f} ({diff:+.4f})")

            if before_accs:
                before_mean = sum(before_accs) / len(before_accs)
                after_mean = sum(after_accs) / len(after_accs) if after_accs else 0
                diff_mean = after_mean - before_mean

                report.append(f"\n  {'平均':<15s}: {before_mean:.4f} → {after_mean:.4f} ({diff_mean:+.4f})")

        report.append(f"\n{'='*80}")

        return '\n'.join(report)


def evaluate_base_model(model, gpu_id=None):
    """
    评估base模型（原始未剪枝模型）

    Args:
        model: 模型名称
        gpu_id: 指定使用的GPU ID，如果为None则自动选择

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*80}")
    print(f"评估Base模型: {model}")
    print(f"{'='*80}")

    # 获取模型路径
    if model not in BASE_MODEL_PATHS:
        print(f"✗ 错误: 未知模型 {model}")
        return False

    model_path = BASE_MODEL_PATHS[model]
    if not Path(model_path).exists():
        print(f"✗ 错误: 模型路径不存在: {model_path}")
        return False

    print(f"\n模型路径: {model_path}")

    # 设置输出目录
    output_dir = Path('results') / 'base_evaluation' / model
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评估结果JSON文件路径
    eval_output_json = output_dir / 'evaluation_results.json'

    print(f"评估输出: {eval_output_json}")

    # 获取GPU ID（如果未指定）
    if gpu_id is None:
        gpu_id = get_best_gpu()

    # 设置环境变量
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"\n使用GPU: {gpu_id}")

    # 构建评估命令
    cmd = [
        'python', 'evaluation/run_evaluation.py',
        '--model_path', model_path,
        '--output', str(eval_output_json),
        '--metrics', 'ppl,zeroshot'
    ]

    print(f"\n执行命令: CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}")

    # 运行评估
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"\n✓ {model} 评估完成")
        print(f"✓ 评估结果已保存: {eval_output_json}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {model} 评估失败: {e}")
        return False


def run_single_task(model, config, stage, gpu_id, skip_completed=False, logger=None):
    """运行单个任务（在指定GPU上）

    Args:
        model: 模型名称
        config: 配置类型
        stage: 执行阶段
        gpu_id: GPU ID
        skip_completed: 是否跳过已完成的任务
        logger: 日志对象
    """
    task_name = f"{model}/{config}"
    log_func = logger.info if logger else print

    try:
        # 检查任务是否已完成
        if skip_completed and check_task_completed(model, config, stage):
            log_func(f"[GPU {gpu_id}] ⊙ 跳过已完成: {task_name}")
            return (task_name, True, "已完成（跳过）")

        log_func(f"[GPU {gpu_id}] 开始处理: {task_name}")
        workflow = FinetuningWorkflow(model, config)

        if stage in ['finetune', 'all']:
            success = workflow.finetune(gpu_id=gpu_id)
            if not success:
                return (task_name, False, "微调失败")

        if stage in ['evaluate', 'all']:
            success = workflow.evaluate(gpu_id=gpu_id)
            if not success:
                return (task_name, False, "评估失败")

        if stage in ['compare', 'all']:
            workflow.compare_results()

        log_func(f"[GPU {gpu_id}] ✓ 完成: {task_name}")
        return (task_name, True, "成功")

    except Exception as e:
        log_func(f"[GPU {gpu_id}] ✗ 处理 {task_name} 时出错: {e}")
        return (task_name, False, str(e))


def run_base_eval_task(model, gpu_id):
    """运行base模型评估任务（在指定GPU上）"""
    try:
        print(f"\n[GPU {gpu_id}] 开始评估base模型: {model}")
        success = evaluate_base_model(model, gpu_id=gpu_id)

        if success:
            print(f"\n[GPU {gpu_id}] ✓ 完成: {model}")
            return (model, True, "成功")
        else:
            return (model, False, "评估失败")

    except Exception as e:
        print(f"\n[GPU {gpu_id}] ✗ 评估 {model} 时出错: {e}")
        return (model, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='LoRA微调工作流管理')
    parser.add_argument('--model', type=str,
                       choices=['Llama', 'Llama-Instruct', 'Qwen', 'Qwen-Instruct', 'Mistral', 'Mistral-Instruct'],
                       help='模型名称')
    parser.add_argument('--config', type=str,
                       choices=['best_acc', 'best_ppl', 'base', 'wanda', 'magnitude',
                               'LLMPruner', 'ShortGPT_remove_6', 'ShortGPT_remove_7', 'ShortGPT_remove_8'],
                       help='配置类型 (best_acc/best_ppl/wanda/magnitude/LLMPruner/ShortGPT_*为剪枝模型，base为原始HF模型)')
    parser.add_argument('--stage', type=str, choices=['finetune', 'evaluate', 'compare', 'all', 'evaluate_base'],
                       default='all', help='执行阶段')
    parser.add_argument('--batch-all', action='store_true',
                       help='批量处理所有模型和配置')
    parser.add_argument('--model-all', action='store_true',
                       help='处理指定模型的所有配置（需配合 --model 使用）')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='并行使用的GPU数量（默认1，顺序执行）')
    parser.add_argument('--evaluate-base', action='store_true',
                       help='评估base模型（原始未剪枝模型）')
    parser.add_argument('--skip-completed', action='store_true',
                       help='跳过已完成的任务（自动检测）')
    parser.add_argument('--log-file', type=str, default=None,
                       help='日志文件路径（默认: logs/finetuning_workflow_YYYYMMDD_HHMMSS.log）')

    args = parser.parse_args()

    # 设置日志
    if args.log_file is None and (args.batch_all or args.model_all or args.evaluate_base):
        # 批量模式自动创建日志文件
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_file = str(log_dir / f'finetuning_workflow_{timestamp}.log')

    logger = setup_logging(args.log_file)

    # 处理evaluate_base选项
    if args.evaluate_base or args.stage == 'evaluate_base':
        models = ['Llama', 'Llama-Instruct', 'Qwen', 'Qwen-Instruct', 'Mistral', 'Mistral-Instruct']

        print(f"\n{'='*80}")
        print(f"评估Base模型")
        print(f"{'='*80}")
        print(f"\n将评估 {len(models)} 个Base模型")

        if args.num_gpus > 1:
            # 并行模式
            print(f"\n并行模式: 使用 {args.num_gpus} 个GPU")

            # 获取最佳的N个GPU
            gpu_ids = get_best_gpus(args.num_gpus)

            # 使用线程池并行执行
            results = []
            with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
                # 提交所有任务
                future_to_model = {}
                for i, model in enumerate(models):
                    # 轮询分配GPU
                    gpu_id = gpu_ids[i % len(gpu_ids)]
                    future = executor.submit(run_base_eval_task, model, gpu_id)
                    future_to_model[future] = model

                # 等待任务完成
                for future in as_completed(future_to_model):
                    model_name, success, msg = future.result()
                    results.append((model_name, success, msg))

            # 打印结果摘要
            print(f"\n{'='*80}")
            print(f"批量评估完成")
            print(f"{'='*80}")
            success_count = sum(1 for _, success, _ in results if success)
            print(f"\n成功: {success_count}/{len(results)}")

            if success_count < len(results):
                print("\n失败的模型:")
                for model_name, success, msg in results:
                    if not success:
                        print(f"  ✗ {model_name}: {msg}")
        else:
            # 顺序模式
            print(f"\n顺序模式: 逐个评估")
            success_count = 0
            for model in models:
                try:
                    if evaluate_base_model(model):
                        success_count += 1
                except Exception as e:
                    print(f"\n✗ 评估 {model} 时出错: {e}")
                    continue

            print(f"\n✓ 批量评估完成")
            print(f"成功: {success_count}/{len(models)}")

        return

    # 处理 model-all 选项：处理指定模型的所有配置
    if args.model_all:
        if not args.model:
            parser.error("使用 --model-all 需要同时指定 --model")

        configs = get_all_configs_for_model(args.model)

        logger.info(f"\n{'='*80}")
        logger.info(f"处理模型所有配置: {args.model}")
        logger.info(f"{'='*80}")
        logger.info(f"\n将处理 {len(configs)} 个配置: {', '.join(configs)}")
        if args.skip_completed:
            logger.info("启用跳过已完成任务功能")

        # 构建任务列表
        tasks = [(args.model, config) for config in configs]

        if args.num_gpus > 1:
            # 并行模式
            logger.info(f"\n并行模式: 使用 {args.num_gpus} 个GPU")

            # 获取最佳的N个GPU
            gpu_ids = get_best_gpus(args.num_gpus)

            # 使用线程池并行执行
            results = []
            with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
                # 提交所有任务
                future_to_task = {}
                for i, (model, config) in enumerate(tasks):
                    # 轮询分配GPU
                    gpu_id = gpu_ids[i % len(gpu_ids)]
                    future = executor.submit(run_single_task, model, config, args.stage, gpu_id,
                                           args.skip_completed, logger)
                    future_to_task[future] = (model, config)

                # 等待任务完成
                for future in as_completed(future_to_task):
                    task_name, success, msg = future.result()
                    results.append((task_name, success, msg))

            # 打印结果摘要
            logger.info(f"\n{'='*80}")
            logger.info(f"处理完成")
            logger.info(f"{'='*80}")
            success_count = sum(1 for _, success, _ in results if success)
            logger.info(f"\n成功: {success_count}/{len(results)}")

            # 统计跳过的任务
            skipped_count = sum(1 for _, success, msg in results if success and "跳过" in msg)
            if skipped_count > 0:
                logger.info(f"跳过: {skipped_count}")

            if success_count < len(results):
                logger.info("\n失败的任务:")
                for task_name, success, msg in results:
                    if not success:
                        logger.info(f"  ✗ {task_name}: {msg}")
        else:
            # 顺序模式
            logger.info(f"\n顺序模式: 逐个处理")
            success_count = 0
            skipped_count = 0

            for model, config in tasks:
                try:
                    # 检查是否跳过
                    if args.skip_completed and check_task_completed(model, config, args.stage):
                        logger.info(f"⊙ 跳过已完成: {model}/{config}")
                        skipped_count += 1
                        success_count += 1
                        continue

                    workflow = FinetuningWorkflow(model, config)

                    if args.stage in ['finetune', 'all']:
                        workflow.finetune()

                    if args.stage in ['evaluate', 'all']:
                        workflow.evaluate()

                    if args.stage in ['compare', 'all']:
                        workflow.compare_results()

                    success_count += 1

                except Exception as e:
                    logger.error(f"\n✗ 处理 {model}/{config} 时出错: {e}")
                    continue

            logger.info(f"\n✓ 处理完成")
            logger.info(f"成功: {success_count}/{len(tasks)}")
            if skipped_count > 0:
                logger.info(f"跳过: {skipped_count}")

        return

    if args.batch_all:
        # 批量处理
        models = ['Llama', 'Llama-Instruct', 'Qwen', 'Qwen-Instruct', 'Mistral', 'Mistral-Instruct']
        # 包含所有配置类型：best系列、base、经典baseline
        configs = ['best_acc', 'best_ppl', 'base', 'wanda', 'magnitude', 'LLMPruner']

        # ShortGPT根据模型类型有不同版本
        # Llama/Llama-Instruct: remove_7, remove_8
        # Mistral/Mistral-Instruct/Qwen/Qwen-Instruct: remove_6, remove_7
        shortgpt_configs = {
            'Llama': ['ShortGPT_remove_7', 'ShortGPT_remove_8'],
            'Llama-Instruct': ['ShortGPT_remove_7', 'ShortGPT_remove_8'],
            'Mistral': ['ShortGPT_remove_6', 'ShortGPT_remove_7'],
            'Mistral-Instruct': ['ShortGPT_remove_6', 'ShortGPT_remove_7'],
            'Qwen': ['ShortGPT_remove_6', 'ShortGPT_remove_7'],
            'Qwen-Instruct': ['ShortGPT_remove_6', 'ShortGPT_remove_7']
        }

        print(f"\n{'='*80}")
        print(f"批量处理所有模型")
        print(f"{'='*80}")

        # 构建任务列表（根据模型类型添加ShortGPT配置）
        tasks = []
        for model in models:
            for config in configs:
                tasks.append((model, config))
            # 添加该模型的ShortGPT配置
            for config in shortgpt_configs[model]:
                tasks.append((model, config))

        total_tasks = len(tasks)
        print(f"\n将处理 {total_tasks} 个配置")
        print(f"  - 每个模型 {len(configs)} 个通用配置 + {len(shortgpt_configs[models[0]])} 个ShortGPT配置")

        if args.num_gpus > 1:
            # 并行模式
            logger.info(f"\n并行模式: 使用 {args.num_gpus} 个GPU")
            if args.skip_completed:
                logger.info("启用跳过已完成任务功能")

            # 获取最佳的N个GPU
            gpu_ids = get_best_gpus(args.num_gpus)

            # 使用线程池并行执行
            results = []
            with ThreadPoolExecutor(max_workers=args.num_gpus) as executor:
                # 提交所有任务
                future_to_task = {}
                for i, (model, config) in enumerate(tasks):
                    # 轮询分配GPU
                    gpu_id = gpu_ids[i % len(gpu_ids)]
                    future = executor.submit(run_single_task, model, config, args.stage, gpu_id,
                                           args.skip_completed, logger)
                    future_to_task[future] = (model, config)

                # 等待任务完成
                for future in as_completed(future_to_task):
                    task_name, success, msg = future.result()
                    results.append((task_name, success, msg))

            # 打印结果摘要
            logger.info(f"\n{'='*80}")
            logger.info(f"批量处理完成")
            logger.info(f"{'='*80}")
            success_count = sum(1 for _, success, _ in results if success)
            logger.info(f"\n成功: {success_count}/{len(results)}")

            # 统计跳过的任务
            skipped_count = sum(1 for _, success, msg in results if success and "跳过" in msg)
            if skipped_count > 0:
                logger.info(f"跳过: {skipped_count}")

            if success_count < len(results):
                logger.info("\n失败的任务:")
                for task_name, success, msg in results:
                    if not success:
                        logger.info(f"  ✗ {task_name}: {msg}")
        else:
            # 顺序模式
            logger.info(f"\n顺序模式: 逐个处理")
            if args.skip_completed:
                logger.info("启用跳过已完成任务功能")

            success_count = 0
            skipped_count = 0

            for model, config in tasks:
                try:
                    # 检查是否跳过
                    if args.skip_completed and check_task_completed(model, config, args.stage):
                        logger.info(f"⊙ 跳过已完成: {model}/{config}")
                        skipped_count += 1
                        success_count += 1
                        continue

                    workflow = FinetuningWorkflow(model, config)

                    if args.stage in ['finetune', 'all']:
                        workflow.finetune()

                    if args.stage in ['evaluate', 'all']:
                        workflow.evaluate()

                    if args.stage in ['compare', 'all']:
                        workflow.compare_results()

                    success_count += 1

                except Exception as e:
                    logger.error(f"\n✗ 处理 {model}/{config} 时出错: {e}")
                    continue

            logger.info(f"\n✓ 批量处理完成")
            logger.info(f"成功: {success_count}/{len(tasks)}")
            if skipped_count > 0:
                logger.info(f"跳过: {skipped_count}")
    else:
        # 单个处理
        if not args.model or not args.config:
            parser.error("需要指定 --model 和 --config，或使用 --batch-all")

        workflow = FinetuningWorkflow(args.model, args.config)

        if args.stage in ['finetune', 'all']:
            workflow.finetune()

        if args.stage in ['evaluate', 'all']:
            workflow.evaluate()

        if args.stage in ['compare', 'all']:
            workflow.compare_results()


if __name__ == '__main__':
    main()
