#!/usr/bin/env python3
"""
LoRA微调工作流管理脚本

用法:
    # 微调单个模型的某个配置
    python run_finetuning_workflow.py --model Llama --config best_acc --stage finetune

    # 评估微调后的模型
    python run_finetuning_workflow.py --model Llama --config best_acc --stage evaluate

    # 完整流程（微调+评估）
    python run_finetuning_workflow.py --model Llama --config best_acc --stage all

    # 批量处理所有模型（顺序执行）
    python run_finetuning_workflow.py --batch-all --stage all

    # 批量处理所有模型（并行使用4个GPU）
    python run_finetuning_workflow.py --batch-all --stage finetune --num-gpus 4

    # 说明：
    # - 每个模型在单张卡上训练（不是分布式训练）
    # - --num-gpus 指定同时运行的任务数，每个任务占用1张卡
    # - 自动选择显存最大的N张GPU
    # - 任务会轮询分配到不同的GPU上
"""

import argparse
import json
import subprocess
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.utils.get_best_gpu import get_best_gpu


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
        self.config_type = config_type  # 'best_acc' or 'best_ppl'

        # 路径设置
        self.pruned_dir = Path('results') / 'for_finetuning' / model / config_type
        self.finetuned_dir = Path('results') / 'finetuned' / model / f'{config_type}_finetuned'
        self.eval_dir = Path('results') / 'finetuned_evaluation' / model / f'{config_type}_finetuned'

        # 加载选择信息
        self.selection_info = self.load_selection_info()

    def load_selection_info(self):
        """加载模型选择信息"""
        info_file = self.pruned_dir / 'selection_info.json'
        if not info_file.exists():
            raise FileNotFoundError(f"选择信息文件不存在: {info_file}")

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
            lora_config = {
                'lora_r': 8,
                'lora_alpha': 16,
                'lora_dropout': 0.05,
                'num_epochs': 2,
                'learning_rate': 1e-4,
                'batch_size': 64,
                'micro_batch_size': 4
            }

        print(f"\nLoRA配置:")
        for k, v in lora_config.items():
            print(f"  {k}: {v}")

        print(f"\n剪枝模型目录: {self.pruned_dir}")
        print(f"微调输出目录: {self.finetuned_dir}")

        # 创建输出目录
        self.finetuned_dir.mkdir(parents=True, exist_ok=True)

        # 保存微调配置
        config_file = self.finetuned_dir / 'finetuning_config.json'
        with open(config_file, 'w') as f:
            json.dump({
                'model': self.model,
                'config_type': self.config_type,
                'pruned_model_info': self.selection_info,
                'lora_config': lora_config
            }, f, indent=2)

        print(f"\n微调配置已保存: {config_file}")

        # 获取GPU ID（如果未指定）
        if gpu_id is None:
            gpu_id = get_best_gpu()

        # 设置环境变量，确保LoRA微调只使用单GPU（避免数据分布在多GPU上出错）
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"\n使用GPU: {gpu_id}")

        # 构建微调命令
        cmd = [
            'python', 'finetune_lora.py',
            '--pruned_model', str(self.pruned_dir / 'pruned_model.bin'),
            '--output_dir', str(self.finetuned_dir),
            '--lora_r', str(lora_config['lora_r']),
            '--lora_alpha', str(lora_config['lora_alpha']),
            '--lora_dropout', str(lora_config['lora_dropout']),
            '--num_epochs', str(lora_config['num_epochs']),
            '--learning_rate', str(lora_config['learning_rate']),
            '--batch_size', str(lora_config['batch_size']),
            '--micro_batch_size', str(lora_config['micro_batch_size'])
        ]

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
        cmd = [
            'python', 'evaluation/run_evaluation.py',
            '--model_path', str(finetuned_model_bin),  # 指定bin文件路径
            '--output', str(eval_output_json),  # 指定输出JSON文件
            '--metrics', 'ppl,zeroshot',  # 只评估PPL和zero-shot任务
            '--auto_select_gpu'  # 自动选择GPU（会使用CUDA_VISIBLE_DEVICES）
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
        before_eval = self.pruned_dir / 'evaluation' / 'evaluation_results.json'
        if not before_eval.exists():
            print(f"⚠️  找不到微调前的评估结果")
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


def run_single_task(model, config, stage, gpu_id):
    """运行单个任务（在指定GPU上）"""
    task_name = f"{model}/{config}"
    try:
        print(f"\n[GPU {gpu_id}] 开始处理: {task_name}")
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

        print(f"\n[GPU {gpu_id}] ✓ 完成: {task_name}")
        return (task_name, True, "成功")

    except Exception as e:
        print(f"\n[GPU {gpu_id}] ✗ 处理 {task_name} 时出错: {e}")
        return (task_name, False, str(e))


def main():
    parser = argparse.ArgumentParser(description='LoRA微调工作流管理')
    parser.add_argument('--model', type=str,
                       choices=['Llama', 'Llama-Instruct', 'Qwen', 'Qwen-Instruct', 'Mistral', 'Mistral-Instruct'],
                       help='模型名称')
    parser.add_argument('--config', type=str, choices=['best_acc', 'best_ppl'],
                       help='配置类型')
    parser.add_argument('--stage', type=str, choices=['finetune', 'evaluate', 'compare', 'all'],
                       default='all', help='执行阶段')
    parser.add_argument('--batch-all', action='store_true',
                       help='批量处理所有模型和配置')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='并行使用的GPU数量（默认1，顺序执行）')

    args = parser.parse_args()

    if args.batch_all:
        # 批量处理
        models = ['Llama', 'Llama-Instruct', 'Qwen', 'Qwen-Instruct', 'Mistral', 'Mistral-Instruct']
        configs = ['best_acc', 'best_ppl']

        print(f"\n{'='*80}")
        print(f"批量处理所有模型")
        print(f"{'='*80}")
        total_tasks = len(models) * len(configs)
        print(f"\n将处理 {len(models)} × {len(configs)} = {total_tasks} 个配置")

        # 构建任务列表
        tasks = [(model, config) for model in models for config in configs]

        if args.num_gpus > 1:
            # 并行模式
            print(f"\n并行模式: 使用 {args.num_gpus} 个GPU")

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
                    future = executor.submit(run_single_task, model, config, args.stage, gpu_id)
                    future_to_task[future] = (model, config)

                # 等待任务完成
                for future in as_completed(future_to_task):
                    task_name, success, msg = future.result()
                    results.append((task_name, success, msg))

            # 打印结果摘要
            print(f"\n{'='*80}")
            print(f"批量处理完成")
            print(f"{'='*80}")
            success_count = sum(1 for _, success, _ in results if success)
            print(f"\n成功: {success_count}/{len(results)}")

            if success_count < len(results):
                print("\n失败的任务:")
                for task_name, success, msg in results:
                    if not success:
                        print(f"  ✗ {task_name}: {msg}")
        else:
            # 顺序模式
            print(f"\n顺序模式: 逐个处理")
            for model, config in tasks:
                try:
                    workflow = FinetuningWorkflow(model, config)

                    if args.stage in ['finetune', 'all']:
                        workflow.finetune()

                    if args.stage in ['evaluate', 'all']:
                        workflow.evaluate()

                    if args.stage in ['compare', 'all']:
                        workflow.compare_results()

                except Exception as e:
                    print(f"\n✗ 处理 {model}/{config} 时出错: {e}")
                    continue

            print(f"\n✓ 批量处理完成")
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
