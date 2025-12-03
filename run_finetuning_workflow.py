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

    # 批量处理所有模型
    python run_finetuning_workflow.py --batch-all --stage all
"""

import argparse
import json
import subprocess
from pathlib import Path
import sys


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

    def finetune(self, lora_config=None):
        """运行LoRA微调"""
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

        # 构建微调命令
        # 注意：这里需要根据实际的微调脚本调整
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

        print(f"\n执行命令: {' '.join(cmd)}")
        print(f"\n⚠️  注意：请确保 finetune_lora.py 存在并且参数正确")
        print(f"如果需要，请修改此脚本中的命令构建逻辑")

        # 取消注释以下行来实际运行微调
        # try:
        #     subprocess.run(cmd, check=True)
        #     print(f"\n✓ 微调完成")
        #     return True
        # except subprocess.CalledProcessError as e:
        #     print(f"\n✗ 微调失败: {e}")
        #     return False

        print(f"\n✓ 微调命令已准备（未执行）")
        return True

    def evaluate(self):
        """评估微调后的模型"""
        print(f"\n{'='*80}")
        print(f"评估微调后模型: {self.model} - {self.config_type}")
        print(f"{'='*80}")

        print(f"\n微调模型目录: {self.finetuned_dir}")
        print(f"评估输出目录: {self.eval_dir}")

        # 创建评估输出目录
        self.eval_dir.mkdir(parents=True, exist_ok=True)

        # 构建评估命令
        cmd = [
            'python', 'run_evaluation.py',
            '--model_path', str(self.finetuned_dir),
            '--output_dir', str(self.eval_dir),
            '--eval_ppl',  # 评估PPL
            '--eval_zeroshot'  # 评估zero-shot ACC
        ]

        print(f"\n执行命令: {' '.join(cmd)}")
        print(f"\n⚠️  注意：请确保 run_evaluation.py 存在并且支持LoRA模型评估")

        # 取消注释以下行来实际运行评估
        # try:
        #     subprocess.run(cmd, check=True)
        #     print(f"\n✓ 评估完成")
        #     return True
        # except subprocess.CalledProcessError as e:
        #     print(f"\n✗ 评估失败: {e}")
        #     return False

        print(f"\n✓ 评估命令已准备（未执行）")
        return True

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

    args = parser.parse_args()

    if args.batch_all:
        # 批量处理
        models = ['Llama', 'Llama-Instruct', 'Qwen', 'Qwen-Instruct', 'Mistral', 'Mistral-Instruct']
        configs = ['best_acc', 'best_ppl']

        print(f"\n{'='*80}")
        print(f"批量处理所有模型")
        print(f"{'='*80}")
        print(f"\n将处理 {len(models)} × {len(configs)} = {len(models)*len(configs)} 个配置")

        for model in models:
            for config in configs:
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
