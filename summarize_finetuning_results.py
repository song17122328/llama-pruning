#!/usr/bin/env python3
"""
汇总所有微调评估结果到表格
"""

import json
from pathlib import Path
import pandas as pd
import re


def extract_comparison_data(comparison_file):
    """从comparison_report.txt提取微调前的数据"""
    if not comparison_file.exists():
        return None

    with open(comparison_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取微调前的PPL
    ppl_match = re.search(r'微调前:\s+([\d.]+)', content)
    ppl_before = float(ppl_match.group(1)) if ppl_match else None

    # 提取微调前的平均准确率
    avg_match = re.search(r'平均\s+:\s+([\d.]+)\s+→', content)
    avg_before = float(avg_match.group(1)) if avg_match else None

    # 提取各任务的微调前准确率
    tasks = {}
    task_pattern = re.compile(r'(\w+)\s+:\s+([\d.]+)\s+→\s+([\d.]+)')
    for match in task_pattern.finditer(content):
        task_name = match.group(1)
        before = float(match.group(2))
        tasks[task_name] = before

    return {
        'ppl_before': ppl_before,
        'avg_acc_before': avg_before,
        'tasks_before': tasks
    }


def load_evaluation_results(eval_dir):
    """加载evaluation_results.json"""
    eval_file = eval_dir / 'evaluation_results.json'
    if not eval_file.exists():
        return None

    with open(eval_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_before_results(model, config):
    """从for_finetuning目录加载微调前的结果"""
    # 对于base模型
    if config == 'base':
        eval_file = Path('results/base_evaluation') / model / 'evaluation_results.json'
    else:
        eval_file = Path('results/for_finetuning') / model / config / 'evaluation' / 'evaluation_results.json'

    if not eval_file.exists():
        return None

    with open(eval_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # 找到所有的微调评估目录
    finetuned_eval_dir = Path('results/finetuned_evaluation')

    if not finetuned_eval_dir.exists():
        print("Error: results/finetuned_evaluation directory not found")
        return

    # 收集所有结果
    results = []

    # 遍历所有模型和配置
    for model_dir in sorted(finetuned_eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir() or not config_dir.name.endswith('_finetuned'):
                continue

            config_name = config_dir.name.replace('_finetuned', '')

            print(f"Processing {model_name}/{config_name}...")

            # 加载微调后的结果
            after_data = load_evaluation_results(config_dir)
            if not after_data:
                print(f"  Warning: No evaluation_results.json found")
                continue

            # 尝试从comparison_report.txt获取微调前的数据
            comparison_file = config_dir / 'comparison_report.txt'
            comparison_data = extract_comparison_data(comparison_file)

            # 如果没有comparison_report，从for_finetuning加载
            if not comparison_data:
                print(f"  Loading before results from for_finetuning...")
                before_data = load_before_results(model_name, config_name)
                if before_data:
                    comparison_data = {
                        'ppl_before': before_data['metrics']['ppl'].get('wikitext2 (wikitext-2-raw-v1)', None),
                        'avg_acc_before': before_data['metrics'].get('avg_zeroshot_acc', None),
                        'tasks_before': {
                            task: data['accuracy']
                            for task, data in before_data['metrics'].get('zeroshot', {}).items()
                        }
                    }

            # 提取数据
            metrics = after_data['metrics']

            # 基本信息
            row = {
                '模型': model_name,
                '配置': config_name,
                '参数量(B)': round(metrics['model_info']['total_params_B'], 2),
            }

            # PPL数据
            ppl_after = metrics['ppl'].get('wikitext2 (wikitext-2-raw-v1)', None)
            ppl_before = comparison_data['ppl_before'] if comparison_data else None

            row['微调前PPL'] = round(ppl_before, 2) if ppl_before else 'N/A'
            row['微调后PPL'] = round(ppl_after, 2) if ppl_after else 'N/A'

            if ppl_before and ppl_after:
                ppl_change = ((ppl_after - ppl_before) / ppl_before) * 100
                row['PPL变化(%)'] = round(ppl_change, 2)
            else:
                row['PPL变化(%)'] = 'N/A'

            # Zero-shot准确率数据
            avg_acc_after = metrics.get('avg_zeroshot_acc', None)
            avg_acc_before = comparison_data['avg_acc_before'] if comparison_data else None

            row['微调前平均ACC'] = round(avg_acc_before, 4) if avg_acc_before else 'N/A'
            row['微调后平均ACC'] = round(avg_acc_after, 4) if avg_acc_after else 'N/A'

            if avg_acc_before and avg_acc_after:
                acc_change = ((avg_acc_after - avg_acc_before) / avg_acc_before) * 100
                row['ACC变化(%)'] = round(acc_change, 2)
            else:
                row['ACC变化(%)'] = 'N/A'

            # 各个任务的详细准确率
            tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
            zeroshot = metrics.get('zeroshot', {})
            tasks_before = comparison_data['tasks_before'] if comparison_data else {}

            for task in tasks:
                if task in zeroshot:
                    acc_after = zeroshot[task]['accuracy']
                    acc_before = tasks_before.get(task, None)

                    row[f'{task}_before'] = round(acc_before, 4) if acc_before else 'N/A'
                    row[f'{task}_after'] = round(acc_after, 4)

                    if acc_before:
                        change = acc_after - acc_before
                        row[f'{task}_change'] = round(change, 4)
                    else:
                        row[f'{task}_change'] = 'N/A'

            results.append(row)

    # 转换为DataFrame
    df = pd.DataFrame(results)

    # 保存为CSV
    csv_file = 'finetuning_results_summary.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 已保存CSV文件: {csv_file}")

    # 保存为Markdown
    md_file = 'finetuning_results_summary.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 微调结果汇总\n\n")
        f.write(f"总计: {len(results)} 个模型配置\n\n")

        # 主要指标表格
        f.write("## 主要指标概览\n\n")
        main_cols = ['模型', '配置', '参数量(B)', '微调前PPL', '微调后PPL', 'PPL变化(%)',
                     '微调前平均ACC', '微调后平均ACC', 'ACC变化(%)']
        main_df = df[main_cols]
        f.write(main_df.to_markdown(index=False))
        f.write("\n\n")

        # 详细任务表格
        f.write("## 各任务详细准确率\n\n")
        for task in tasks:
            f.write(f"### {task.upper()}\n\n")
            task_cols = ['模型', '配置', f'{task}_before', f'{task}_after', f'{task}_change']
            if all(col in df.columns for col in task_cols):
                task_df = df[task_cols]
                task_df.columns = ['模型', '配置', '微调前', '微调后', '变化']
                f.write(task_df.to_markdown(index=False))
                f.write("\n\n")

    print(f"✅ 已保存Markdown文件: {md_file}")

    # 打印统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"总计: {len(results)} 个模型配置")
    print(f"模型: {df['模型'].unique().tolist()}")
    print(f"配置: {df['配置'].unique().tolist()}")

    # PPL改善统计
    ppl_changes = df['PPL变化(%)'].replace('N/A', float('nan')).astype(float)
    print(f"\nPPL变化: 平均 {ppl_changes.mean():.2f}%, 最大改善 {ppl_changes.min():.2f}%")

    # ACC改善统计
    acc_changes = df['ACC变化(%)'].replace('N/A', float('nan')).astype(float)
    print(f"ACC变化: 平均 {acc_changes.mean():.2f}%, 最大改善 {acc_changes.max():.2f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
