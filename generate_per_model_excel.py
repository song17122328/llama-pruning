#!/usr/bin/env python3
"""
为每个模型生成单独的Excel表格
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


def process_model(model_name, finetuned_eval_dir):
    """处理单个模型的所有配置"""
    model_dir = finetuned_eval_dir / model_name
    if not model_dir.is_dir():
        return None

    results = []

    for config_dir in sorted(model_dir.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.endswith('_finetuned'):
            continue

        config_name = config_dir.name.replace('_finetuned', '')

        print(f"  Processing {model_name}/{config_name}...")

        # 加载微调后的结果
        after_data = load_evaluation_results(config_dir)
        if not after_data:
            print(f"    Warning: No evaluation_results.json found")
            continue

        # 尝试从comparison_report.txt获取微调前的数据
        comparison_file = config_dir / 'comparison_report.txt'
        comparison_data = extract_comparison_data(comparison_file)

        # 如果没有comparison_report，从for_finetuning加载
        if not comparison_data:
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
        # 处理zeroshot为None的情况
        if zeroshot is None:
            zeroshot = {}
        tasks_before = comparison_data['tasks_before'] if comparison_data else {}

        for task in tasks:
            if task in zeroshot:
                acc_after = zeroshot[task]['accuracy']
                acc_before = tasks_before.get(task, None)

                row[f'{task}_微调前'] = round(acc_before, 4) if acc_before else 'N/A'
                row[f'{task}_微调后'] = round(acc_after, 4)

                if acc_before:
                    change = acc_after - acc_before
                    row[f'{task}_变化'] = round(change, 4)
                else:
                    row[f'{task}_变化'] = 'N/A'
            else:
                # 如果任务数据不存在，填充N/A
                row[f'{task}_微调前'] = 'N/A'
                row[f'{task}_微调后'] = 'N/A'
                row[f'{task}_变化'] = 'N/A'

        results.append(row)

    return results


def main():
    # 找到所有的微调评估目录
    finetuned_eval_dir = Path('results/finetuned_evaluation')

    if not finetuned_eval_dir.exists():
        print("Error: results/finetuned_evaluation directory not found")
        return

    # 获取所有模型
    models = sorted([d.name for d in finetuned_eval_dir.iterdir() if d.is_dir()])

    print(f"找到 {len(models)} 个模型: {models}\n")

    # 为每个模型生成Excel
    output_dir = Path('outputs/per_model_excel')
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []  # 收集所有模型的结果

    for model in models:
        print(f"处理模型: {model}")
        results = process_model(model, finetuned_eval_dir)

        if not results:
            print(f"  ⚠️ 没有找到数据")
            continue

        # 为每个结果添加模型名称
        for r in results:
            r_with_model = {'模型': model}
            r_with_model.update(r)
            all_results.append(r_with_model)

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 保存为Excel
        excel_file = output_dir / f'{model}_finetuning_summary.xlsx'

        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # 主要指标sheet
            main_cols = ['配置', '参数量(B)', '微调前PPL', '微调后PPL', 'PPL变化(%)',
                        '微调前平均ACC', '微调后平均ACC', 'ACC变化(%)']
            df[main_cols].to_excel(writer, sheet_name='主要指标', index=False)

            # 各任务详细数据sheet
            tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
            for task in tasks:
                task_cols = ['配置', f'{task}_微调前', f'{task}_微调后', f'{task}_变化']
                if all(col in df.columns for col in task_cols):
                    task_df = df[task_cols].copy()
                    task_df.columns = ['配置', '微调前', '微调后', '变化']
                    task_df.to_excel(writer, sheet_name=task.upper(), index=False)

        print(f"  ✅ 已保存: {excel_file} ({len(results)} 个配置)\n")

    # 生成汇总所有模型的大表格
    if all_results:
        print("生成包含所有模型的汇总表格...")
        all_df = pd.DataFrame(all_results)

        all_excel_file = output_dir / 'ALL_MODELS_finetuning_summary.xlsx'

        with pd.ExcelWriter(all_excel_file, engine='openpyxl') as writer:
            # 主要指标sheet
            main_cols = ['模型', '配置', '参数量(B)', '微调前PPL', '微调后PPL', 'PPL变化(%)',
                        '微调前平均ACC', '微调后平均ACC', 'ACC变化(%)']
            all_df[main_cols].to_excel(writer, sheet_name='主要指标', index=False)

            # 各任务详细数据sheet
            tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
            for task in tasks:
                task_cols = ['模型', '配置', f'{task}_微调前', f'{task}_微调后', f'{task}_变化']
                if all(col in all_df.columns for col in task_cols):
                    task_df = all_df[task_cols].copy()
                    task_df.columns = ['模型', '配置', '微调前', '微调后', '变化']
                    task_df.to_excel(writer, sheet_name=task.upper(), index=False)

        print(f"  ✅ 已保存汇总表格: {all_excel_file} ({len(all_results)} 个配置)\n")

    print(f"\n所有Excel文件已保存到: {output_dir}/")
    print(f"  - 6个单独模型表格")
    print(f"  - 1个汇总所有模型的表格")
    print(f"  - 总计7个Excel文件")
    print("="*80)


if __name__ == "__main__":
    main()
