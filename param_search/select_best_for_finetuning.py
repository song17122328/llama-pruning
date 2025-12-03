#!/usr/bin/env python3
"""
为每个模型选择ACC最高和PPL最低的配置，准备用于微调

用法:
    python param_search/select_best_for_finetuning.py
"""

import csv
import json
import shutil
from pathlib import Path


def load_and_select_best(model):
    """加载模型结果并选择ACC最高和PPL最低的配置"""
    csv_file = Path('results') / f'consolidated_{model}_20' / 'all_methods_results.csv'

    if not csv_file.exists():
        print(f"⚠️  {model}: CSV文件不存在")
        return None, None

    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只保留成功的实验
            if row.get('success') == 'True' and row.get('acc_mean') and row.get('ppl'):
                try:
                    row['acc_mean'] = float(row['acc_mean'])
                    row['ppl'] = float(row['ppl'])
                    results.append(row)
                except (ValueError, TypeError):
                    continue

    if not results:
        print(f"⚠️  {model}: 没有有效结果")
        return None, None

    # 选择ACC最高的
    best_acc = max(results, key=lambda x: x['acc_mean'])

    # 选择PPL最低的（注意：PPL越低越好）
    best_ppl = min(results, key=lambda x: x['ppl'])

    return best_acc, best_ppl


def copy_model_for_finetuning(source_dir, dest_dir, selection_info):
    """复制模型文件到微调目录"""
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        print(f"    ⚠️  源目录不存在: {source_dir}")
        return False

    # 创建目标目录
    dest_path.mkdir(parents=True, exist_ok=True)

    # 需要复制的文件和文件夹
    items_to_copy = [
        'pruned_model.bin',      # 剪枝后的模型
        # 'config.json',           # 模型配置
        'evaluation',            # 评估结果
        'analysis',              # 分析数据
        'visualization',         # 可视化
        'logs'                   # 日志
    ]

    copied_count = 0
    for item in items_to_copy:
        src_item = source_path / item
        dst_item = dest_path / item

        if src_item.exists():
            # 如果目标已存在，先删除
            if dst_item.exists():
                if dst_item.is_dir():
                    shutil.rmtree(dst_item)
                else:
                    dst_item.unlink()

            # 复制
            if src_item.is_dir():
                shutil.copytree(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)
            copied_count += 1
        else:
            print(f"    ⚠️  项目不存在: {src_item}")

    # 保存选择信息
    info_file = dest_path / 'selection_info.json'
    with open(info_file, 'w') as f:
        json.dump(selection_info, f, indent=2)

    return copied_count > 0


def process_model(model):
    """处理单个模型：选择最佳配置并复制"""
    print(f"\n{'='*80}")
    print(f"处理模型: {model}")
    print(f"{'='*80}")

    best_acc, best_ppl = load_and_select_best(model)

    if not best_acc or not best_ppl:
        print(f"  ✗ 跳过（无有效结果）")
        return

    # 准备目录
    base_dir = Path('results') / 'for_finetuning' / model

    # 处理ACC最高的配置
    print(f"\n📊 ACC最高的配置:")
    print(f"  ACC: {best_acc['acc_mean']:.4f}")
    print(f"  PPL: {best_acc['ppl']:.2f}")
    print(f"  方法: {best_acc.get('pruning_method', 'N/A').upper()}")
    print(f"  源目录: {best_acc['output_dir']}")

    acc_info = {
        'selection_criterion': 'best_acc',
        'acc_mean': best_acc['acc_mean'],
        'ppl': best_acc['ppl'],
        'pruning_method': best_acc.get('pruning_method', 'N/A'),
        'taylor_seq_len': best_acc.get('taylor_seq_len', 'N/A'),
        'taylor_num_samples': best_acc.get('taylor_num_samples', 'N/A'),
        'source_dir': best_acc['output_dir'],
        'model': model,
        'task_accuracies': {}
    }

    # 提取7个任务的ACC
    tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
    for task in tasks:
        col_name = f'acc_{task}'
        if col_name in best_acc and best_acc[col_name]:
            try:
                acc_info['task_accuracies'][task] = float(best_acc[col_name])
            except:
                pass

    acc_dest = base_dir / 'best_acc'
    if copy_model_for_finetuning(best_acc['output_dir'], acc_dest, acc_info):
        print(f"  ✓ 已复制到: {acc_dest}")
    else:
        print(f"  ✗ 复制失败")

    # 处理PPL最低的配置
    print(f"\n📊 PPL最低的配置:")
    print(f"  PPL: {best_ppl['ppl']:.2f}")
    print(f"  ACC: {best_ppl['acc_mean']:.4f}")
    print(f"  方法: {best_ppl.get('pruning_method', 'N/A').upper()}")
    print(f"  源目录: {best_ppl['output_dir']}")

    ppl_info = {
        'selection_criterion': 'best_ppl',
        'ppl': best_ppl['ppl'],
        'acc_mean': best_ppl['acc_mean'],
        'pruning_method': best_ppl.get('pruning_method', 'N/A'),
        'taylor_seq_len': best_ppl.get('taylor_seq_len', 'N/A'),
        'taylor_num_samples': best_ppl.get('taylor_num_samples', 'N/A'),
        'source_dir': best_ppl['output_dir'],
        'model': model,
        'task_accuracies': {}
    }

    for task in tasks:
        col_name = f'acc_{task}'
        if col_name in best_ppl and best_ppl[col_name]:
            try:
                ppl_info['task_accuracies'][task] = float(best_ppl[col_name])
            except:
                pass

    ppl_dest = base_dir / 'best_ppl'
    if copy_model_for_finetuning(best_ppl['output_dir'], ppl_dest, ppl_info):
        print(f"  ✓ 已复制到: {ppl_dest}")
    else:
        print(f"  ✗ 复制失败")

    # 检查是否是同一个配置
    if best_acc['output_dir'] == best_ppl['output_dir']:
        print(f"\n💡 注意：ACC最高和PPL最低是同一个配置！")


def main():
    models = [
        'Llama',
        'Llama-Instruct',
        'Qwen',
        'Qwen-Instruct',
        'Mistral',
        'Mistral-Instruct'
    ]

    print("\n" + "="*80)
    print("为微调选择最佳配置")
    print("="*80)
    print("\n选择标准:")
    print("  1. ACC最高: 用于评估剪枝后性能恢复")
    print("  2. PPL最低: 用于评估困惑度恢复")
    print(f"\n将为每个模型准备2个配置，共 {len(models)} × 2 = {len(models)*2} 个模型")

    for model in models:
        process_model(model)

    print("\n" + "="*80)
    print("选择完成")
    print("="*80)
    print(f"\n结果保存在: results/for_finetuning/")
    print(f"\n每个模型包含:")
    print(f"  - best_acc/: ACC最高的配置")
    print(f"  - best_ppl/: PPL最低的配置")
    print(f"\n每个配置包含:")
    print(f"  - pruned_model.bin: 剪枝后的模型权重")
    print(f"  - config.json: 模型配置")
    print(f"  - selection_info.json: 选择信息和基准指标")
    print(f"  - evaluation/: 评估结果")
    print(f"  - analysis/: 分析数据")

    # 生成摘要报告
    summary_file = Path('results') / 'for_finetuning' / 'SUMMARY.md'
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        f.write("# 微调前的模型选择摘要\n\n")
        f.write("本目录包含为LoRA微调准备的剪枝模型。\n\n")
        f.write("## 选择标准\n\n")
        f.write("- **best_acc**: ACC最高的配置（评估zero-shot任务性能恢复）\n")
        f.write("- **best_ppl**: PPL最低的配置（评估语言建模能力恢复）\n\n")
        f.write("## 模型列表\n\n")
        f.write("| 模型 | 类型 | best_acc | best_ppl | 是否相同 |\n")
        f.write("|------|------|----------|----------|----------|\n")

        for model in models:
            acc_file = Path('results') / 'for_finetuning' / model / 'best_acc' / 'selection_info.json'
            ppl_file = Path('results') / 'for_finetuning' / model / 'best_ppl' / 'selection_info.json'

            if acc_file.exists() and ppl_file.exists():
                with open(acc_file, 'r') as af:
                    acc_info = json.load(af)
                with open(ppl_file, 'r') as pf:
                    ppl_info = json.load(pf)

                is_same = "✓" if acc_info['source_dir'] == ppl_info['source_dir'] else ""
                model_type = "Instruct" if "Instruct" in model else "Base"

                f.write(f"| {model} | {model_type} | ACC:{acc_info['acc_mean']:.4f} | PPL:{ppl_info['ppl']:.2f} | {is_same} |\n")

        f.write(f"\n## 下一步\n\n")
        f.write(f"1. 对每个配置运行LoRA微调\n")
        f.write(f"2. 评估微调后的模型\n")
        f.write(f"3. 对比微调前后的性能\n\n")
        f.write(f"详细信息参见各模型目录下的 `selection_info.json` 文件。\n")

    print(f"\n✓ 摘要报告已保存到: {summary_file}")
    print(f"\n✓ 完成！\n")


if __name__ == '__main__':
    main()
