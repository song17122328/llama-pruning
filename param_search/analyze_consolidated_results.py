#!/usr/bin/env python3
"""
分析汇总结果，找出什么条件下能获得最佳 ACC

用法:
    python param_search/analyze_consolidated_results.py --model llama
    python param_search/analyze_consolidated_results.py --model qwen
    python param_search/analyze_consolidated_results.py --model mistral
    python param_search/analyze_consolidated_results.py --all  # 分析所有模型
"""

import csv
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def load_csv_data(csv_file):
    """加载 CSV 数据"""
    data = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值类型
            for key in row:
                if row[key] == '' or row[key] == 'None':
                    row[key] = None
                elif key not in ['output_dir', 'pruning_method', 'success'] and row[key] is not None:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            data.append(row)
    return data


def extract_params_from_path(path):
    """从输出路径中提取参数"""
    parts = Path(path).name.split('_')
    params = {}

    i = 0
    while i < len(parts):
        if parts[i] == 'taylor' and i + 2 < len(parts):
            if parts[i + 1] == 'seq':
                # taylor_seq_len16
                param_name = 'taylor_seq_len'
                value_str = parts[i + 2]
                value = ''.join([c for c in value_str if c.isdigit()])
                if value:
                    params[param_name] = int(value)
                i += 3
            elif parts[i + 1] == 'num':
                # taylor_num_samples128
                param_name = 'taylor_num_samples'
                value_str = parts[i + 2]
                value = ''.join([c for c in value_str if c.isdigit()])
                if value:
                    params[param_name] = int(value)
                i += 3
            else:
                i += 1
        else:
            i += 1

    return params


def calculate_correlation(x_vals, y_vals):
    """计算相关系数"""
    n = len(x_vals)
    if n < 2:
        return 0.0

    mean_x = sum(x_vals) / n
    mean_y = sum(y_vals) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    return numerator / (denominator_x * denominator_y)


def group_by(data, key, value_key):
    """按键分组并计算统计"""
    groups = defaultdict(list)
    for item in data:
        if item.get(key) is not None and item.get(value_key) is not None:
            groups[item[key]].append(item[value_key])

    stats = {}
    for group_key, values in groups.items():
        if values:
            stats[group_key] = {
                'mean': sum(values) / len(values),
                'std': math.sqrt(sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }

    return stats


def analyze_model(model_name):
    """分析单个模型的结果"""
    csv_file = Path('results') / f'consolidated_{model_name}_20' / 'all_methods_results.csv'

    if not csv_file.exists():
        print(f"❌ 未找到文件: {csv_file}")
        return None

    # 读取数据
    data = load_csv_data(csv_file)

    # 提取参数（如果没有的话）
    for row in data:
        if 'taylor_seq_len' not in row or row['taylor_seq_len'] is None:
            params = extract_params_from_path(row['output_dir'])
            row['taylor_seq_len'] = params.get('taylor_seq_len')
            row['taylor_num_samples'] = params.get('taylor_num_samples')

    # 过滤有效结果
    valid_data = [row for row in data if row.get('acc_mean') is not None]
    invalid_data = [row for row in data if row.get('acc_mean') is None]

    print(f"\n{'='*80}")
    print(f"{model_name.upper()} 模型分析")
    print(f"{'='*80}")
    print(f"总实验数: {len(data)}, 有效结果: {len(valid_data)}, 无效实验: {len(invalid_data)}")

    # 显示无效实验详情
    if len(invalid_data) > 0:
        print(f"\n{'─'*80}")
        print("无效实验详情")
        print(f"{'─'*80}")

        for i, row in enumerate(invalid_data, 1):
            print(f"\n{i}. {row['output_dir']}")
            print(f"   剪枝方法: {row.get('pruning_method', 'N/A')}")

            # 显示参数
            if row.get('taylor_seq_len') is not None:
                print(f"   taylor_seq_len: {int(row['taylor_seq_len'])}")
            if row.get('taylor_num_samples') is not None:
                print(f"   taylor_num_samples: {int(row['taylor_num_samples'])}")

            # 显示失败原因
            reasons = []
            if row.get('ppl') is None:
                reasons.append("PPL缺失")
            if row.get('acc_mean') is None:
                reasons.append("ACC缺失")
            if row.get('grad_norm_ratio') is None:
                reasons.append("梯度统计缺失")

            if reasons:
                print(f"   失败原因: {', '.join(reasons)}")

            # 显示是否标记为成功
            if row.get('success') is not None:
                print(f"   success标记: {row['success']}")

    if len(valid_data) == 0:
        print("\n❌ 没有有效结果")
        return None

    # 1. 按剪枝方法分组分析
    print(f"\n{'─'*80}")
    print("1. 剪枝方法对比")
    print(f"{'─'*80}")

    method_stats = group_by(valid_data, 'pruning_method', 'acc_mean')

    print(f"{'方法':<15} {'平均ACC':<12} {'标准差':<10} {'最小值':<10} {'最大值':<10} {'实验数':<8}")
    print(f"{'-'*80}")
    for method, stats in sorted(method_stats.items()):
        print(f"{method.upper():<15} {stats['mean']:<12.4f} {stats['std']:<10.4f} "
              f"{stats['min']:<10.4f} {stats['max']:<10.4f} {stats['count']:<8}")

    # 找出最佳方法
    best_row = max(valid_data, key=lambda x: x['acc_mean'])
    best_method = best_row['pruning_method']
    best_acc = best_row['acc_mean']
    print(f"\n✓ 最佳剪枝方法: {best_method.upper()} (ACC: {best_acc:.4f})")

    # 2. 参数相关性分析（仅针对有 taylor 参数的）
    taylor_data = [row for row in valid_data if row.get('taylor_seq_len') is not None]

    if len(taylor_data) > 0:
        print(f"\n{'─'*80}")
        print("2. Taylor 参数影响分析")
        print(f"{'─'*80}")

        # 按 seq_len 分组
        seq_len_stats = group_by(taylor_data, 'taylor_seq_len', 'acc_mean')
        if len(seq_len_stats) > 1:
            print("\n按 taylor_seq_len 分组:")
            print(f"{'Seq Len':<12} {'平均ACC':<12} {'标准差':<10} {'实验数':<8}")
            print(f"{'-'*50}")
            for seq_len, stats in sorted(seq_len_stats.items()):
                print(f"{int(seq_len):<12} {stats['mean']:<12.4f} {stats['std']:<10.4f} {stats['count']:<8}")

            best_seq_len = max(seq_len_stats.items(), key=lambda x: x[1]['mean'])
            print(f"✓ 最佳 seq_len: {int(best_seq_len[0])} (平均 ACC: {best_seq_len[1]['mean']:.4f})")

        # 按 num_samples 分组
        samples_stats = group_by(taylor_data, 'taylor_num_samples', 'acc_mean')
        if len(samples_stats) > 1:
            print("\n按 taylor_num_samples 分组:")
            print(f"{'Num Samples':<12} {'平均ACC':<12} {'标准差':<10} {'实验数':<8}")
            print(f"{'-'*50}")
            for samples, stats in sorted(samples_stats.items()):
                print(f"{int(samples):<12} {stats['mean']:<12.4f} {stats['std']:<10.4f} {stats['count']:<8}")

            best_samples = max(samples_stats.items(), key=lambda x: x[1]['mean'])
            print(f"✓ 最佳 num_samples: {int(best_samples[0])} (平均 ACC: {best_samples[1]['mean']:.4f})")

    # 3. 梯度指标与 ACC 的相关性
    print(f"\n{'─'*80}")
    print("3. 梯度指标与 ACC 的相关性")
    print(f"{'─'*80}")

    grad_metrics = ['grad_mean_ratio', 'grad_norm_ratio', 'grad_std_ratio',
                    'grad_max_ratio', 'grad_mean_range', 'grad_norm_range']

    correlations = {}
    for metric in grad_metrics:
        pairs = [(row['acc_mean'], row[metric])
                 for row in valid_data
                 if row.get(metric) is not None and row.get('acc_mean') is not None]

        if len(pairs) > 1:
            acc_vals = [p[0] for p in pairs]
            metric_vals = [p[1] for p in pairs]
            corr = calculate_correlation(acc_vals, metric_vals)
            correlations[metric] = corr

    if correlations:
        print(f"{'指标':<25} {'相关系数':<12}")
        print(f"{'-'*40}")
        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for metric, corr in sorted_corrs:
            print(f"{metric:<25} {corr:>11.4f}")

        # 解释相关性
        print("\n相关性解释:")
        for metric, corr in sorted_corrs:
            if abs(corr) > 0.5:
                direction = "正" if corr > 0 else "负"
                strength = "强" if abs(corr) > 0.7 else "中等"
                print(f"  • {metric}: {strength}{direction}相关 ({corr:.3f})")

    # 4. 极端剪枝层数影响
    extreme_data = [row for row in valid_data if row.get('extreme_pruning_layers') is not None]
    if len(extreme_data) > 1:
        print(f"\n{'─'*80}")
        print("4. 极端剪枝层数影响")
        print(f"{'─'*80}")

        acc_vals = [row['acc_mean'] for row in extreme_data]
        extreme_vals = [row['extreme_pruning_layers'] for row in extreme_data]
        corr = calculate_correlation(acc_vals, extreme_vals)
        print(f"与 ACC 的相关系数: {corr:.4f}")

        # 按层数分组
        extreme_stats = group_by(extreme_data, 'extreme_pruning_layers', 'acc_mean')
        print("\n按极端剪枝层数分组:")
        print(f"{'层数':<8} {'平均ACC':<12} {'实验数':<8}")
        print(f"{'-'*30}")
        for layers, stats in sorted(extreme_stats.items()):
            print(f"{int(layers):<8} {stats['mean']:<12.4f} {stats['count']:<8}")

    # 5. PPL 与 ACC 的关系
    ppl_data = [row for row in valid_data if row.get('ppl') is not None]
    if len(ppl_data) > 1:
        print(f"\n{'─'*80}")
        print("5. PPL 与 ACC 的关系")
        print(f"{'─'*80}")

        acc_vals = [row['acc_mean'] for row in ppl_data]
        ppl_vals = [row['ppl'] for row in ppl_data]
        corr = calculate_correlation(acc_vals, ppl_vals)
        print(f"PPL 与 ACC 的相关系数: {corr:.4f}")

    # 6. 最佳配置详情
    print(f"\n{'─'*80}")
    print("6. 最佳配置详情")
    print(f"{'─'*80}")

    print(f"剪枝方法: {best_row['pruning_method'].upper()}")
    print(f"ACC mean: {best_row['acc_mean']:.4f}")
    if best_row.get('ppl') is not None:
        print(f"PPL: {best_row['ppl']:.2f}")
    if best_row.get('taylor_seq_len') is not None:
        print(f"taylor_seq_len: {int(best_row['taylor_seq_len'])}")
    if best_row.get('taylor_num_samples') is not None:
        print(f"taylor_num_samples: {int(best_row['taylor_num_samples'])}")
    if best_row.get('grad_norm_ratio') is not None:
        print(f"grad_norm_ratio: {best_row['grad_norm_ratio']:.2f}")
    if best_row.get('extreme_pruning_layers') is not None:
        print(f"extreme_pruning_layers: {int(best_row['extreme_pruning_layers'])}")

    # 各任务详细 ACC
    print("\n各任务详细 ACC:")
    task_cols = ['acc_boolq', 'acc_piqa', 'acc_hellaswag', 'acc_winogrande',
                 'acc_arc_easy', 'acc_arc_challenge', 'acc_openbookqa']
    for task in task_cols:
        if best_row.get(task) is not None:
            task_name = task.replace('acc_', '').upper()
            print(f"  • {task_name:15s}: {best_row[task]:.4f}")

    # 返回汇总统计
    return {
        'model': model_name,
        'total_experiments': len(data),
        'valid_experiments': len(valid_data),
        'best_method': best_method,
        'best_acc': float(best_acc),
        'best_ppl': float(best_row['ppl']) if best_row.get('ppl') is not None else None,
        'correlations': {k: float(v) for k, v in correlations.items()} if correlations else {}
    }


def analyze_all_models():
    """分析所有模型并进行跨模型对比"""
    models = ['Llama', 'Qwen', 'Mistral']
    all_stats = []

    for model in models:
        stats = analyze_model(model)
        if stats:
            all_stats.append(stats)

    if len(all_stats) > 1:
        print(f"\n\n{'='*80}")
        print("跨模型对比")
        print(f"{'='*80}")

        # 排序
        all_stats_sorted = sorted(all_stats, key=lambda x: x['best_acc'], reverse=True)

        print(f"\n{'模型':<12} {'最佳方法':<12} {'最佳ACC':<12} {'最佳PPL':<12} {'有效实验数':<12}")
        print(f"{'-'*80}")
        for stats in all_stats_sorted:
            ppl_str = f"{stats['best_ppl']:.2f}" if stats['best_ppl'] is not None else "N/A"
            print(f"{stats['model']:<12} {stats['best_method'].upper():<12} "
                  f"{stats['best_acc']:<12.4f} {ppl_str:<12} {stats['valid_experiments']:<12}")

        # 总体最佳
        best_overall = all_stats_sorted[0]
        print(f"\n✓ 总体最佳: {best_overall['model'].upper()} 模型")
        print(f"  - 剪枝方法: {best_overall['best_method'].upper()}")
        print(f"  - ACC: {best_overall['best_acc']:.4f}")
        if best_overall['best_ppl'] is not None:
            print(f"  - PPL: {best_overall['best_ppl']:.2f}")

        # 保存跨模型对比
        output_file = Path('results') / 'cross_model_comparison.json'
        with open(output_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n✓ 已保存跨模型对比到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析汇总结果')
    parser.add_argument('--model', type=str, choices=['Llama', 'Qwen', 'Mistral', 'llama', 'qwen', 'mistral'],
                       help='分析指定模型')
    parser.add_argument('--all', action='store_true',
                       help='分析所有模型')
    args = parser.parse_args()

    if args.all:
        analyze_all_models()
    elif args.model:
        # 首字母大写
        model = args.model.capitalize()
        analyze_model(model)
    else:
        print("请指定 --model <模型名> 或 --all")
        parser.print_help()


if __name__ == '__main__':
    main()
