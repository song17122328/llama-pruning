#!/usr/bin/env python3
"""
展示每个模型ACC前5的实验结果

用法:
    python param_search/show_top5_results.py
"""

import csv
from pathlib import Path


def load_and_rank_results(model):
    """加载模型结果并按ACC排序"""
    csv_file = Path('results') / f'consolidated_{model}_20' / 'all_methods_results.csv'

    if not csv_file.exists():
        return None

    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 只保留成功的实验
            if row.get('success') == 'True' and row.get('ppl'):
                results.append(row)

    # 按 acc_mean 降序排序
    results.sort(key=lambda x: float(x['ppl']), reverse=False)

    return results


def display_top5(model, results, top_n=5):
    """显示前N个结果"""
    if not results:
        print(f"⚠️  {model}: 没有找到结果")
        return

    model_display = model.replace('-', ' ')
    print(f"\n{'='*100}")
    print(f"{model_display} - PPL Top {top_n}")
    print(f"{'='*100}")

    # 表头
    print(f"\n{'排名':<6} {'方法':<12} {'seq_len':<10} {'samples':<10} {'ACC':<12} {'PPL':<10} {'梯度范数比':<12}")
    print("-"*100)

    for i, row in enumerate(results[:top_n], 1):
        method = row.get('pruning_method', 'N/A').upper()
        seq_len = row.get('taylor_seq_len', '')
        samples = row.get('taylor_num_samples', '')
        acc = float(row['acc_mean'])
        ppl = float(row['ppl']) if row.get('ppl') and row['ppl'] else None
        grad_norm = float(row['grad_norm_ratio']) if row.get('grad_norm_ratio') and row['grad_norm_ratio'] not in ['', 'Infinity'] else None

        # 格式化输出
        seq_len_str = f"{int(float(seq_len))}" if seq_len and seq_len != 'N/A' else 'N/A'
        samples_str = f"{int(float(samples))}" if samples and samples != 'N/A' else 'N/A'
        ppl_str = f"{ppl:.2f}" if ppl else 'N/A'
        grad_str = f"{grad_norm:.2f}" if grad_norm else 'N/A'

        print(f"#{i:<5} {method:<12} {seq_len_str:<10} {samples_str:<10} {acc:<12.4f} {ppl_str:<10} {grad_str:<12}")

    print(f"\n详细任务PPL （前3名）:")
    print("-"*100)

    tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
    for i, row in enumerate(results[:3], 1):
        method = row.get('pruning_method', 'N/A').upper()
        seq_len = row.get('taylor_seq_len', '')
        samples = row.get('taylor_num_samples', '')

        seq_len_str = f"{int(float(seq_len))}" if seq_len and seq_len != 'N/A' else 'N/A'
        samples_str = f"{int(float(samples))}" if samples and samples != 'N/A' else 'N/A'

        print(f"\n#{i} - {method} (seq_len={seq_len_str}, samples={samples_str})")

        for task in tasks:
            col_name = f'acc_{task}'
            if col_name in row and row[col_name]:
                acc_val = float(row[col_name])
                print(f"  {task:15s}: {acc_val:.4f}")


def main():
    models = [
        'Llama',
        'Llama-Instruct',
        'Qwen',
        'Qwen-Instruct',
        'Mistral',
        'Mistral-Instruct'
    ]

    print("\n" + "="*100)
    print("所有模型 PPL Top 5 结果")
    print("="*100)

    for model in models:
        results = load_and_rank_results(model)
        display_top5(model, results, top_n=5)

    # 生成跨模型对比（每个模型的最佳结果）
    print(f"\n\n{'='*100}")
    print("跨模型最佳结果对比")
    print(f"{'='*100}")

    best_results = []
    for model in models:
        results = load_and_rank_results(model)
        if results:
            best = results[0]
            best_results.append({
                'model': model,
                'method': best.get('pruning_method', 'N/A').upper(),
                'seq_len': best.get('taylor_seq_len', 'N/A'),
                'samples': best.get('taylor_num_samples', 'N/A'),
                'acc': float(best['acc_mean']),
                'ppl': float(best['ppl']) if best.get('ppl') else None
            })

    # 按PPL升序排序
    best_results.sort(key=lambda x: x['ppl'], reverse=False)

    print(f"\n{'排名':<6} {'模型':<20} {'方法':<12} {'seq_len':<10} {'samples':<10} {'ACC':<12} {'PPL':<10}")
    print("-"*100)

    for i, result in enumerate(best_results, 1):
        model_display = result['model'].replace('-', ' ')
        seq_len_str = f"{int(float(result['seq_len']))}" if result['seq_len'] and result['seq_len'] != 'N/A' else 'N/A'
        samples_str = f"{int(float(result['samples']))}" if result['samples'] and result['samples'] != 'N/A' else 'N/A'
        ppl_str = f"{result['ppl']:.2f}" if result['ppl'] else 'N/A'

        print(f"#{i:<5} {model_display:<20} {result['method']:<12} {seq_len_str:<10} {samples_str:<10} "
              f"{result['acc']:<12.4f} {ppl_str:<10}")

    print(f"\n✓ 分析完成！\n")


if __name__ == '__main__':
    main()
