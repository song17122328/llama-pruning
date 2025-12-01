#!/usr/bin/env python3
"""
汇总模型所有剪枝方法的结果并进行综合分析

用法:
    python param_search/consolidate_model_results.py --model llama
    python param_search/consolidate_model.py --model qwen
    python param_search/consolidate_model_results.py --model mistral
"""

import csv
import json
import argparse
from pathlib import Path
import sys


def load_search_results(csv_file):
    """加载搜索结果CSV"""
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值类型
            for key in row:
                if row[key] == '':
                    row[key] = None
                elif key not in ['output_dir'] and row[key] is not None:
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description='汇总模型所有剪枝方法的结果')
    parser.add_argument('--model', type=str, required=True,
                       choices=['Llama', 'Qwen', 'Mistral'],
                       help='模型名称')
    args = parser.parse_args()

    model = args.model  # 用于显示和目录名（统一大写开头）

    # 查找所有相关搜索目录
    search_patterns = [
        f'search_{model}_20',          # 普通 Taylor
        f'search_{model}_layerwise_20', # Layerwise
        f'search_{model}_blockwise_20'  # Blockwise
    ]

    all_results = []
    method_stats = {}

    print(f"\n{'='*80}")
    print(f"汇总 {model} 模型所有剪枝方法的结果")
    print(f"{'='*80}\n")

    for pattern in search_patterns:
        csv_file = Path('results') / pattern / 'search_results.csv'

        if not csv_file.exists():
            print(f"⚠️  未找到: {csv_file}")
            continue

        # 确定剪枝方法
        if 'layerwise' in pattern:
            method = 'layerwise'
        elif 'blockwise' in pattern:
            method = 'blockwise'
        else:
            method = 'taylor'

        # 加载结果
        results = load_search_results(csv_file)

        # 添加方法标签
        for result in results:
            result['pruning_method'] = method
            all_results.append(result)

        # 统计信息
        valid_results = [r for r in results if r.get('acc_mean') is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x['acc_mean'])
            method_stats[method] = {
                'total_experiments': len(results),
                'valid_experiments': len(valid_results),
                'best_acc': best['acc_mean'],
                'best_ppl': best.get('ppl'),
                'best_grad_norm_ratio': best.get('grad_norm_ratio'),
                'best_params': {
                    'taylor_seq_len': best.get('taylor_seq_len'),
                    'taylor_num_samples': best.get('taylor_num_samples')
                }
            }

            print(f"✓ {method.upper():12s}: {len(results)} 实验, 最佳 ACC = {best['acc_mean']:.4f}, PPL = {best.get('ppl', 'N/A')}")
        else:
            print(f"✗ {method.upper():12s}: {len(results)} 实验, 无有效结果")

    if not all_results:
        print("\n❌ 没有找到任何结果")
        return

    # 保存汇总结果
    output_dir = Path('results') / f'consolidated_{model}_20'
    output_dir.mkdir(exist_ok=True)

    # 保存 CSV
    csv_file = output_dir / 'all_methods_results.csv'
    keys = all_results[0].keys()
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\n✓ 已保存汇总CSV到: {csv_file}")
    print(f"  总实验数: {len(all_results)}")

    # 找出全局最佳配置
    valid_results = [r for r in all_results if r.get('acc_mean') is not None]
    if valid_results:
        global_best = max(valid_results, key=lambda x: x['acc_mean'])

        print(f"\n{'='*80}")
        print("全局最佳配置")
        print(f"{'='*80}")
        print(f"剪枝方法: {global_best['pruning_method'].upper()}")
        print(f"ACC mean: {global_best['acc_mean']:.4f}")
        print(f"PPL:      {global_best.get('ppl', 'N/A'):.2f}" if global_best.get('ppl') else "PPL:      N/A")
        print(f"梯度范数比: {global_best.get('grad_norm_ratio', 'N/A'):.2f}" if global_best.get('grad_norm_ratio') else "梯度范数比: N/A")
        print(f"参数:")
        print(f"  - taylor_seq_len: {global_best.get('taylor_seq_len')}")
        print(f"  - taylor_num_samples: {global_best.get('taylor_num_samples')}")

        # 保存全局最佳配置
        best_config = {
            "model": model,
            "pruning_method": global_best['pruning_method'],
            "params": {
                "taylor_seq_len": global_best.get('taylor_seq_len'),
                "taylor_num_samples": global_best.get('taylor_num_samples')
            },
            "metrics": {
                "acc_mean": global_best['acc_mean'],
                "ppl": global_best.get('ppl'),
                "grad_norm_ratio": global_best.get('grad_norm_ratio'),
                "grad_mean_ratio": global_best.get('grad_mean_ratio'),
                "extreme_pruning_layers": global_best.get('extreme_pruning_layers')
            },
            "output_dir": global_best['output_dir']
        }

        best_config_file = output_dir / 'global_best_config.json'
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)

        print(f"\n✓ 已保存全局最佳配置到: {best_config_file}")

    # 保存方法对比统计
    stats_file = output_dir / 'method_comparison.json'
    with open(stats_file, 'w') as f:
        json.dump(method_stats, f, indent=2)

    print(f"✓ 已保存方法对比统计到: {stats_file}")

    # 方法对比总结
    print(f"\n{'='*80}")
    print("剪枝方法对比总结")
    print(f"{'='*80}")
    print(f"{'方法':<15} {'实验数':<10} {'最佳ACC':<12} {'最佳PPL':<12} {'梯度范数比':<12}")
    print(f"{'-'*80}")

    for method in ['taylor', 'layerwise', 'blockwise']:
        if method in method_stats:
            stats = method_stats[method]
            print(f"{method.upper():<15} "
                  f"{stats['total_experiments']:<10} "
                  f"{stats['best_acc']:<12.4f} "
                  f"{stats['best_ppl']:<12.2f}" if stats['best_ppl'] else f"{'N/A':<12} "
                  f"{stats['best_grad_norm_ratio']:<12.2f}" if stats['best_grad_norm_ratio'] else f"{'N/A':<12}")

    print(f"\n✓ 汇总完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
