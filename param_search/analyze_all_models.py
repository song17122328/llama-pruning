#!/usr/bin/env python3
"""
跨模型综合分析：对比所有6个模型的最佳配置

用法:
    python param_search/analyze_all_models.py
"""

import json
import csv
from pathlib import Path


def load_best_config(model):
    """加载模型的最佳配置"""
    config_file = Path('results') / f'consolidated_{model}_20' / 'global_best_config.json'
    if not config_file.exists():
        return None

    with open(config_file, 'r') as f:
        return json.load(f)


def load_all_results(model):
    """加载模型的所有结果"""
    csv_file = Path('results') / f'consolidated_{model}_20' / 'all_methods_results.csv'
    if not csv_file.exists():
        return None

    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def main():
    models = {
        'base': ['Llama', 'Qwen', 'Mistral'],
        'instruct': ['Llama-Instruct', 'Qwen-Instruct', 'Mistral-Instruct']
    }

    all_models = models['base'] + models['instruct']

    print("\n" + "="*100)
    print("跨模型综合分析报告")
    print("="*100 + "\n")

    # ========== Part 1: 加载所有模型的最佳配置 ==========
    best_configs = {}
    for model in all_models:
        config = load_best_config(model)
        if config:
            best_configs[model] = config
        else:
            print(f"⚠️  未找到 {model} 的最佳配置")

    if not best_configs:
        print("❌ 没有找到任何模型的配置")
        return

    # ========== Part 2: 最佳配置总览 ==========
    print("\n" + "="*100)
    print("所有模型最佳配置总览")
    print("="*100)
    print(f"{'模型':<20} {'类型':<10} {'方法':<12} {'ACC':<10} {'PPL':<10} "
          f"{'seq_len':<10} {'samples':<10}")
    print("-"*100)

    for model_type in ['base', 'instruct']:
        for model in models[model_type]:
            if model not in best_configs:
                continue

            config = best_configs[model]
            model_base = model.replace('-Instruct', '')
            type_label = 'Instruct' if '-Instruct' in model else 'Base'

            acc = config['metrics']['acc_mean']
            ppl = config['metrics'].get('ppl', 'N/A')
            method = config['pruning_method'].upper()
            seq_len = config['params'].get('taylor_seq_len', 'N/A')
            samples = config['params'].get('taylor_num_samples', 'N/A')

            ppl_str = f"{ppl:.2f}" if ppl != 'N/A' else 'N/A'

            print(f"{model_base:<20} {type_label:<10} {method:<12} {acc:<10.4f} {ppl_str:<10} "
                  f"{seq_len:<10} {samples:<10}")

        if model_type == 'base':
            print("-"*100)

    # ========== Part 3: Base vs Instruct 对比 ==========
    print("\n" + "="*100)
    print("Base vs Instruct 模型对比")
    print("="*100)

    for model_base in ['Llama', 'Qwen', 'Mistral']:
        base_model = model_base
        instruct_model = f"{model_base}-Instruct"

        if base_model not in best_configs or instruct_model not in best_configs:
            print(f"⚠️  {model_base}: 缺少Base或Instruct配置")
            continue

        base_config = best_configs[base_model]
        instruct_config = best_configs[instruct_model]

        base_acc = base_config['metrics']['acc_mean']
        instruct_acc = instruct_config['metrics']['acc_mean']
        acc_diff = instruct_acc - base_acc

        base_ppl = base_config['metrics'].get('ppl', None)
        instruct_ppl = instruct_config['metrics'].get('ppl', None)

        print(f"\n{model_base}:")
        base_ppl_str = f"{base_ppl:.2f}" if base_ppl else 'N/A'
        instruct_ppl_str = f"{instruct_ppl:.2f}" if instruct_ppl else 'N/A'
        print(f"  Base    : ACC = {base_acc:.4f}, PPL = {base_ppl_str}, "
              f"Method = {base_config['pruning_method'].upper()}")
        print(f"  Instruct: ACC = {instruct_acc:.4f}, PPL = {instruct_ppl_str}, "
              f"Method = {instruct_config['pruning_method'].upper()}")
        print(f"  差异    : ACC {acc_diff:+.4f} ({acc_diff/base_acc*100:+.2f}%)")

        # 参数对比
        base_params = base_config['params']
        instruct_params = instruct_config['params']

        if base_params.get('taylor_seq_len') != instruct_params.get('taylor_seq_len'):
            print(f"  参数差异: seq_len {base_params.get('taylor_seq_len')} → "
                  f"{instruct_params.get('taylor_seq_len')}")
        if base_params.get('taylor_num_samples') != instruct_params.get('taylor_num_samples'):
            print(f"  参数差异: samples {base_params.get('taylor_num_samples')} → "
                  f"{instruct_params.get('taylor_num_samples')}")

    # ========== Part 4: 剪枝方法偏好统计 ==========
    print("\n" + "="*100)
    print("剪枝方法偏好统计")
    print("="*100)

    method_counts = {'taylor': 0, 'layerwise': 0, 'blockwise': 0}
    method_by_type = {
        'base': {'taylor': 0, 'layerwise': 0, 'blockwise': 0},
        'instruct': {'taylor': 0, 'layerwise': 0, 'blockwise': 0}
    }

    for model, config in best_configs.items():
        method = config['pruning_method']
        method_counts[method] += 1

        if '-Instruct' in model:
            method_by_type['instruct'][method] += 1
        else:
            method_by_type['base'][method] += 1

    print(f"\n总体分布:")
    for method in ['taylor', 'layerwise', 'blockwise']:
        count = method_counts[method]
        print(f"  {method.upper():<12}: {count}/6 模型 ({count/6*100:.1f}%)")

    print(f"\nBase 模型:")
    for method in ['taylor', 'layerwise', 'blockwise']:
        count = method_by_type['base'][method]
        print(f"  {method.upper():<12}: {count}/3 模型 ({count/3*100:.1f}%)")

    print(f"\nInstruct 模型:")
    for method in ['taylor', 'layerwise', 'blockwise']:
        count = method_by_type['instruct'][method]
        print(f"  {method.upper():<12}: {count}/3 模型 ({count/3*100:.1f}%)")

    # ========== Part 5: 参数分布统计 ==========
    print("\n" + "="*100)
    print("最佳参数分布统计")
    print("="*100)

    seq_lens = []
    num_samples = []

    for model, config in best_configs.items():
        params = config['params']
        if params.get('taylor_seq_len'):
            seq_lens.append(params['taylor_seq_len'])
        if params.get('taylor_num_samples'):
            num_samples.append(params['taylor_num_samples'])

    if seq_lens:
        print(f"\ntaylor_seq_len 分布:")
        seq_len_counts = {}
        for val in seq_lens:
            # 转换为int以处理可能的浮点数
            val = int(float(val)) if isinstance(val, (str, float)) else val
            seq_len_counts[val] = seq_len_counts.get(val, 0) + 1
        for val in sorted(seq_len_counts.keys()):
            count = seq_len_counts[val]
            print(f"  {int(val):4d}: {count}/6 模型 ({count/6*100:.1f}%)")

    if num_samples:
        print(f"\ntaylor_num_samples 分布:")
        samples_counts = {}
        for val in num_samples:
            # 转换为int以处理可能的浮点数
            val = int(float(val)) if isinstance(val, (str, float)) else val
            samples_counts[val] = samples_counts.get(val, 0) + 1
        for val in sorted(samples_counts.keys()):
            count = samples_counts[val]
            print(f"  {int(val):4d}: {count}/6 模型 ({count/6*100:.1f}%)")

    # ========== Part 6: 架构间对比（Llama vs Qwen vs Mistral）==========
    print("\n" + "="*100)
    print("模型架构对比 (平均 Base + Instruct)")
    print("="*100)

    arch_stats = {}
    for arch in ['Llama', 'Qwen', 'Mistral']:
        base = best_configs.get(arch)
        instruct = best_configs.get(f"{arch}-Instruct")

        accs = []
        ppls = []
        if base:
            accs.append(base['metrics']['acc_mean'])
            if base['metrics'].get('ppl'):
                ppls.append(base['metrics']['ppl'])
        if instruct:
            accs.append(instruct['metrics']['acc_mean'])
            if instruct['metrics'].get('ppl'):
                ppls.append(instruct['metrics']['ppl'])

        if accs:
            arch_stats[arch] = {
                'avg_acc': sum(accs) / len(accs),
                'avg_ppl': sum(ppls) / len(ppls) if ppls else None,
                'count': len(accs)
            }

    print(f"\n{'架构':<10} {'平均ACC':<12} {'平均PPL':<12} {'样本数':<10}")
    print("-"*50)
    for arch in ['Llama', 'Qwen', 'Mistral']:
        if arch in arch_stats:
            stats = arch_stats[arch]
            ppl_str = f"{stats['avg_ppl']:.2f}" if stats['avg_ppl'] else 'N/A'
            print(f"{arch:<10} {stats['avg_acc']:<12.4f} {ppl_str:<12} {stats['count']:<10}")

    # ========== Part 7: 保存综合分析结果 ==========
    output_dir = Path('results') / 'cross_model_analysis'
    output_dir.mkdir(exist_ok=True)

    # 保存完整对比表
    comparison_data = []
    for model in all_models:
        if model not in best_configs:
            continue

        config = best_configs[model]
        row = {
            'model': model.replace('-Instruct', ''),
            'type': 'Instruct' if '-Instruct' in model else 'Base',
            'pruning_method': config['pruning_method'],
            'acc_mean': config['metrics']['acc_mean'],
            'ppl': config['metrics'].get('ppl'),
            'taylor_seq_len': config['params'].get('taylor_seq_len'),
            'taylor_num_samples': config['params'].get('taylor_num_samples'),
            'grad_norm_ratio': config['metrics'].get('grad_norm_ratio'),
            'grad_mean_ratio': config['metrics'].get('grad_mean_ratio'),
            'extreme_pruning_layers': config['metrics'].get('extreme_pruning_layers'),
            'output_dir': config['output_dir']
        }
        comparison_data.append(row)

    # 保存为CSV
    csv_file = output_dir / 'all_models_best_configs.csv'
    if comparison_data:
        with open(csv_file, 'w', newline='') as f:
            fieldnames = comparison_data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(comparison_data)

    print(f"\n✓ 综合对比表已保存到: {csv_file}")

    # 保存JSON格式的统计数据
    analysis_summary = {
        'method_distribution': {
            'overall': method_counts,
            'base': method_by_type['base'],
            'instruct': method_by_type['instruct']
        },
        'parameter_distribution': {
            'taylor_seq_len': seq_len_counts if seq_lens else {},
            'taylor_num_samples': samples_counts if num_samples else {}
        },
        'architecture_comparison': arch_stats,
        'best_configs': best_configs
    }

    json_file = output_dir / 'analysis_summary.json'
    with open(json_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)

    print(f"✓ 统计摘要已保存到: {json_file}")
    print(f"\n✓ 跨模型综合分析完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    main()
