#!/usr/bin/env python3
"""
展示每个模型ACC前5的实验结果，并导出per-layer分析和可视化

用法:
    python param_search/acc-top5_results.py
"""

import csv
import json
import shutil
from pathlib import Path
from datetime import datetime


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
            if row.get('success') == 'True' and row.get('acc_mean'):
                results.append(row)

    # 按 acc_mean 降序排序
    results.sort(key=lambda x: float(x['acc_mean']), reverse=True)

    return results


def load_per_layer_analysis(model, method, seq_len, samples):
    """尝试加载对应实验的per-layer分析数据"""
    # 在for_finetuning目录中查找匹配的config
    for_finetuning_dir = Path('results') / 'for_finetuning' / model

    if not for_finetuning_dir.exists():
        return None

    # 遍历所有config目录
    for config_dir in for_finetuning_dir.iterdir():
        if not config_dir.is_dir():
            continue

        selection_info_file = config_dir / 'selection_info.json'
        if not selection_info_file.exists():
            continue

        # 读取selection_info.json并匹配参数
        try:
            with open(selection_info_file, 'r') as f:
                selection_info = json.load(f)

            # 匹配pruning_method和taylor参数
            if (selection_info.get('pruning_method') == method and
                str(selection_info.get('taylor_seq_len')) == str(seq_len) and
                str(selection_info.get('taylor_num_samples')) == str(samples)):

                # 找到匹配的config，加载分析数据
                analysis_dir = config_dir / 'analysis'
                if not analysis_dir.exists():
                    return None

                analysis_data = {
                    'config_name': config_dir.name,
                    'config_dir': config_dir,  # 保存config_dir路径用于后续复制visualization
                    'selection_info': selection_info,
                    'per_layer_summary': None,
                    'layer_importance': None,
                    'block_importance': None,
                    'gradient_diagnosis': None
                }

                # 读取per-layer文本摘要
                summary_file = analysis_dir / 'pruning_summary_by_layer.txt'
                if summary_file.exists():
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        analysis_data['per_layer_summary'] = f.read()

                # 读取层重要性JSON
                layer_imp_file = analysis_dir / 'layer_importance_loss.json'
                if layer_imp_file.exists():
                    with open(layer_imp_file, 'r') as f:
                        analysis_data['layer_importance'] = json.load(f)

                # 读取块重要性JSON
                block_imp_file = analysis_dir / 'block_importance_loss.json'
                if block_imp_file.exists():
                    with open(block_imp_file, 'r') as f:
                        analysis_data['block_importance'] = json.load(f)

                # 读取梯度诊断JSON
                grad_diag_file = analysis_dir / 'gradient_diagnosis.json'
                if grad_diag_file.exists():
                    with open(grad_diag_file, 'r') as f:
                        analysis_data['gradient_diagnosis'] = json.load(f)

                return analysis_data
        except Exception as e:
            print(f"Warning: Error loading analysis for {config_dir}: {e}")
            continue

    return None


def export_results(model, results, top_n=5):
    """导出per-layer分析和可视化文件"""
    output_dir = Path('param_search') / 'top_results' / 'acc' / model
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0

    for i, row in enumerate(results[:top_n], 1):
        method = row.get('pruning_method', 'N/A')
        seq_len = row.get('taylor_seq_len', 'N/A')
        samples = row.get('taylor_num_samples', 'N/A')

        # 尝试加载per-layer分析
        if method != 'N/A' and seq_len != 'N/A' and samples != 'N/A':
            analysis = load_per_layer_analysis(model, method, seq_len, samples)
            if analysis:
                # 保存per-layer summary文本
                if analysis.get('per_layer_summary'):
                    summary_file = output_dir / f'rank{i}_per_layer_summary.txt'
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        f.write(analysis['per_layer_summary'])

                # 复制visualization文件夹
                config_dir = analysis.get('config_dir')
                if config_dir:
                    viz_source = config_dir / 'visualization'
                    if viz_source.exists() and viz_source.is_dir():
                        viz_dest = output_dir / f'rank{i}_visualization'
                        # 如果目标文件夹已存在，先删除
                        if viz_dest.exists():
                            shutil.rmtree(viz_dest)
                        # 复制整个文件夹
                        shutil.copytree(viz_source, viz_dest)
                        exported_count += 1

    return exported_count


def create_summary_folder(models, metric='acc'):
    """创建summary文件夹并复制cross_model_best.json和各模型的pruning_ratio.png"""
    summary_dir = Path('param_search') / 'top_results' / metric / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 移动cross_model_best.json到summary文件夹
    cross_model_file = Path('param_search') / 'top_results' / metric / 'cross_model_best.json'
    if cross_model_file.exists():
        summary_json = summary_dir / 'cross_model_best.json'
        shutil.copy2(cross_model_file, summary_json)
        cross_model_file.unlink()  # 删除原文件

    # 复制每个模型的pruning_ratio.png
    copied_count = 0
    for model in models:
        # 从rank1_visualization复制pruning_ratio.png
        viz_dir = Path('param_search') / 'top_results' / metric / model / 'rank1_visualization'
        pruning_ratio_src = viz_dir / 'pruning_ratio.png'

        if pruning_ratio_src.exists():
            pruning_ratio_dest = summary_dir / f'{model}_pruning_ratio.png'
            shutil.copy2(pruning_ratio_src, pruning_ratio_dest)
            copied_count += 1

    return copied_count


def display_top5(model, results, top_n=5):
    """显示前N个结果"""
    if not results:
        print(f"⚠️  {model}: 没有找到结果")
        return

    model_display = model.replace('-', ' ')
    print(f"\n{'='*100}")
    print(f"{model_display} - ACC Top {top_n}")
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

    # 显示详细的7个任务ACC（仅前3名）
    print(f"\n详细任务ACC（前3名）:")
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
    print("所有模型 ACC Top 5 结果")
    print("="*100)

    # 为每个模型处理并导出结果
    for model in models:
        results = load_and_rank_results(model)
        display_top5(model, results, top_n=5)

        # 导出per-layer分析和可视化文件
        if results:
            exported_count = export_results(model, results, top_n=5)
            print(f"  ✓ 已导出 {exported_count} 个可视化文件夹到 param_search/top_results/acc/{model}/")

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

    # 按ACC降序排序
    best_results.sort(key=lambda x: x['acc'], reverse=True)

    print(f"\n{'排名':<6} {'模型':<20} {'方法':<12} {'seq_len':<10} {'samples':<10} {'ACC':<12} {'PPL':<10}")
    print("-"*100)

    for i, result in enumerate(best_results, 1):
        model_display = result['model'].replace('-', ' ')
        seq_len_str = f"{int(float(result['seq_len']))}" if result['seq_len'] and result['seq_len'] != 'N/A' else 'N/A'
        samples_str = f"{int(float(result['samples']))}" if result['samples'] and result['samples'] != 'N/A' else 'N/A'
        ppl_str = f"{result['ppl']:.2f}" if result['ppl'] else 'N/A'

        print(f"#{i:<5} {model_display:<20} {result['method']:<12} {seq_len_str:<10} {samples_str:<10} "
              f"{result['acc']:<12.4f} {ppl_str:<10}")

    # 导出跨模型对比JSON
    cross_model_output = Path('param_search') / 'top_results' / 'acc'
    cross_model_output.mkdir(parents=True, exist_ok=True)
    cross_model_file = cross_model_output / 'cross_model_best.json'

    cross_model_data = {
        'metric': 'accuracy',
        'generated_at': datetime.now().isoformat(),
        'best_results': best_results
    }

    with open(cross_model_file, 'w', encoding='utf-8') as f:
        json.dump(cross_model_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 跨模型对比已导出: {cross_model_file}")

    # 创建summary文件夹
    print(f"\n{'='*100}")
    print("生成Summary文件夹")
    print(f"{'='*100}\n")

    copied_count = create_summary_folder(models, metric='acc')
    print(f"✓ Summary文件夹已创建: param_search/top_results/acc/summary/")
    print(f"✓ 已复制 cross_model_best.json")
    print(f"✓ 已复制 {copied_count} 个模型的 pruning_ratio.png")

    print(f"\n✓ 分析完成！所有结果已保存到 param_search/top_results/acc/\n")


if __name__ == '__main__':
    main()
