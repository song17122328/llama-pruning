#!/usr/bin/env python3
"""
重新提取搜索结果

用于修复 extract_results 函数后，重新从已完成的实验中提取所有指标。
"""

import json
import csv
import sys
from pathlib import Path


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


def extract_results(output_dir):
    """从输出目录提取评估结果（包括梯度统计和详细 ACC）"""
    results = {
        "ppl": None,
        "acc_mean": None,
        "params_count": None,
        "pruning_ratio": None,
        # 梯度统计指标
        "grad_mean_ratio": None,
        "grad_norm_ratio": None,
        "grad_std_ratio": None,
        "grad_max_ratio": None,
        "grad_mean_range": None,
        "grad_norm_range": None,
        "extreme_pruning_layers": None,
        # 详细 ACC 指标
        "acc_boolq": None,
        "acc_piqa": None,
        "acc_hellaswag": None,
        "acc_winogrande": None,
        "acc_arc_easy": None,
        "acc_arc_challenge": None,
        "acc_openbookqa": None,
        "elapsed_time": None,
        "success": True,
    }

    # 读取评估结果
    eval_file = Path(output_dir) / "evaluation" / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)

            # 提取 PPL（WikiText2）
            # 实际路径: metrics.ppl["wikitext2 (wikitext-2-raw-v1)"]
            if "metrics" in eval_data:
                metrics = eval_data["metrics"]

                # 提取 PPL
                if "ppl" in metrics:
                    ppl_data = metrics["ppl"]
                    # 尝试各种 wikitext2 的键名
                    for key in ppl_data.keys():
                        if "wikitext2" in key.lower() or "wikitext-2" in key.lower():
                            results["ppl"] = ppl_data[key]
                            break

                # 提取所有 7 个 zero-shot 任务的单独 ACC
                # 实际路径: metrics.zeroshot.<task>.accuracy
                if "zeroshot" in metrics and metrics["zeroshot"] is not None:
                    zeroshot_data = metrics["zeroshot"]
                    accs = []

                    for task_name, task_data in zeroshot_data.items():
                        # 提取任务的 accuracy
                        if isinstance(task_data, dict) and "accuracy" in task_data:
                            task_acc = task_data["accuracy"]
                            accs.append(task_acc)

                            # 保存单独任务的 ACC
                            task_lower = task_name.lower()
                            if "boolq" in task_lower:
                                results["acc_boolq"] = task_acc
                            elif "piqa" in task_lower:
                                results["acc_piqa"] = task_acc
                            elif "hellaswag" in task_lower:
                                results["acc_hellaswag"] = task_acc
                            elif "winogrande" in task_lower:
                                results["acc_winogrande"] = task_acc
                            elif "arc_easy" in task_lower or "arc-easy" in task_lower:
                                results["acc_arc_easy"] = task_acc
                            elif "arc_challenge" in task_lower or "arc-challenge" in task_lower:
                                results["acc_arc_challenge"] = task_acc
                            elif "openbookqa" in task_lower:
                                results["acc_openbookqa"] = task_acc

                    # 计算平均 ACC（如果有任务结果）
                    if accs:
                        results["acc_mean"] = sum(accs) / len(accs)

                    # 也尝试直接读取平均值
                    if "avg_zeroshot_acc" in metrics:
                        results["acc_mean"] = metrics["avg_zeroshot_acc"]

    # 读取模型对比结果
    model_comp_file = Path(output_dir) / "analysis" / "model_comparison.json"
    if model_comp_file.exists():
        with open(model_comp_file, 'r') as f:
            comp_data = json.load(f)

            if "pruned_model" in comp_data and "total_params" in comp_data["pruned_model"]:
                results["params_count"] = comp_data["pruned_model"]["total_params"]

            if "total_params" in comp_data and "reduction_ratio" in comp_data["total_params"]:
                results["pruning_ratio"] = comp_data["total_params"]["reduction_ratio"]

    # 读取梯度诊断信息（包含梯度统计和极端剪枝层信息）
    grad_diag_file = Path(output_dir) / "analysis" / "gradient_diagnosis.json"
    if grad_diag_file.exists():
        with open(grad_diag_file, 'r') as f:
            diag_data = json.load(f)

            # 提取梯度统计信息（实际路径：gradient_statistics）
            if "gradient_statistics" in diag_data:
                grad_stats = diag_data["gradient_statistics"]

                # 提取梯度均值比率
                if "mean_ratio" in grad_stats:
                    results["grad_mean_ratio"] = grad_stats["mean_ratio"]

                # 提取梯度范数比率
                if "norm_ratio" in grad_stats:
                    results["grad_norm_ratio"] = grad_stats["norm_ratio"]

                # 提取梯度标准差比率（如果有）
                if "std_ratio" in grad_stats:
                    results["grad_std_ratio"] = grad_stats["std_ratio"]

                # 提取梯度最大值比率（如果有）
                if "max_ratio" in grad_stats:
                    results["grad_max_ratio"] = grad_stats["max_ratio"]

                # 提取梯度均值范围
                if "mean_range" in grad_stats:
                    mean_range = grad_stats["mean_range"]
                    if isinstance(mean_range, list) and len(mean_range) == 2:
                        # 格式: [min, max]
                        results["grad_mean_range"] = mean_range[1] - mean_range[0]
                    elif isinstance(mean_range, str):
                        # 格式: "8.3733e-02 ~ 9.0496e-01"
                        parts = mean_range.split('~')
                        if len(parts) == 2:
                            min_val = float(parts[0].strip())
                            max_val = float(parts[1].strip())
                            results["grad_mean_range"] = max_val - min_val

                # 提取梯度范数范围
                if "norm_range" in grad_stats:
                    norm_range = grad_stats["norm_range"]
                    if isinstance(norm_range, list) and len(norm_range) == 2:
                        # 格式: [min, max]
                        results["grad_norm_range"] = norm_range[1] - norm_range[0]
                    elif isinstance(norm_range, str):
                        parts = norm_range.split('~')
                        if len(parts) == 2:
                            min_val = float(parts[0].strip())
                            max_val = float(parts[1].strip())
                            results["grad_norm_range"] = max_val - min_val

            # 提取极端剪枝层数量
            if "num_extreme_layers" in diag_data:
                results["extreme_pruning_layers"] = diag_data["num_extreme_layers"]
            elif "extreme_pruning_layers" in diag_data:
                extreme_layers = diag_data["extreme_pruning_layers"]
                if isinstance(extreme_layers, list):
                    results["extreme_pruning_layers"] = len(extreme_layers)
                elif isinstance(extreme_layers, int):
                    results["extreme_pruning_layers"] = extreme_layers

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='重新提取搜索结果')
    parser.add_argument('--search_dir', type=str, required=True,
                       help='搜索结果目录（如 results/search_mistral_20）')
    args = parser.parse_args()

    search_dir = Path(args.search_dir)
    if not search_dir.exists():
        print(f"错误: 目录不存在 {search_dir}")
        return

    # 查找所有实验目录
    exp_dirs = sorted([d for d in search_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')])

    if not exp_dirs:
        print(f"未找到实验目录（exp_*）在 {search_dir}")
        return

    print(f"找到 {len(exp_dirs)} 个实验目录")

    # 提取所有结果
    all_results = []
    for exp_dir in exp_dirs:
        print(f"提取 {exp_dir.name} ...")

        # 从目录名解析参数（使用统一的提取函数）
        params = extract_params_from_path(exp_dir.name)

        # 提取结果
        results = extract_results(str(exp_dir))

        # 合并参数和结果
        results.update(params)
        results['output_dir'] = str(exp_dir)

        all_results.append(results)

    # 确定所有列，并按search_best_params.py的顺序排列
    all_keys = set()
    for result in all_results:
        all_keys.update(result.keys())

    # 定义列顺序（与search_best_params.py完全一致）
    param_cols = sorted([col for col in all_keys if col.startswith('taylor_') or col.startswith('layer_') or col.startswith('block_')])

    # 固定列顺序
    fixed_cols = [
        'output_dir', 'ppl', 'acc_mean',
        'acc_boolq', 'acc_piqa', 'acc_hellaswag', 'acc_winogrande',
        'acc_arc_easy', 'acc_arc_challenge', 'acc_openbookqa',
        'params_count', 'pruning_ratio',
        'grad_mean_ratio', 'grad_norm_ratio', 'grad_std_ratio', 'grad_max_ratio',
        'grad_mean_range', 'grad_norm_range', 'extreme_pruning_layers',
        'elapsed_time', 'success'
    ]

    # 组合列顺序：参数列在前，其他列在后（与search_best_params.py一致）
    all_cols = param_cols + [col for col in fixed_cols if col in all_keys]

    # 保存为 CSV
    output_file = search_dir / "search_results.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for result in all_results:
            writer.writerow({k: result.get(k, '') for k in all_cols})

    print(f"\n✓ 已保存结果到 {output_file}")
    print(f"✓ 共 {len(all_results)} 个实验")

    # 统计有效数据
    ppl_count = sum(1 for r in all_results if r.get('ppl') is not None)
    acc_count = sum(1 for r in all_results if r.get('acc_mean') is not None)
    grad_norm_count = sum(1 for r in all_results if r.get('grad_norm_ratio') is not None)
    grad_mean_count = sum(1 for r in all_results if r.get('grad_mean_ratio') is not None)
    extreme_count = sum(1 for r in all_results if r.get('extreme_pruning_layers') is not None)

    print(f"\n数据完整性:")
    print(f"  - PPL:                  {ppl_count}/{len(all_results)}")
    print(f"  - ACC mean:             {acc_count}/{len(all_results)}")
    print(f"  - Grad norm ratio:      {grad_norm_count}/{len(all_results)}")
    print(f"  - Grad mean ratio:      {grad_mean_count}/{len(all_results)}")
    print(f"  - Extreme pruning:      {extreme_count}/{len(all_results)}")

    # 找出最佳配置
    valid_results = [r for r in all_results if r.get('acc_mean') is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['acc_mean'])

        print(f"\n最佳配置 (最高 ACC):")
        for col in param_cols:
            if col in best_result:
                print(f"  - {col}: {best_result[col]}")
        print(f"  - ACC mean: {best_result['acc_mean']:.4f}")
        if best_result.get('ppl') is not None:
            print(f"  - PPL: {best_result['ppl']:.2f}")
        if best_result.get('grad_norm_ratio') is not None:
            print(f"  - Grad norm ratio: {best_result['grad_norm_ratio']:.2f}")

        # 提取所有 7 个任务的 ACC（与search_best_params.py一致）
        acc_details = {}
        for task in ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']:
            col_name = f'acc_{task}'
            if best_result.get(col_name) is not None:
                acc_details[task] = float(best_result[col_name])

        # 提取梯度统计（与search_best_params.py一致）
        grad_stats = {}
        for stat in ['grad_mean_ratio', 'grad_norm_ratio', 'grad_std_ratio', 'grad_max_ratio',
                     'grad_mean_range', 'grad_norm_range', 'extreme_pruning_layers']:
            if best_result.get(stat) is not None:
                val = best_result[stat]
                # 确保是Python原生类型
                if isinstance(val, (int, float)):
                    grad_stats[stat] = float(val) if isinstance(val, float) else int(val)
                else:
                    grad_stats[stat] = val

        # 保存最佳配置（与search_best_params.py格式完全一致）
        best_config = {
            "params": {col: best_result[col] for col in param_cols if col in best_result},
            "metrics": {
                "acc_mean": float(best_result['acc_mean']),
                "acc_details": acc_details,
                "ppl": float(best_result['ppl']) if best_result.get('ppl') is not None else None,
                "pruning_ratio": float(best_result['pruning_ratio']) if best_result.get('pruning_ratio') is not None else None,
                "gradient_statistics": grad_stats
            },
            "output_dir": best_result['output_dir']
        }

        best_config_file = search_dir / "best_config.json"
        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\n✓ 已保存最佳配置到 {best_config_file}")


if __name__ == '__main__':
    main()
