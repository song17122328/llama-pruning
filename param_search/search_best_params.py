#!/usr/bin/env python3
"""
参数网格搜索脚本 - 寻找最佳的序列长度和样本数组合

用法:
    python search_best_params.py --config configs/mistral_search.json

配置文件格式:
{
    "base_model": "/path/to/model",
    "pruning_ratio": 0.2,
    "output_base": "results/param_search",
    "search_params": {
        "taylor_seq_len": [16, 32, 64, 128],
        "taylor_num_samples": [128, 256, 512]
    },
    "other_args": {
        "dataset": "c4",
        "temperature": 0.0,
        "importance_method": "taylor"
    }
}
"""

import os
import json
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from datetime import datetime
import time


def run_pruning_experiment(
    base_model,
    output_name,
    pruning_ratio,
    taylor_seq_len,
    taylor_num_samples,
    layer_importance_seq_len=None,
    layer_importance_num_samples=None,
    block_importance_seq_len=None,
    block_importance_num_samples=None,
    other_args=None
):
    """运行一次剪枝实验"""

    # 如果未指定，默认使用 taylor 参数
    if layer_importance_seq_len is None:
        layer_importance_seq_len = taylor_seq_len
    if layer_importance_num_samples is None:
        layer_importance_num_samples = taylor_num_samples // 5  # 默认1/5
    if block_importance_seq_len is None:
        block_importance_seq_len = taylor_seq_len
    if block_importance_num_samples is None:
        block_importance_num_samples = taylor_num_samples // 5

    # 构建命令
    cmd = [
        "python", "run_global_pruning.py",
        "--base_model", base_model,
        "--output_name", output_name,
        "--pruning_ratio", str(pruning_ratio),
        "--taylor_seq_len", str(taylor_seq_len),
        "--taylor_num_samples", str(taylor_num_samples),
        "--layer_importance_seq_len", str(layer_importance_seq_len),
        "--layer_importance_num_samples", str(layer_importance_num_samples),
        "--block_importance_seq_len", str(block_importance_seq_len),
        "--block_importance_num_samples", str(block_importance_num_samples),
    ]

    # 添加其他参数
    if other_args:
        for key, value in other_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))

    print(f"\n{'='*80}")
    print(f"运行实验: {output_name}")
    print(f"参数: taylor_seq_len={taylor_seq_len}, taylor_num_samples={taylor_num_samples}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # 运行命令
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed_time = time.time() - start_time

    return {
        "success": result.returncode == 0,
        "elapsed_time": elapsed_time,
        "return_code": result.returncode
    }


def extract_results(output_dir):
    """从输出目录提取评估结果（包括梯度统计和详细 ACC）"""
    results = {
        "ppl": None,
        "acc_mean": None,
        "params": None,
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
                if "zeroshot" in metrics:
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
                results["params"] = comp_data["pruned_model"]["total_params"]

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
    parser = argparse.ArgumentParser(description='参数网格搜索')
    parser.add_argument('--config', type=str, required=True,
                       help='搜索配置文件（JSON格式）')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='最大实验次数（用于调试）')
    parser.add_argument('--resume', action='store_true',
                       help='从之前的搜索继续（跳过已完成的实验）')

    args = parser.parse_args()

    # 读取配置
    with open(args.config, 'r') as f:
        config = json.load(f)

    base_model = config['base_model']
    pruning_ratio = config['pruning_ratio']
    output_base = config['output_base']
    search_params = config['search_params']
    other_args = config.get('other_args', {})

    # 生成所有参数组合
    param_names = list(search_params.keys())
    param_values = list(search_params.values())
    param_combinations = list(product(*param_values))

    print(f"总共 {len(param_combinations)} 个参数组合")
    print(f"搜索参数: {param_names}")

    # 创建结果记录
    results_file = Path(output_base) / "search_results.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # 如果是续传，读取已有结果
    completed_experiments = set()
    if args.resume and results_file.exists():
        df_existing = pd.read_csv(results_file)
        for _, row in df_existing.iterrows():
            key = tuple(row[param_names].values)
            completed_experiments.add(key)
        print(f"已完成 {len(completed_experiments)} 个实验，将跳过")

    # 运行实验
    all_results = []

    for i, param_combo in enumerate(param_combinations):
        # 检查是否达到最大实验次数
        if args.max_experiments and i >= args.max_experiments:
            print(f"\n达到最大实验次数 {args.max_experiments}，停止搜索")
            break

        # 检查是否已完成
        if param_combo in completed_experiments:
            print(f"\n跳过已完成的实验 {i+1}/{len(param_combinations)}: {dict(zip(param_names, param_combo))}")
            continue

        # 构建参数字典
        params = dict(zip(param_names, param_combo))

        # 生成输出目录名称
        param_str = "_".join([f"{k}{v}" for k, v in params.items()])
        output_name = f"{output_base}/exp_{i+1:03d}_{param_str}"

        print(f"\n{'='*80}")
        print(f"实验 {i+1}/{len(param_combinations)}")
        print(f"参数: {params}")
        print(f"{'='*80}")

        # 运行实验
        run_result = run_pruning_experiment(
            base_model=base_model,
            output_name=output_name,
            pruning_ratio=pruning_ratio,
            other_args=other_args,
            **params
        )

        # 提取结果
        if run_result["success"]:
            output_dir = f"results/{output_name}"
            metrics = extract_results(output_dir)

            result_entry = {
                **params,
                "output_dir": output_dir,
                "ppl": metrics["ppl"],
                "acc_mean": metrics["acc_mean"],
                "acc_boolq": metrics.get("acc_boolq"),
                "acc_piqa": metrics.get("acc_piqa"),
                "acc_hellaswag": metrics.get("acc_hellaswag"),
                "acc_winogrande": metrics.get("acc_winogrande"),
                "acc_arc_easy": metrics.get("acc_arc_easy"),
                "acc_arc_challenge": metrics.get("acc_arc_challenge"),
                "acc_openbookqa": metrics.get("acc_openbookqa"),
                "params_count": metrics["params"],
                "pruning_ratio": metrics["pruning_ratio"],
                "grad_mean_ratio": metrics.get("grad_mean_ratio"),
                "grad_norm_ratio": metrics.get("grad_norm_ratio"),
                "grad_std_ratio": metrics.get("grad_std_ratio"),
                "grad_max_ratio": metrics.get("grad_max_ratio"),
                "grad_mean_range": metrics.get("grad_mean_range"),
                "grad_norm_range": metrics.get("grad_norm_range"),
                "extreme_pruning_layers": metrics.get("extreme_pruning_layers"),
                "elapsed_time": run_result["elapsed_time"],
                "success": True
            }
        else:
            result_entry = {
                **params,
                "output_dir": f"results/{output_name}",
                "ppl": None,
                "acc_mean": None,
                "acc_boolq": None,
                "acc_piqa": None,
                "acc_hellaswag": None,
                "acc_winogrande": None,
                "acc_arc_easy": None,
                "acc_arc_challenge": None,
                "acc_openbookqa": None,
                "params_count": None,
                "pruning_ratio": None,
                "grad_mean_ratio": None,
                "grad_norm_ratio": None,
                "grad_std_ratio": None,
                "grad_max_ratio": None,
                "grad_mean_range": None,
                "grad_norm_range": None,
                "extreme_pruning_layers": None,
                "elapsed_time": run_result["elapsed_time"],
                "success": False
            }

        all_results.append(result_entry)

        # 实时保存结果
        df_results = pd.DataFrame(all_results)

        # 如果是续传，合并之前的结果
        if args.resume and results_file.exists():
            df_existing = pd.read_csv(results_file)
            df_results = pd.concat([df_existing, df_results], ignore_index=True)

        df_results.to_csv(results_file, index=False)
        print(f"\n✓ 结果已保存到 {results_file}")

        # 显示当前最佳结果
        if "acc_mean" in df_results.columns:
            df_valid = df_results[df_results['success'] == True]
            if not df_valid.empty and df_valid['acc_mean'].notna().any():
                best_idx = df_valid['acc_mean'].idxmax()
                best_row = df_valid.loc[best_idx]
                print(f"\n当前最佳配置:")
                for param_name in param_names:
                    print(f"  {param_name}: {best_row[param_name]}")
                print(f"  ACC (mean): {best_row['acc_mean']:.4f}")
                print(f"  PPL: {best_row['ppl']:.2f}")
                if best_row.get('grad_norm_ratio'):
                    print(f"  梯度范数比率: {best_row['grad_norm_ratio']:.2f}x")
                if best_row.get('extreme_pruning_layers') is not None:
                    print(f"  极端剪枝层数: {best_row['extreme_pruning_layers']}")

    # 生成最终报告
    print(f"\n{'='*80}")
    print(f"搜索完成！")
    print(f"{'='*80}\n")

    df_results = pd.read_csv(results_file)
    df_valid = df_results[df_results['success'] == True]

    if not df_valid.empty and df_valid['acc_mean'].notna().any():
        # 按 ACC 排序
        df_sorted = df_valid.sort_values('acc_mean', ascending=False)

        print("Top 5 配置（按平均 ACC 排序）:")
        print("-" * 80)
        for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
            print(f"\n#{i+1}")
            for param_name in param_names:
                print(f"  {param_name}: {row[param_name]}")
            print(f"  ACC (mean): {row['acc_mean']:.4f}")
            print(f"  PPL: {row['ppl']:.2f}")
            if row.get('grad_norm_ratio'):
                print(f"  梯度范数比率: {row['grad_norm_ratio']:.2f}x")
            if row.get('extreme_pruning_layers') is not None:
                print(f"  极端剪枝层数: {row['extreme_pruning_layers']}")
            print(f"  输出目录: {row['output_dir']}")

        # 保存最佳配置
        best_config_file = Path(output_base) / "best_config.json"
        best_row = df_sorted.iloc[0]

        # 提取所有 7 个任务的 ACC
        acc_details = {}
        for task in ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']:
            col_name = f'acc_{task}'
            if col_name in best_row and pd.notna(best_row[col_name]):
                acc_details[task] = best_row[col_name]

        # 提取梯度统计
        grad_stats = {}
        for stat in ['grad_mean_ratio', 'grad_norm_ratio', 'grad_std_ratio', 'grad_max_ratio',
                     'grad_mean_range', 'grad_norm_range', 'extreme_pruning_layers']:
            if stat in best_row and pd.notna(best_row[stat]):
                # 转换为 Python 原生类型以支持 JSON 序列化
                val = best_row[stat]
                if isinstance(val, (np.integer, np.floating)):
                    grad_stats[stat] = val.item()
                else:
                    grad_stats[stat] = val

        # 辅助函数：将 numpy 类型转换为 Python 原生类型
        def to_python_type(val):
            if pd.isna(val):
                return None
            elif isinstance(val, (np.integer, np.floating)):
                return val.item()
            elif isinstance(val, np.ndarray):
                return val.tolist()
            else:
                return val

        best_config = {
            "params": {k: to_python_type(best_row[k]) for k in param_names},
            "metrics": {
                "acc_mean": to_python_type(best_row['acc_mean']),
                "acc_details": {k: to_python_type(v) for k, v in acc_details.items()},
                "ppl": to_python_type(best_row['ppl']),
                "pruning_ratio": to_python_type(best_row['pruning_ratio']),
                "gradient_statistics": grad_stats
            },
            "output_dir": best_row['output_dir']
        }

        with open(best_config_file, 'w') as f:
            json.dump(best_config, f, indent=2)

        print(f"\n✓ 最佳配置已保存到 {best_config_file}")
    else:
        print("没有成功的实验结果")


if __name__ == '__main__':
    main()
