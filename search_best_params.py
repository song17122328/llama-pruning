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
    """从输出目录提取评估结果"""
    results = {
        "ppl": None,
        "acc": None,
        "params": None,
        "pruning_ratio": None
    }

    # 读取评估结果
    eval_file = Path(output_dir) / "evaluation" / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)

            # 提取 PPL（WikiText2）
            if "ppl" in eval_data and "wikitext2" in eval_data["ppl"]:
                results["ppl"] = eval_data["ppl"]["wikitext2"]

            # 提取平均 ACC
            if "zeroshot" in eval_data and "results" in eval_data["zeroshot"]:
                accs = []
                for task, task_results in eval_data["zeroshot"]["results"].items():
                    if "acc" in task_results:
                        accs.append(task_results["acc"])
                    elif "acc_norm" in task_results:
                        accs.append(task_results["acc_norm"])

                if accs:
                    results["acc"] = sum(accs) / len(accs)

    # 读取模型对比结果
    model_comp_file = Path(output_dir) / "analysis" / "model_comparison.json"
    if model_comp_file.exists():
        with open(model_comp_file, 'r') as f:
            comp_data = json.load(f)

            if "pruned_model" in comp_data and "total_params" in comp_data["pruned_model"]:
                results["params"] = comp_data["pruned_model"]["total_params"]

            if "total_params" in comp_data and "reduction_ratio" in comp_data["total_params"]:
                results["pruning_ratio"] = comp_data["total_params"]["reduction_ratio"]

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
                "acc": metrics["acc"],
                "params_count": metrics["params"],
                "pruning_ratio": metrics["pruning_ratio"],
                "elapsed_time": run_result["elapsed_time"],
                "success": True
            }
        else:
            result_entry = {
                **params,
                "output_dir": f"results/{output_name}",
                "ppl": None,
                "acc": None,
                "params_count": None,
                "pruning_ratio": None,
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
        if "acc" in df_results.columns:
            df_valid = df_results[df_results['success'] == True]
            if not df_valid.empty and df_valid['acc'].notna().any():
                best_idx = df_valid['acc'].idxmax()
                best_row = df_valid.loc[best_idx]
                print(f"\n当前最佳配置:")
                for param_name in param_names:
                    print(f"  {param_name}: {best_row[param_name]}")
                print(f"  ACC: {best_row['acc']:.4f}")
                print(f"  PPL: {best_row['ppl']:.2f}")

    # 生成最终报告
    print(f"\n{'='*80}")
    print(f"搜索完成！")
    print(f"{'='*80}\n")

    df_results = pd.read_csv(results_file)
    df_valid = df_results[df_results['success'] == True]

    if not df_valid.empty and df_valid['acc'].notna().any():
        # 按 ACC 排序
        df_sorted = df_valid.sort_values('acc', ascending=False)

        print("Top 5 配置（按 ACC 排序）:")
        print("-" * 80)
        for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
            print(f"\n#{i+1}")
            for param_name in param_names:
                print(f"  {param_name}: {row[param_name]}")
            print(f"  ACC: {row['acc']:.4f}")
            print(f"  PPL: {row['ppl']:.2f}")
            print(f"  输出目录: {row['output_dir']}")

        # 保存最佳配置
        best_config_file = Path(output_base) / "best_config.json"
        best_row = df_sorted.iloc[0]
        best_config = {
            "params": {k: best_row[k] for k in param_names},
            "metrics": {
                "acc": best_row['acc'],
                "ppl": best_row['ppl'],
                "pruning_ratio": best_row['pruning_ratio']
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
