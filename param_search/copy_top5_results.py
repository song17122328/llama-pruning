#!/usr/bin/env python3

"""

复制每个模型的Top 5实验结果到专用目录

 

用法:

    python param_search/copy_top5_results.py

"""

 

import csv

import shutil

from pathlib import Path

 

 

def load_and_rank_results(model):

    """加载模型结果并按ACC排序"""

    csv_file = Path('results') / f'consolidated_{model}_20' / 'all_methods_results.csv'

 

    if not csv_file.exists():

        print(f"⚠️  {model}: CSV文件不存在")

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

 

 

def copy_experiment_dirs(source_dir, dest_dir, folders_to_copy):

    """复制指定的文件夹"""

    source_path = Path(source_dir)

    dest_path = Path(dest_dir)

 

    if not source_path.exists():

        print(f"    ⚠️  源目录不存在: {source_dir}")

        return False

 

    # 创建目标目录

    dest_path.mkdir(parents=True, exist_ok=True)

 

    copied_count = 0

    for folder in folders_to_copy:

        src_folder = source_path / folder

        dst_folder = dest_path / folder

 

        if src_folder.exists():

            # 如果目标已存在，先删除

            if dst_folder.exists():

                shutil.rmtree(dst_folder)

 

            # 复制文件夹

            shutil.copytree(src_folder, dst_folder)

            copied_count += 1

        else:

            print(f"    ⚠️  文件夹不存在: {src_folder}")

 

    return copied_count > 0

 

 

def copy_top5_for_model(model, top_n=5):

    """复制某个模型的Top N结果"""

    results = load_and_rank_results(model)

 

    if not results:

        print(f"\n⚠️  {model}: 没有找到结果，跳过")

        return

 

    print(f"\n{'='*80}")

    print(f"处理模型: {model}")

    print(f"{'='*80}")

 

    # 创建模型的Top5目录

    top5_dir = Path('results') / f'top5_{model}_20'

 

    # 需要复制的文件夹

    folders_to_copy = ['visualization', 'logs', 'evaluation', 'analysis']

 

    success_count = 0

    for i, row in enumerate(results[:top_n], 1):

        output_dir = row['output_dir']

        acc = float(row['acc_mean'])

        method = row.get('pruning_method', 'unknown').upper()

        seq_len = row.get('taylor_seq_len', 'N/A')

        samples = row.get('taylor_num_samples', 'N/A')

 

        # 格式化参数显示

        seq_len_str = f"{int(float(seq_len))}" if seq_len and seq_len not in ['', 'N/A'] else 'N/A'

        samples_str = f"{int(float(samples))}" if samples and samples not in ['', 'N/A'] else 'N/A'

 

        print(f"\n#{i} - ACC: {acc:.4f} - {method} (seq_len={seq_len_str}, samples={samples_str})")

        print(f"  源目录: {output_dir}")

 

        # 获取实验目录名称

        exp_name = Path(output_dir).name

 

        # 目标目录：添加排名前缀

        dest_dir = top5_dir / f"rank{i:02d}_{exp_name}"

        print(f"  目标目录: {dest_dir}")

 

        # 复制指定文件夹

        if copy_experiment_dirs(output_dir, dest_dir, folders_to_copy):

            print(f"  ✓ 复制成功")

            success_count += 1

 

            # 创建一个描述文件

            info_file = dest_dir / 'rank_info.txt'

            with open(info_file, 'w') as f:

                f.write(f"排名: #{i}\n")

                f.write(f"模型: {model}\n")

                f.write(f"剪枝方法: {method}\n")

                f.write(f"ACC (平均): {acc:.6f}\n")

                f.write(f"PPL: {row.get('ppl', 'N/A')}\n")

                f.write(f"taylor_seq_len: {seq_len_str}\n")

                f.write(f"taylor_num_samples: {samples_str}\n")

                f.write(f"梯度范数比: {row.get('grad_norm_ratio', 'N/A')}\n")

                f.write(f"源目录: {output_dir}\n")

                f.write(f"\n7个任务的ACC:\n")

 

                tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']

                for task in tasks:

                    col_name = f'acc_{task}'

                    if col_name in row and row[col_name]:

                        f.write(f"  {task:15s}: {float(row[col_name]):.6f}\n")

        else:

            print(f"  ✗ 复制失败")

 

    print(f"\n✓ {model} 完成：成功复制 {success_count}/{top_n} 个实验")

    print(f"  保存位置: {top5_dir}")

 

 

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

    print("复制所有模型的Top 5实验结果")

    print("="*80)

    print("\n将复制以下文件夹：")

    print("  - visualization/")

    print("  - logs/")

    print("  - evaluation/")

    print("  - analysis/")

 

    total_success = 0

    total_attempts = 0

 

    for model in models:

        copy_top5_for_model(model, top_n=5)

        total_attempts += 5

 

    print("\n" + "="*80)

    print("所有模型处理完成")

    print("="*80)

    print(f"\n结果保存在: results/top5_*_20/ 目录")

    print("\n每个实验目录包含:")

    print("  - rank_info.txt: 排名和配置信息")

    print("  - visualization/: 可视化图表")

    print("  - logs/: 训练日志")

    print("  - evaluation/: 评估结果")

    print("  - analysis/: 分析数据")

    print("\n✓ 完成！\n")

 

 

if __name__ == '__main__':

    main()