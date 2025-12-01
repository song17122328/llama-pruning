#!/usr/bin/env python3
"""
复制每个模型的最佳实验结果到专门的文件夹

用法:
    python param_search/copy_best_results.py --model llama
    python param_search/copy_best_results.py --model qwen
    python param_search/copy_best_results.py --model mistral
    python param_search/copy_best_results.py --all  # 处理所有模型
"""

import json
import shutil
import argparse
from pathlib import Path


def copy_best_result(model):
    """复制指定模型的最佳实验结果"""
    # 读取全局最佳配置
    config_file = Path('results') / f'consolidated_{model}_20' / 'global_best_config.json'

    if not config_file.exists():
        print(f"❌ 未找到配置文件: {config_file}")
        print(f"   请先运行: python param_search/consolidate_model_results.py --model {model}")
        return False

    with open(config_file, 'r') as f:
        config = json.load(f)

    source_dir = Path(config['output_dir'])

    if not source_dir.exists():
        print(f"❌ 源目录不存在: {source_dir}")
        return False

    # 创建目标目录
    dest_dir = Path('results') / f'best_{model}_20'

    # 如果目标目录已存在，先删除
    if dest_dir.exists():
        print(f"⚠️  目标目录已存在，删除旧版本: {dest_dir}")
        shutil.rmtree(dest_dir)

    # 复制整个实验目录
    print(f"\n{'='*80}")
    print(f"{model.upper()} 模型最佳结果复制")
    print(f"{'='*80}")
    print(f"剪枝方法: {config['pruning_method'].upper()}")
    print(f"ACC: {config['metrics']['acc_mean']:.4f}")
    print(f"PPL: {config['metrics']['ppl']:.2f}" if config['metrics']['ppl'] else "PPL: N/A")
    print(f"参数:")
    print(f"  - taylor_seq_len: {config['params']['taylor_seq_len']}")
    print(f"  - taylor_num_samples: {config['params']['taylor_num_samples']}")
    print(f"\n源目录: {source_dir}")
    print(f"目标目录: {dest_dir}")

    # 复制目录
    shutil.copytree(source_dir, dest_dir)

    # 同时复制配置文件到目标目录
    shutil.copy(config_file, dest_dir / 'best_config.json')

    # 统计复制的文件
    total_files = sum(1 for _ in dest_dir.rglob('*') if _.is_file())
    total_size = sum(f.stat().st_size for f in dest_dir.rglob('*') if f.is_file())
    size_mb = total_size / (1024 * 1024)

    print(f"\n✓ 复制完成！")
    print(f"  - 文件数: {total_files}")
    print(f"  - 总大小: {size_mb:.2f} MB")

    # 列出主要文件
    print(f"\n主要文件:")
    important_files = [
        'pruned_model.bin',
        'evaluation/evaluation_results.json',
        'analysis/gradient_diagnosis.json',
        'analysis/pruning_comparison.json',
        'best_config.json'
    ]

    for file_name in important_files:
        file_path = dest_dir / file_name
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {file_name:45s} ({file_size:>8.2f} MB)")
        else:
            print(f"  - {file_name:45s} (不存在)")

    return True


def copy_all_best_results():
    """复制所有模型的最佳结果"""
    models = ['Llama', 'Qwen', 'Mistral']
    success_count = 0

    for model in models:
        if copy_best_result(model):
            success_count += 1
        print()

    print(f"\n{'='*80}")
    print(f"总结: 成功复制 {success_count}/{len(models)} 个模型的最佳结果")
    print(f"{'='*80}")

    # 显示所有最佳结果目录
    print(f"\n最佳结果目录:")
    for model in models:
        dest_dir = Path('results') / f'best_{model}_20'
        if dest_dir.exists():
            print(f"  ✓ {dest_dir}")
        else:
            print(f"  ✗ {dest_dir} (未创建)")


def main():
    parser = argparse.ArgumentParser(description='复制最佳实验结果')
    parser.add_argument('--model', type=str,
                       choices=['Llama', 'Qwen', 'Mistral', 'llama', 'qwen', 'mistral'],
                       help='指定模型')
    parser.add_argument('--all', action='store_true',
                       help='复制所有模型的最佳结果')
    args = parser.parse_args()

    if args.all:
        copy_all_best_results()
    elif args.model:
        # 首字母大写
        model = args.model.capitalize()
        copy_best_result(model)
    else:
        print("请指定 --model <模型名> 或 --all")
        parser.print_help()


if __name__ == '__main__':
    main()
