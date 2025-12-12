#!/usr/bin/env python3
"""
从指定目录收集所有评估结果并生成Excel汇总表

用法:
    python collect_evaluation_results.py <目录路径>

示例:
    python collect_evaluation_results.py results/all_model_blockwise_128_128_8_finetuned
    python collect_evaluation_results.py results/grid_search_taylor_32_block_32
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys


def find_evaluation_files(directory):
    """
    递归查找目录下所有的evaluation_results.json文件

    Args:
        directory: 要搜索的目录路径

    Returns:
        list: 所有找到的evaluation_results.json文件路径
    """
    directory = Path(directory)
    return sorted(directory.rglob('evaluation_results.json'))


def extract_model_method(file_path, base_dir):
    """
    从文件路径中提取模型名和方法名

    例如: results/all_model_blockwise_128_128_8_finetuned/Llama/blockwise/evaluation_after_finetune/evaluation_results.json
    提取出: Llama, blockwise

    Args:
        file_path: evaluation_results.json的路径
        base_dir: 基础目录

    Returns:
        tuple: (model_name, method_name)
    """
    rel_path = file_path.relative_to(base_dir)
    parts = rel_path.parts

    # 通常格式是: <model>/<method>/evaluation_*/evaluation_results.json
    if len(parts) >= 2:
        model_name = parts[0]
        method_name = parts[1]
        return model_name, method_name

    return "Unknown", "Unknown"


def parse_evaluation_file(file_path):
    """
    解析evaluation_results.json文件

    Args:
        file_path: JSON文件路径

    Returns:
        dict: 提取的指标数据
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    metrics = data.get('metrics', {})

    # 提取PPL
    ppl = metrics.get('ppl', {})
    wikitext2_ppl = ppl.get('wikitext2 (wikitext-2-raw-v1)', None)
    ptb_ppl = ppl.get('ptb', None)

    # 提取zeroshot任务
    zeroshot = metrics.get('zeroshot', {})

    result = {
        'wikitext2_ppl': wikitext2_ppl,
        'ptb_ppl': ptb_ppl,
        'avg_zeroshot_acc': metrics.get('avg_zeroshot_acc', None)
    }

    # 提取每个zeroshot任务的准确率
    zeroshot_tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande',
                      'arc_easy', 'arc_challenge', 'openbookqa']

    for task in zeroshot_tasks:
        if task in zeroshot:
            result[task] = zeroshot[task].get('accuracy', None)
        else:
            result[task] = None

    return result


def collect_results(directory):
    """
    收集目录下所有评估结果

    Args:
        directory: 要搜索的目录

    Returns:
        pd.DataFrame: 包含所有结果的DataFrame
    """
    directory = Path(directory)

    if not directory.exists():
        raise ValueError(f"目录不存在: {directory}")

    print(f"搜索目录: {directory}")

    # 查找所有evaluation_results.json文件
    eval_files = find_evaluation_files(directory)

    if not eval_files:
        raise ValueError(f"在 {directory} 中未找到任何 evaluation_results.json 文件")

    print(f"找到 {len(eval_files)} 个评估结果文件")

    # 收集数据
    all_results = []

    for file_path in eval_files:
        try:
            # 提取模型和方法
            model, method = extract_model_method(file_path, directory)

            # 解析评估结果
            metrics = parse_evaluation_file(file_path)

            # 合并数据
            row = {
                'Model': model,
                'Method': method,
                'File': str(file_path.relative_to(directory)),
                **metrics
            }

            all_results.append(row)
            print(f"  ✓ {model}/{method}")

        except Exception as e:
            print(f"  ✗ 解析失败 {file_path}: {e}")
            continue

    if not all_results:
        raise ValueError("没有成功解析任何评估结果")

    # 创建DataFrame
    df = pd.DataFrame(all_results)

    # 重新排列列的顺序
    base_cols = ['Model', 'Method', 'File']
    ppl_cols = ['wikitext2_ppl', 'ptb_ppl']
    zeroshot_cols = ['boolq', 'piqa', 'hellaswag', 'winogrande',
                     'arc_easy', 'arc_challenge', 'openbookqa', 'avg_zeroshot_acc']

    ordered_cols = base_cols + ppl_cols + zeroshot_cols

    # 只保留存在的列
    ordered_cols = [col for col in ordered_cols if col in df.columns]
    df = df[ordered_cols]

    return df


def save_to_excel(df, output_path):
    """
    保存DataFrame到Excel文件

    Args:
        df: DataFrame
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Evaluation Results', index=False)

        # 获取worksheet以进行格式化
        worksheet = writer.sheets['Evaluation Results']

        # 调整列宽
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"\n✓ Excel文件已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='从指定目录收集评估结果并生成Excel汇总表',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python collect_evaluation_results.py results/all_model_blockwise_128_128_8_finetuned
  python collect_evaluation_results.py results/grid_search_taylor_32_block_32
  python collect_evaluation_results.py results/all_model_blockwise_128_128_8_finetuned -o custom_name.xlsx
        """
    )

    parser.add_argument('directory', type=str,
                       help='包含评估结果的目录路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出Excel文件名（默认为：evaluation_summary.xlsx）')

    args = parser.parse_args()

    try:
        # 收集结果
        df = collect_results(args.directory)

        # 确定输出路径
        if args.output:
            output_path = Path(args.directory) / args.output
        else:
            output_path = Path(args.directory) / 'evaluation_summary.xlsx'

        # 保存到Excel
        save_to_excel(df, output_path)

        # 打印摘要
        print(f"\n{'='*80}")
        print(f"摘要:")
        print(f"  - 总模型数: {df['Model'].nunique()}")
        print(f"  - 总方法数: {df['Method'].nunique()}")
        print(f"  - 总记录数: {len(df)}")
        print(f"{'='*80}")

        # 打印表格预览
        print("\n表格预览:")
        print(df.to_string(index=False, max_rows=10))

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
