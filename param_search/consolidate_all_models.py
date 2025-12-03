#!/usr/bin/env python3
"""
批量汇总所有模型的剪枝方法结果

用法:
    python param_search/consolidate_all_models.py
"""

import subprocess
import sys
from pathlib import Path


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
    print("批量汇总所有模型的剪枝方法结果")
    print("="*80 + "\n")

    failed_models = []

    for model in models:
        print(f"\n{'='*80}")
        print(f"处理模型: {model}")
        print(f"{'='*80}")

        try:
            # 运行 consolidate_model_results.py
            result = subprocess.run(
                ['python', 'param_search/consolidate_model_results.py', '--model', model],
                check=True,
                capture_output=False
            )
            print(f"✓ {model} 汇总完成")
        except subprocess.CalledProcessError as e:
            print(f"✗ {model} 汇总失败")
            failed_models.append(model)
        except FileNotFoundError as e:
            print(f"✗ {model} 未找到结果文件")
            failed_models.append(model)

    print("\n" + "="*80)
    print("批量汇总结果")
    print("="*80)
    print(f"成功: {len(models) - len(failed_models)}/{len(models)}")
    if failed_models:
        print(f"失败的模型: {', '.join(failed_models)}")
    else:
        print("✓ 所有模型汇总成功！")


if __name__ == '__main__':
    main()
