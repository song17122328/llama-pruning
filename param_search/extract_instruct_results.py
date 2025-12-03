#!/usr/bin/env python3
"""批量提取Instruct模型的结果"""

import subprocess
from pathlib import Path

models = ['Llama-Instruct', 'Qwen-Instruct', 'Mistral-Instruct']
methods = ['', '_layerwise', '_blockwise']

for model in models:
    for method in methods:
        search_dir = Path('results') / f'search_{model}{method}_20'
        if search_dir.exists():
            print(f"\n{'='*80}")
            print(f"提取 {search_dir.name} ...")
            print(f"{'='*80}")
            try:
                subprocess.run(['python', 'param_search/re_extract_results.py',
                              '--search_dir', str(search_dir)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"✗ 提取失败: {e}")
