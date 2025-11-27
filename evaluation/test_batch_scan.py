#!/usr/bin/env python3
"""
测试批量扫描功能（不加载模型）
"""

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def detect_model_type(model_dir: Path) -> str:
    """检测模型类型"""
    # SliceGPT: 有 .json 配置文件（但不是 config.json）
    json_files = list(model_dir.glob("*.json"))
    if json_files:
        for json_file in json_files:
            if json_file.name != "config.json":
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if 'embedding_dimensions' in data or 'attention_input_dimensions' in data:
                            return 'slicegpt'
                except:
                    pass

    # HuggingFace: 有 config.json
    if (model_dir / "config.json").exists():
        return 'huggingface'

    # .bin 格式
    if (model_dir / "pruned_model.bin").exists():
        return 'bin'

    # pruned_model 子目录
    if (model_dir / "pruned_model").is_dir():
        return 'huggingface'

    return 'no_model'


def scan_models(models_dir: Path):
    """扫描模型目录"""
    models = {
        'standard': [],
        'slicegpt': [],
        'no_model': []
    }

    print(f"\n{'='*80}")
    print(f"扫描模型目录: {models_dir}")
    print(f"{'='*80}\n")

    for item in sorted(models_dir.iterdir()):
        if not item.is_dir():
            continue

        model_type = detect_model_type(item)

        if model_type == 'slicegpt':
            models['slicegpt'].append(item)
            print(f"✓ SliceGPT 模型: {item.name}")
        elif model_type in ['huggingface', 'bin']:
            models['standard'].append(item)
            print(f"✓ 标准模型: {item.name} (类型: {model_type})")
        else:
            models['no_model'].append(item)
            print(f"⊘ 跳过: {item.name} (无模型文件)")

    print(f"\n汇总:")
    print(f"  - 标准模型: {len(models['standard'])} 个")
    print(f"  - SliceGPT 模型: {len(models['slicegpt'])} 个")
    print(f"  - 无模型文件: {len(models['no_model'])} 个")

    return models


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True)
    args = parser.parse_args()

    models = scan_models(Path(args.models_dir))

    print(f"\n{'='*80}")
    print(f"详细信息:")
    print(f"{'='*80}\n")

    for model_type, model_list in models.items():
        if model_list:
            print(f"\n{model_type.upper()}:")
            for model_dir in model_list:
                print(f"  - {model_dir.name}")
                # 显示目录内容
                files = list(model_dir.iterdir())[:5]
                for f in files:
                    print(f"    * {f.name}")
                if len(list(model_dir.iterdir())) > 5:
                    print(f"    ... (共 {len(list(model_dir.iterdir()))} 个文件/目录)")
