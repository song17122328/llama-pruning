#!/usr/bin/env python3
"""
样本数据加载工具
用于从各种数据集中加载示例样本，供模型剪枝时计算梯度使用
"""

import torch
from datasets import load_dataset,load_from_disk
from typing import Optional


def get_examples(
    dataset_name: str,
    tokenizer,
    num_samples: int = 10,
    seq_len: int = 128,
    split: str = 'train'
) -> torch.Tensor:
    """
    从指定数据集加载样本数据并进行tokenization

    Args:
        dataset_name: 数据集名称，支持 'wikitext', 'c4', 'ptb', 'wikitext_zh', 'c4_zh' 等
        tokenizer: HuggingFace tokenizer实例
        num_samples: 需要的样本数量
        seq_len: 序列长度
        split: 数据集划分 ('train', 'test', 'validation')

    Returns:
        torch.Tensor: tokenized input_ids, shape [num_samples, seq_len]
    """

    # 支持 wikitext2 和 c4 数据集（英文）
    if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
        try:
            dataset = load_from_disk("/newdata/DataSets/wikitext2")[split]
            # print("✅ 本地加载成功！")
        except Exception as e:
            print(f"⚠️ 本地加载失败， (文件可能损坏或格式不匹配): {e}")
            print("从网上获取")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text_field = 'text'

    elif dataset_name.lower() in ['c4']:
        try:
            print("尝试从本地加载 C4 数据集: /newdata/DataSets/c4/")
            dataset = load_from_disk("/newdata/DataSets/c4")
            # C4 数据集可能已经包含 train/validation split，或者是完整的数据集
            if hasattr(dataset, split):
                dataset = dataset[split]
            print(f"✓ 成功从本地加载 C4 数据集 ({len(dataset)} 样本)")
        except Exception as e:
            print(f"⚠️ C4 本地加载失败: {e}")
            print("尝试从 HuggingFace 下载...")
            dataset = load_dataset("c4", "en", split=split, streaming=False)
        text_field = 'text'

    # 中文数据集支持
    elif dataset_name.lower() in ['wikitext_zh', 'wikitext-zh', 'wikipedia_zh', 'wikipedia-zh']:
        try:
            # print("尝试从本地加载中文维基百科数据集: /newdata/DataSets/wikipedia_zh/")
            dataset = load_from_disk("/newdata/DataSets/wikipedia_zh")
            if hasattr(dataset, split):
                dataset = dataset[split]
            elif split == 'train':
                # 中文维基可能只有一个split，使用前80%作为train
                total_len = len(dataset)
                dataset = dataset.select(range(int(total_len * 0.8)))
            else:
                # test split 使用后20%
                total_len = len(dataset)
                dataset = dataset.select(range(int(total_len * 0.8), total_len))
            # print(f"✓ 成功从本地加载中文维基百科数据集 ({len(dataset)} 样本)")
        except Exception as e:
            print(f"⚠️ 中文维基本地加载失败: {e}")
            print("尝试从 HuggingFace 下载 wikipedia zh...")
            try:
                dataset = load_dataset("wikipedia", "20220301.zh", split=split)
            except:
                print("⚠️ HuggingFace wikipedia zh 加载失败，尝试备选数据集...")
                # 备选：使用 CLUECorpus2020
                dataset = load_dataset("clue", "cluecorpussmall", split=split)
        text_field = 'text'

    elif dataset_name.lower() in ['c4_zh', 'c4-zh']:
        try:
            print("尝试从本地加载 C4 中文数据集: /newdata/DataSets/c4_zh/")
            dataset = load_from_disk("/newdata/DataSets/c4_zh")
            if hasattr(dataset, split):
                dataset = dataset[split]
            print(f"✓ 成功从本地加载 C4 中文数据集 ({len(dataset)} 样本)")
        except Exception as e:
            print(f"⚠️ C4 中文本地加载失败: {e}")
            print("尝试从 HuggingFace 下载 C4 中文版本...")
            try:
                dataset = load_dataset("c4", "zh", split=split, streaming=False)
            except:
                print("⚠️ C4 中文加载失败，回退到中文维基百科...")
                dataset = load_dataset("wikipedia", "20220301.zh", split=split)
        text_field = 'text'

    else:
        raise ValueError(
            f"不支持的数据集: {dataset_name}\n"
            f"支持的数据集:\n"
            f"  英文: wikitext2, c4\n"
            f"  中文: wikitext_zh, c4_zh"
        )

    # 收集文本样本
    texts = []
    for item in dataset:
        if 'text' in item:
            content = item['text']
        elif 'completion' in item:
            content = item['completion']
        elif 'content' in item:
            content = item['content']
        else:
            # 兜底：获取第一个字符串类型的列
            content = list(item.values())[0]
        text = content.strip()
        # 过滤掉太短的文本
        if len(text) > 50:
            texts.append(text)

        if len(texts) >= num_samples:
            break

    # 如果样本不够，重复使用
    while len(texts) < num_samples:
        texts.extend(texts[:num_samples - len(texts)])

    texts = texts[:num_samples]

    # 确保 tokenizer 有 pad_token (Llama 等模型默认没有)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenization
    encodings = tokenizer(
        texts,
        return_tensors='pt',
        max_length=seq_len,
        truncation=True,
        padding='max_length'
    )

    return encodings['input_ids']


def get_examples_from_text(
    texts: list,
    tokenizer,
    seq_len: int = 128
) -> torch.Tensor:
    """
    从给定的文本列表创建样本

    Args:
        texts: 文本列表
        tokenizer: HuggingFace tokenizer实例
        seq_len: 序列长度

    Returns:
        torch.Tensor: tokenized input_ids
    """
    # 确保 tokenizer 有 pad_token (Llama 等模型默认没有)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encodings = tokenizer(
        texts,
        return_tensors='pt',
        max_length=seq_len,
        truncation=True,
        padding='max_length'
    )

    return encodings['input_ids']


def get_calibration_data(
    dataset_name: str = 'wikitext',
    tokenizer = None,
    num_samples: int = 128,
    seq_len: int = 2048
) -> torch.Tensor:
    """
    获取校准数据（用于量化等场景）

    Args:
        dataset_name: 数据集名称
        tokenizer: tokenizer实例
        num_samples: 样本数量
        seq_len: 序列长度

    Returns:
        torch.Tensor: calibration data
    """
    return get_examples(dataset_name, tokenizer, num_samples, seq_len)


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer

    print("测试 get_examples 函数...")

    # 使用一个小模型的tokenizer进行测试
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 测试加载wikitext
    print("\n1. 测试加载 wikitext 数据集:")
    try:
        examples = get_examples('wikitext', tokenizer, num_samples=5, seq_len=64)
        print(f"   ✓ 成功加载，shape: {examples.shape}")
        print(f"   样本示例: {tokenizer.decode(examples[0][:20])}")
    except Exception as e:
        print(f"   ✗ 失败: {e}")

    print("\n测试完成！")
