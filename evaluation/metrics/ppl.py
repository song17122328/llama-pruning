#!/usr/bin/env python3
"""
困惑度(Perplexity)评估工具
用于评估语言模型在各种数据集上的困惑度
"""

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Optional


class PPLMetric:
    """
    困惑度评估类

    用法:
        ppl_metric = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'])
        results = ppl_metric  # 自动计算并返回结果字典
    """

    def __init__(
        self,
        model,
        tokenizer,
        datasets: List[str],
        seq_len: int = 128,
        device: str = 'cuda',
        stride: int = None,
        batch_size: int = 1
    ):
        """
        初始化PPL评估器

        Args:
            model: 语言模型
            tokenizer: tokenizer实例
            datasets: 要评估的数据集列表，如 ['wikitext2', 'ptb', 'c4']
            seq_len: 序列长度
            device: 计算设备
            stride: 滑动窗口步长（None则等于seq_len，即不重叠）
            batch_size: 批处理大小
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_names = datasets
        self.seq_len = seq_len
        self.device = device
        self.stride = stride if stride is not None else seq_len
        self.batch_size = batch_size

        # 确保 tokenizer 有 pad_token (Llama 等模型默认没有)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 确保模型在正确的设备上
        if hasattr(model, 'to'):
            self.model.to(device)

        self.model.eval()

        # 自动计算PPL
        self.results = self._evaluate_all()

    def _evaluate_all(self) -> Dict[str, float]:
        """评估所有数据集"""
        results = {}

        for dataset_name in self.dataset_names:
            try:
                ppl = self._evaluate_dataset(dataset_name)
                # 使用与数据集加载一致的键名格式
                if dataset_name.lower() in ['wikitext', 'wikitext2', 'wikitext-2']:
                    key = 'wikitext2 (wikitext-2-raw-v1)'
                elif dataset_name.lower() in ['wikitext103', 'wikitext-103']:
                    key = 'wikitext103 (wikitext-103-raw-v1)'
                elif dataset_name.lower() in ['ptb', 'penn-treebank']:
                    key = 'ptb'
                elif dataset_name.lower() == 'c4':
                    key = 'c4'
                else:
                    key = dataset_name

                results[key] = ppl
                print(f"✓ {key}: PPL = {ppl:.2f}")

            except Exception as e:
                print(f"✗ 评估 {dataset_name} 时出错: {e}")
                results[dataset_name] = float('inf')

        return results

    def _evaluate_dataset(self, dataset_name: str) -> float:
        """
        评估单个数据集的困惑度

        Args:
            dataset_name: 数据集名称

        Returns:
            float: 困惑度值
        """
        # 加载数据集
        encodings = self._load_dataset(dataset_name)

        # 计算PPL
        ppl = self._calculate_perplexity(encodings)

        return ppl

    def _load_dataset(self, dataset_name: str) -> torch.Tensor:
        """
        加载数据集并tokenize

        支持从本地或HuggingFace Hub加载多种数据集

        Args:
            dataset_name: 数据集名称

        Returns:
            torch.Tensor: tokenized数据
        """
        dataset_name_lower = dataset_name.lower()

        print(f"加载数据集: {dataset_name}")

        # WikiText2
        if dataset_name_lower in ['wikitext', 'wikitext2', 'wikitext-2']:
            import os
            from datasets import load_from_disk

            # 优先从项目 data/ 目录加载
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            local_path = os.path.join(project_root, "data", "wikitext2")

            if os.path.exists(local_path):
                print(f"  从本地加载: {local_path}")
                try:
                    dataset = load_from_disk(local_path)
                    dataset = dataset['test']
                    text_field = 'text'
                except Exception as e:
                    print(f"  本地加载失败: {e}，尝试在线下载...")
                    dataset = None
            else:
                dataset = None

            # 如果本地没有，尝试在线加载
            if dataset is None:
                print("  本地数据不存在，请先运行: python evaluation/download_datasets.py")
                try:
                    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                    text_field = 'text'
                except Exception as e:
                    raise ValueError(f"无法加载 WikiText2: {e}")

        # PTB (Penn TreeBank)
        elif dataset_name_lower in ['ptb', 'penn-treebank', 'penn_treebank']:
            import os
            from datasets import load_from_disk

            # 优先从项目 data/ 目录加载
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            local_path = os.path.join(project_root, "data", "ptb")

            if os.path.exists(local_path):
                print(f"  从本地加载: {local_path}")
                try:
                    dataset = load_from_disk(local_path)
                    dataset = dataset['test']
                    text_field = 'sentence'
                except Exception as e:
                    print(f"  本地加载失败: {e}，尝试在线下载...")
                    dataset = None
            else:
                dataset = None

            # 如果本地没有，尝试在线加载
            if dataset is None:
                print("  本地数据不存在，请先运行: python evaluation/download_datasets.py --datasets ptb")
                try:
                    dataset = load_dataset('ptb_text_only', 'penn_treebank', split='test')
                    text_field = 'sentence' if 'sentence' in dataset.column_names else 'text'
                except Exception as e:
                    raise ValueError(f"无法加载 PTB: {e}")

        # C4
        elif dataset_name_lower in ['c4']:
            import os
            from datasets import load_from_disk

            # 优先从项目 data/ 目录加载
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            local_path = os.path.join(project_root, "data", "c4")

            if os.path.exists(local_path):
                print(f"  从本地加载: {local_path}")
                try:
                    dataset = load_from_disk(local_path)
                    text_field = 'text'
                except Exception as e:
                    print(f"  本地加载失败: {e}，尝试在线下载...")
                    dataset = None
            else:
                dataset = None

            # 如果本地没有，尝试在线加载
            if dataset is None:
                print("  本地数据不存在，请先运行: python evaluation/download_datasets.py --datasets c4")
                try:
                    dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=False, trust_remote_code=True)
                    dataset = dataset.select(range(min(10000, len(dataset))))
                    text_field = 'text'
                except Exception as e:
                    raise ValueError(f"无法加载 C4: {e}")

        else:
            raise ValueError(
                f"不支持的数据集: {dataset_name}\n"
                f"当前支持: wikitext2, ptb, c4"
            )

        # 合并所有文本
        if isinstance(dataset, list):
            texts = [item[text_field] for item in dataset]
        else:
            texts = [item[text_field] for item in dataset]

        # 过滤空文本并合并
        text = '\n\n'.join([t for t in texts if t.strip()])

        print(f"  数据集加载完成，总字符数: {len(text):,}")

        # Tokenize整个文本
        encodings = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=False
        )

        return encodings['input_ids'].squeeze(0)

    def _calculate_perplexity(self, encodings: torch.Tensor) -> float:
        """
        使用滑动窗口计算困惑度

        Args:
            encodings: tokenized input_ids

        Returns:
            float: 困惑度
        """
        seq_len = self.seq_len
        stride = self.stride

        nlls = []  # negative log-likelihoods
        prev_end_loc = 0

        # 滑动窗口遍历
        for begin_loc in tqdm(range(0, encodings.size(0), stride), desc="计算PPL"):
            end_loc = min(begin_loc + seq_len, encodings.size(0))
            trg_len = end_loc - prev_end_loc  # 可能小于stride的最后一个序列

            input_ids = encodings[begin_loc:end_loc].unsqueeze(0).to(self.device)

            # 如果序列太短，跳过
            if input_ids.size(1) < 2:
                break

            target_ids = input_ids.clone()

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                # outputs.loss 已经是平均loss
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == encodings.size(0):
                break

        # 计算困惑度
        ppl = torch.exp(torch.stack(nlls).mean())

        return ppl.item()

    def get(self, key: str, default=None):
        """字典式访问接口"""
        return self.results.get(key, default)

    def __contains__(self, key: str):
        """支持 'key in ppl' 操作"""
        return key in self.results

    def __getitem__(self, key: str):
        """支持 ppl['wikitext2'] 访问"""
        return self.results[key]

    def __repr__(self):
        """打印结果"""
        return str(self.results)

    def __str__(self):
        """转为字符串"""
        lines = []
        for dataset, ppl_value in self.results.items():
            lines.append(f"  {dataset}: {ppl_value:.2f}")
        return "\n".join(lines)


def evaluate_perplexity(
    model,
    tokenizer,
    dataset_name: str = 'wikitext2',
    seq_len: int = 128,
    device: str = 'cuda'
) -> float:
    """
    快捷函数：评估单个数据集的困惑度

    Args:
        model: 语言模型
        tokenizer: tokenizer
        dataset_name: 数据集名称
        seq_len: 序列长度
        device: 设备

    Returns:
        float: 困惑度
    """
    metric = PPLMetric(model, tokenizer, [dataset_name], seq_len, device)
    return list(metric.results.values())[0]


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("测试 PPLMetric 类...")

    # 使用一个小模型进行测试
    print(f"\n加载测试模型 (Llama-3-8B-Instruct)...")
    # model_name = "/newdata/LLMs/Llama-3-8B-Instruct"
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)


    best_checkpoint_path = "/data/home/yuanxiaosong/GAQ-Aware-Prune/prune_log/ppl_search_20251118_005448_ratio_1.0_9.0_freeze_0/pytorch_model.bin"
    checkpoint = torch.load(best_checkpoint_path, weights_only=False)
    model = checkpoint['model']
    tokenizer = checkpoint['tokenizer']
    print("✅ 从检查点重新加载成功")

    print("\n计算 wikitext2 PPL...,默认使用的seq_len是128")
    try:
        ppl_metric = PPLMetric(
            model,
            tokenizer,
            datasets=['wikitext2'],
            seq_len=128,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print(f"\n结果:")
        print(ppl_metric)

        # 测试字典访问
        print(f"\n字典访问测试:")
        for key in ppl_metric.results:
            print(f"  ppl['{key}'] = {ppl_metric[key]:.2f}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n测试完成！")
