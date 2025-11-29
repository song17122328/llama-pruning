#!/usr/bin/env python3
"""
统一的数据集管理器
整合所有数据集加载逻辑，避免代码重复
"""

import torch
from typing import Optional, List
from .example_samples import get_examples


class DatasetManager:
    """
    统一管理数据集加载和缓存

    支持的数据集:
    - wikitext2: WikiText-2 数据集
    - ptb: Penn TreeBank 数据集
    - c4: C4 数据集
    """

    def __init__(self, dataset_name: str = 'wikitext2', tokenizer=None):
        """
        初始化数据集管理器

        Args:
            dataset_name: 数据集名称 ('wikitext2', 'ptb', 'c4')
            tokenizer: HuggingFace tokenizer
        """
        self.dataset_name = dataset_name.lower()
        self.tokenizer = tokenizer

        # 验证数据集名称
        supported_datasets = [
            'wikitext', 'wikitext2', 'ptb', 'penn-treebank', 'c4',
            'wikitext_zh', 'wikitext-zh', 'wikipedia_zh', 'wikipedia-zh',
            'c4_zh', 'c4-zh'
        ]
        if self.dataset_name not in supported_datasets:
            raise ValueError(
                f"不支持的数据集: {dataset_name}\n"
                f"支持的数据集:\n"
                f"  英文: wikitext2, ptb, c4\n"
                f"  中文: wikitext_zh, c4_zh"
            )

        # 标准化数据集名称
        if self.dataset_name in ['wikitext', 'wikitext2']:
            self.dataset_name = 'wikitext2'
        elif self.dataset_name in ['ptb', 'penn-treebank']:
            self.dataset_name = 'ptb'
        elif self.dataset_name in ['wikitext_zh', 'wikitext-zh', 'wikipedia_zh', 'wikipedia-zh']:
            self.dataset_name = 'wikitext_zh'
        elif self.dataset_name in ['c4_zh', 'c4-zh']:
            self.dataset_name = 'c4_zh'

        print(f"✓ 数据集管理器初始化: {self.dataset_name}")

    def get_samples(self,
                   num_samples: int,
                   seq_len: int = 128,
                   split: str = 'train',
                   purpose: str = 'general') -> torch.Tensor:
        """
        获取数据样本（tokenized）

        Args:
            num_samples: 样本数量
            seq_len: 序列长度
            split: 数据集分割 ('train', 'test', 'validation')
            purpose: 用途描述（用于日志）

        Returns:
            torch.Tensor: shape [num_samples, seq_len]
        """
        # print(f"\n{'='*60}")
        # print(f"加载数据集: {self.dataset_name}")
        # print(f"  用途: {purpose}")
        # print(f"  样本数: {num_samples}")
        # print(f"  序列长度: {seq_len}")
        # print(f"  数据分割: {split}")
        # print(f"{'='*60}")

        samples = get_examples(
            dataset_name=self.dataset_name,
            tokenizer=self.tokenizer,
            num_samples=num_samples,
            seq_len=seq_len,
            split=split
        )

        return samples

    def get_text_samples(self,
                        num_samples: int,
                        seq_len: int = 512,
                        split: str = 'test') -> List[str]:
        """
        获取文本样本（未tokenized，用于 LayerImportanceAnalyzer）

        Args:
            num_samples: 样本数量
            seq_len: 最大序列长度（用于过滤）
            split: 数据集分割

        Returns:
            List[str]: 文本列表
        """
        # 先获取 tokenized 样本
        tokenized_samples = self.get_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split=split,
            purpose='文本样本提取'
        )

        # 转换回文本
        texts = []
        for i in range(tokenized_samples.size(0)):
            text = self.tokenizer.decode(tokenized_samples[i], skip_special_tokens=True)
            if text.strip():  # 过滤空文本
                texts.append(text)

        return texts

    def get_calibration_data(self,
                            num_samples: int = 128,
                            seq_len: int = 2048) -> torch.Tensor:
        """
        获取校准数据（用于量化等场景）

        Args:
            num_samples: 样本数量
            seq_len: 序列长度

        Returns:
            torch.Tensor: calibration data
        """
        return self.get_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split='train',
            purpose='校准数据'
        )

    def get_ppl_evaluation_data(self,
                               seq_len: int = 128,
                               num_samples: Optional[int] = None) -> torch.Tensor:
        """
        获取 PPL 评估数据

        Args:
            seq_len: 序列长度
            num_samples: 样本数量（None = 使用全部测试集）

        Returns:
            torch.Tensor: evaluation data
        """
        if num_samples is None:
            # 对于 PPL 评估，使用较多样本
            num_samples = 500

        return self.get_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split='test',
            purpose='PPL 评估'
        )

    def get_gradient_samples(self,
                            num_samples: int = 128,
                            seq_len: int = 128) -> torch.Tensor:
        """
        获取梯度计算样本（用于 Taylor importance）

        Args:
            num_samples: 样本数量
            seq_len: 序列长度

        Returns:
            torch.Tensor: gradient computation samples
        """
        return self.get_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split='train',
            purpose='梯度计算'
        )

    def get_activation_samples(self,
                              num_samples: int = 128,
                              seq_len: int = 128) -> torch.Tensor:
        """
        获取激活值收集样本（用于 Wanda importance）

        Args:
            num_samples: 样本数量
            seq_len: 序列长度

        Returns:
            torch.Tensor: activation computation samples
        """
        return self.get_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split='train',
            purpose='激活值收集'
        )

    def get_finetuning_samples(self,
                              num_samples: int = 500,
                              seq_len: int = 512) -> torch.Tensor:
        """
        获取微调样本

        Args:
            num_samples: 样本数量
            seq_len: 序列长度

        Returns:
            torch.Tensor: fine-tuning samples
        """
        return self.get_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split='train',
            purpose='模型微调'
        )

    def get_layer_importance_samples(self,
                                    num_samples: int = 32,
                                    seq_len: int = 128) -> List[str]:
        """
        获取层重要性分析样本（文本格式）

        Args:
            num_samples: 样本数量
            seq_len: 序列长度

        Returns:
            List[str]: 文本样本列表
        """
        return self.get_text_samples(
            num_samples=num_samples,
            seq_len=seq_len,
            split='test'
        )

    def __repr__(self):
        return f"DatasetManager(dataset='{self.dataset_name}')"


# 便捷函数：直接创建 DatasetManager
def create_dataset_manager(dataset_name: str = 'wikitext2',
                          tokenizer=None) -> DatasetManager:
    """
    创建数据集管理器的便捷函数

    Args:
        dataset_name: 数据集名称
        tokenizer: tokenizer

    Returns:
        DatasetManager实例
    """
    return DatasetManager(dataset_name=dataset_name, tokenizer=tokenizer)
