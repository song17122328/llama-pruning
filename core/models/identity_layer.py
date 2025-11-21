#!/usr/bin/env python3
"""
Identity Layer - 用于替换被剪空的层

当使用 H-GSP (Hybrid Global Structural Pruning) 进行剪枝时，
某些层可能被完全剪空。为了保持模型层数结构完整（避免 HuggingFace Transformers
内部假设被打破），我们将这些层替换为简单的 pass-through 层。
"""

import torch
import torch.nn as nn


class IdentityDecoderLayer(nn.Module):
    """
    Identity 层：直接传递输入，不做任何计算

    这个层相当于一个跳过连接，输入什么就输出什么。
    在 Transformer 架构中，由于有残差连接，这等效于完全移除该层。

    用途：
    - 替换被完全剪空的 Transformer 层
    - 保持模型的层数结构不变
    - 避免 HuggingFace 内部对层数的各种假设被打破

    示例：
        >>> layer = IdentityDecoderLayer()
        >>> hidden_states = torch.randn(1, 10, 768)
        >>> output = layer(hidden_states)
        >>> assert torch.equal(output, hidden_states)
    """

    def __init__(self):
        """初始化 Identity 层（无参数）"""
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        """
        前向传播：直接返回输入

        Args:
            hidden_states: 输入的 hidden states [batch_size, seq_len, hidden_dim]
            *args: 其他位置参数（被忽略）
            **kwargs: 其他关键字参数（用于判断返回格式）

        Returns:
            根据 kwargs 返回不同格式：
            - 如果 output_attentions 或 use_cache 为 True：返回元组
            - 否则：直接返回 hidden_states
        """
        # 检查返回格式要求（兼容 HuggingFace Transformers）
        output_attentions = kwargs.get('output_attentions', False)
        use_cache = kwargs.get('use_cache', False)

        if output_attentions or use_cache:
            # 返回元组格式 (hidden_states, attention_weights, past_key_value)
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (None,)  # attention_weights = None
            if use_cache:
                outputs += (None,)  # past_key_value = None
            return outputs
        else:
            # 只返回 hidden_states
            return hidden_states

    def __repr__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}()"


class ZeroAttention(nn.Module):
    """
    Zero Attention：输出全零的 Attention 模块

    用于替换被完全剪空的 Attention block。
    在残差连接中：hidden = hidden + 0 = hidden（相当于跳过 Attention）

    这比保留1个head更优：
    - 真正删除所有参数
    - 不会因为维度0导致forward崩溃
    - 利用残差连接的特性
    """

    def __init__(self):
        """初始化 Zero Attention（无参数）"""
        super().__init__()

    def forward(self, hidden_states, *args, **kwargs):
        """
        前向传播：返回全零tensor

        Args:
            hidden_states: 输入 [batch_size, seq_len, hidden_dim]

        Returns:
            全零tensor，shape与输入相同
        """
        # 返回全零（在残差连接中等效于跳过）
        output = torch.zeros_like(hidden_states)

        # 兼容 HuggingFace 接口
        output_attentions = kwargs.get('output_attentions', False)
        use_cache = kwargs.get('use_cache', False)

        if output_attentions or use_cache:
            outputs = (output,)
            if output_attentions:
                outputs += (None,)  # attention_weights = None
            if use_cache:
                outputs += (None,)  # past_key_value = None
            return outputs
        else:
            return output

    def __repr__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}()"


class ZeroMLP(nn.Module):
    """
    Zero MLP：输出全零的 MLP 模块

    用于替换被完全剪空的 MLP block。
    在残差连接中：hidden = hidden + 0 = hidden（相当于跳过 MLP）
    """

    def __init__(self):
        """初始化 Zero MLP（无参数）"""
        super().__init__()

    def forward(self, hidden_states):
        """
        前向传播：返回全零tensor

        Args:
            hidden_states: 输入 [batch_size, seq_len, hidden_dim]

        Returns:
            全零tensor，shape与输入相同
        """
        # 返回全零（在残差连接中等效于跳过）
        return torch.zeros_like(hidden_states)

    def __repr__(self):
        """字符串表示"""
        return f"{self.__class__.__name__}()"
