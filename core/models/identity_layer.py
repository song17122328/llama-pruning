
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

    def __init__(self, original_layer=None, config=None, layer_idx=None):
        """
        初始化 Identity 层

        Args:
            original_layer: 原始的 decoder layer（可选，用于复制必要属性）
            config: 模型配置（可选，用于 Qwen2 等模型）
            layer_idx: 层索引（可选，用于 Qwen2 等模型）
        """
        super().__init__()

        # Qwen2 需要 attention_type 属性
        if config is not None and hasattr(config, 'model_type'):
            model_type = config.model_type
            if model_type in ['qwen2', 'qwen'] and hasattr(config, 'layer_types') and layer_idx is not None:
                self.attention_type = config.layer_types[layer_idx]
            elif original_layer is not None and hasattr(original_layer, 'attention_type'):
                self.attention_type = original_layer.attention_type

    def forward(self, hidden_states, *args, **kwargs):
        """
        前向传播：直接返回输入

        Args:
            hidden_states: 输入的 hidden states [batch_size, seq_len, hidden_dim]
            *args: 其他位置参数（被忽略）
            **kwargs: 其他关键字参数（被忽略）

        Returns:
            hidden_states tensor（始终返回张量，不返回元组）

        Note:
            在 HuggingFace Transformers 中，decoder layer 总是直接返回 hidden_states 张量。
            即使 use_cache=True 或 output_attentions=True，cache 和 attention 也是通过
            model.forward 的返回值传递，而不是通过 decoder_layer.forward 返回。
        """
        # 始终只返回 hidden_states 张量
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

    def __init__(self, model_type='llama'):
        """
        初始化 Zero Attention（无参数）

        Args:
            model_type: 模型类型 ('llama', 'mistral', 'qwen', 'qwen2' 等)
                      用于确定返回值格式
        """
        super().__init__()
        self.model_type = model_type.lower()

    def forward(self, hidden_states, *args, **kwargs):
        """
        前向传播：返回全零tensor

        Args:
            hidden_states: 输入 [batch_size, seq_len, hidden_dim]

        Returns:
            根据模型类型返回不同格式：
            - Mistral/Qwen/Qwen2: 总是返回 (output, None) - 2个值
            - LLaMA: use_cache=True 时返回 (output, None, None) - 3个值
        """
        # 返回全零（在残差连接中等效于跳过）
        output = torch.zeros_like(hidden_states)

        # Mistral 和 Qwen 系列模型：总是返回2个值
        if self.model_type in ['mistral', 'qwen', 'qwen2']:
            # Attention 总是返回 (attn_output, attn_weights)
            return output, None
        else:
            # LLaMA 等模型：根据 use_cache 决定返回格式
            use_cache = kwargs.get('use_cache', False)

            if use_cache:
                # 返回 (output, None, None) - (attn_output, attn_weights, past_key_value)
                return output, None, None
            else:
                # 返回 (output, None) - (attn_output, attn_weights)
                return output, None

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
