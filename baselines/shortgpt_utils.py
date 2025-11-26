#!/usr/bin/env python3
"""
ShortGPT 工具函数
基于 Block Influence (BI) 的层重要性计算
"""

import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular: bool = False
) -> torch.Tensor:
    """
    计算 Block Influence (BI) - 层输入和输出的相似度

    BI 的核心思想：
    - 如果层的输入和输出非常相似，说明该层的变换作用很小
    - 相似度高 → 重要性低 → 可以被剪枝

    Args:
        input_hidden_state: 层的输入隐藏状态 [batch, seq_len, hidden_dim]
        output_hidden_state: 层的输出隐藏状态 [batch, seq_len, hidden_dim]
        angular: 是否使用角度距离（默认使用余弦距离）

    Returns:
        影响分数 [batch * seq_len]，值越大表示影响越大（重要性越高）
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    # 计算归一化
    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    # 计算余弦相似度
    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output + 1e-8)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        # 角度距离：arccos(sim) / π
        return (torch.arccos(sim.clamp(-1, 1)) / torch.pi)
    else:
        # 余弦距离：1 - sim
        # 相似度越高，距离越小，重要性越低
        return 1 - sim


def compute_layer_importances_bi(
    model,
    tokenizer,
    texts: List[str],
    stride: int = 256,
    device: str = 'cuda',
    verbose: bool = True
) -> np.ndarray:
    """
    使用 Block Influence 计算每层的重要性

    Args:
        model: LLaMA 模型
        tokenizer: tokenizer
        texts: 文本列表
        stride: 滑动窗口步长
        device: 设备
        verbose: 是否显示进度

    Returns:
        importances: 每层的重要性分数 [num_layers]
    """
    num_layers = len(model.model.layers)
    importances = np.zeros(num_layers)

    model.eval()

    if verbose:
        print(f"\n计算 Block Influence (BI) 重要性...")
        print(f"  文本数: {len(texts)}")
        print(f"  Stride: {stride}")

    pbar = tqdm(texts, desc="计算 BI", disable=not verbose)

    for text in pbar:
        # Tokenize
        tokens = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=4096  # 限制最大长度
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        seq_len = input_ids.shape[1]

        # 滑动窗口处理长文本
        for start in range(0, seq_len, stride):
            end = min(start + stride, seq_len)

            if end - start < 10:  # 跳过太短的片段
                continue

            inputs = input_ids[:, start:end]
            attn = attention_mask[:, start:end]

            with torch.no_grad():
                outputs = model(
                    input_ids=inputs,
                    attention_mask=attn,
                    output_hidden_states=True
                )

            # 获取每层的隐藏状态
            hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim]

            # 计算每层的 BI
            # hidden_states[0] 是 embedding 输出
            # hidden_states[1] 是第一层输出，以此类推
            for layer_idx in range(num_layers):
                input_hidden = hidden_states[layer_idx]      # 层输入
                output_hidden = hidden_states[layer_idx + 1]  # 层输出

                # 计算 BI（相似度距离）
                bi = block_influence(input_hidden, output_hidden)

                # 累加到该层的重要性分数
                # BI 越大 → 输入输出差异越大 → 该层作用越大 → 重要性越高
                importances[layer_idx] += bi.sum().cpu().item()

    if verbose:
        print(f"✓ BI 重要性计算完成")

    return importances


def select_layers_to_remove(
    importances: np.ndarray,
    n_remove: int,
    verbose: bool = True
) -> List[int]:
    """
    根据重要性分数选择要移除的层

    Args:
        importances: 每层的重要性分数
        n_remove: 要移除的层数
        verbose: 是否打印信息

    Returns:
        layers_to_remove: 要移除的层索引列表
    """
    # 按重要性排序，选择最不重要的 n 层
    layers_to_remove = np.argsort(importances)[:n_remove].tolist()

    if verbose:
        print(f"\n{'='*60}")
        print(f"ShortGPT 层选择结果")
        print(f"{'='*60}")
        print(f"总层数: {len(importances)}")
        print(f"移除层数: {n_remove}")
        print(f"保留层数: {len(importances) - n_remove}")
        print(f"\n要移除的层（按重要性从低到高）:")

        for i, layer_idx in enumerate(sorted(layers_to_remove)):
            imp = importances[layer_idx]
            print(f"  {i+1}. Layer {layer_idx:2d}  (重要性: {imp:.4f})")

        print(f"\n剪枝顺序: {', '.join(str(i) for i in layers_to_remove)}")
        print(f"{'='*60}")

    return layers_to_remove


def remove_layers_from_model(
    model,
    layers_to_remove: List[int],
    verbose: bool = True
) -> Tuple[int, int]:
    """
    从模型中物理移除指定的层

    Args:
        model: LLaMA 模型
        layers_to_remove: 要移除的层索引列表
        verbose: 是否打印信息

    Returns:
        (original_num_layers, new_num_layers)
    """
    original_num_layers = len(model.model.layers)

    if verbose:
        print(f"\n移除层...")
        print(f"  原始层数: {original_num_layers}")
        print(f"  要移除: {len(layers_to_remove)} 层")

    # 从后往前删除，避免索引变化
    for layer_idx in sorted(layers_to_remove, reverse=True):
        del model.model.layers[layer_idx]
        if verbose:
            print(f"  ✓ 移除 Layer {layer_idx}")

    # 更新 layer_idx（重要：KV cache 需要正确的 layer_idx）
    for layer_idx, layer_module in enumerate(model.model.layers):
        if hasattr(layer_module.self_attn, 'layer_idx'):
            layer_module.self_attn.layer_idx = layer_idx

    new_num_layers = len(model.model.layers)

    if verbose:
        print(f"  新层数: {new_num_layers}")
        print(f"✓ 层移除完成")

    return original_num_layers, new_num_layers
