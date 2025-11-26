#!/usr/bin/env python3
"""
SlimGPT 工具函数
基于 Hessian 的最优脑损伤（Optimal Brain Surgeon）剪枝方法

论文：SlimGPT: Layer-wise Structured Pruning for Large Language Models
核心思想：通过 Hessian 矩阵计算剪枝误差，选择最小误差的神经元剪枝，并补偿其他权重
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from tqdm import tqdm


def collect_layer_inputs(
    model,
    dataloader,
    layer_idx: int,
    device: str = 'cuda',
    max_samples: int = 128
) -> torch.Tensor:
    """
    收集指定层的输入特征（用于计算 Hessian）

    Args:
        model: LLaMA 模型
        dataloader: 数据加载器
        layer_idx: 层索引
        device: 设备
        max_samples: 最大样本数（限制内存）

    Returns:
        X: 层输入特征 [total_tokens, hidden_dim]
    """
    inputs_list = []
    total_tokens = 0

    def hook_fn(module, input, output):
        # input[0] shape: [batch, seq_len, hidden_dim]
        inp = input[0].detach()
        # 展平为 [batch * seq_len, hidden_dim]
        inp = inp.reshape(-1, inp.shape[-1])
        inputs_list.append(inp.cpu())

    # 注册 hook
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook_fn)

    model.eval()
    try:
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(device)
                else:
                    input_ids = batch.to(device)

                model(input_ids)

                # 限制样本数（避免内存爆炸）
                total_tokens += input_ids.numel()
                if total_tokens > max_samples * 512:  # 约 max_samples 个序列
                    break
    finally:
        handle.remove()

    # 拼接所有输入
    X = torch.cat(inputs_list, dim=0)

    # 如果样本太多，随机采样
    if X.shape[0] > max_samples * 512:
        indices = torch.randperm(X.shape[0])[:max_samples * 512]
        X = X[indices]

    return X.to(device)


def compute_hessian_inv(
    X: torch.Tensor,
    damping: float = 1e-6
) -> torch.Tensor:
    """
    计算 Hessian 矩阵的逆（使用 X.T @ X 近似）

    H = 2 * X.T @ X / n_samples

    Args:
        X: 输入特征 [n_samples, hidden_dim]
        damping: 阻尼系数（避免数值不稳定）

    Returns:
        H_inv: Hessian 逆矩阵 [hidden_dim, hidden_dim]
    """
    n_samples = X.shape[0]
    hidden_dim = X.shape[1]

    # 计算 Hessian: H = 2 * X.T @ X / n
    H = 2 * (X.T @ X) / n_samples

    # 添加阻尼以提高数值稳定性
    H = H + damping * torch.eye(hidden_dim, device=X.device, dtype=X.dtype)

    # 计算逆
    try:
        H_inv = torch.inverse(H)
    except RuntimeError:
        # 如果失败，使用更大的阻尼
        print(f"  Warning: Hessian inversion failed, using larger damping (1e-4)")
        H = H + 1e-4 * torch.eye(hidden_dim, device=X.device, dtype=X.dtype)
        H_inv = torch.inverse(H)

    return H_inv


def prune_single_column(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    col_idx: int
) -> torch.Tensor:
    """
    剪枝单列并补偿其他列

    使用 OBS 公式：
    delta = -(W[:, col] / H_inv[col, col]) * H_inv[col, :]

    Args:
        W: 权重矩阵 [out_dim, in_dim]
        H_inv: Hessian 逆 [in_dim, in_dim]
        col_idx: 要剪枝的列索引

    Returns:
        W: 更新后的权重矩阵
    """
    # 检查是否已经被剪枝
    if (W[:, col_idx].abs() < 1e-10).all():
        return W

    # 检查 H_inv 对角元素
    if H_inv[col_idx, col_idx].abs() < 1e-10:
        W[:, col_idx] = 0
        return W

    # 计算补偿
    w_col = W[:, col_idx:col_idx + 1]  # [out_dim, 1]
    h_row = H_inv[col_idx:col_idx + 1, :]  # [1, in_dim]

    # delta = -(w / h_ii) * h_i
    delta = -(w_col / H_inv[col_idx, col_idx]) @ h_row  # [out_dim, in_dim]

    # 更新权重
    W = W + delta
    W[:, col_idx] = 0  # 置零剪枝列

    return W


def compute_column_error(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    col_idx: int
) -> float:
    """
    计算剪枝单列的误差

    error = sum(W[:, col]^2) / H_inv[col, col]

    Args:
        W: 权重矩阵
        H_inv: Hessian 逆
        col_idx: 列索引

    Returns:
        error: 剪枝误差
    """
    if H_inv[col_idx, col_idx].abs() < 1e-10:
        return float('inf')

    w_col = W[:, col_idx]
    error = (w_col ** 2).sum() / H_inv[col_idx, col_idx]

    return error.item()


def prune_columns_range(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    start: int,
    end: int
) -> torch.Tensor:
    """
    剪枝连续的多列（用于 head 剪枝）

    Args:
        W: 权重矩阵
        H_inv: Hessian 逆
        start: 起始列索引
        end: 结束列索引（不包含）

    Returns:
        W: 更新后的权重
    """
    for col in range(start, end):
        W = prune_single_column(W, H_inv, col)

    return W


def prune_attention_heads(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    num_heads: int,
    pruning_ratio: float,
    head_dim: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, List[int]]:
    """
    使用分组 Cholesky 分解剪枝 Attention heads

    Args:
        W: o_proj 权重 [hidden_dim, hidden_dim]
        H_inv: Hessian 逆 [hidden_dim, hidden_dim]
        num_heads: 头数
        pruning_ratio: 剪枝率
        head_dim: 每个头的维度
        device: 设备

    Returns:
        W: 剪枝后的权重
        pruned_heads: 被剪枝的 head 索引列表
    """
    num_prune = int(num_heads * pruning_ratio)
    pruned_heads = []

    W = W.clone()

    for _ in range(num_prune):
        # 计算每个 head 的误差
        head_errors = []

        for h in range(num_heads):
            if h in pruned_heads:
                head_errors.append(float('inf'))
                continue

            start = h * head_dim
            end = start + head_dim

            # 提取该 head 对应的子矩阵
            H_sub = H_inv[start:end, start:end]
            W_head = W[:, start:end]

            try:
                # Cholesky 分解
                L = torch.linalg.cholesky(H_sub)
                diag_L = torch.diagonal(L)

                # 计算该 head 的误差
                error = ((W_head ** 2) / (diag_L ** 2)).sum()
                head_errors.append(error.item())
            except RuntimeError:
                # Cholesky 失败，跳过
                head_errors.append(float('inf'))

        # 选择误差最小的 head
        head_to_prune = np.argmin(head_errors)

        if head_errors[head_to_prune] == float('inf'):
            print(f"  Warning: All remaining heads have infinite error, stopping")
            break

        # 剪枝该 head
        start = head_to_prune * head_dim
        end = start + head_dim
        W = prune_columns_range(W, H_inv, start, end)

        pruned_heads.append(head_to_prune)

    return W, pruned_heads


def prune_ffn_channels(
    W: torch.Tensor,
    H_inv: torch.Tensor,
    pruning_ratio: float,
    group_sizes: List[int] = None
) -> Tuple[torch.Tensor, List[int]]:
    """
    使用动态分组策略剪枝 FFN 通道

    Args:
        W: down_proj 权重 [hidden_dim, intermediate_dim]
        H_inv: Hessian 逆 [intermediate_dim, intermediate_dim]
        pruning_ratio: 剪枝率
        group_sizes: 组大小列表（默认 [1024, 512, 256, 128, 64, 32, 16, 8]）

    Returns:
        W: 剪枝后的权重
        pruned_channels: 被剪枝的通道索引列表
    """
    if group_sizes is None:
        group_sizes = [1024, 512, 256, 128, 64, 32, 16, 8]

    num_channels = W.shape[1]
    num_prune = int(num_channels * pruning_ratio)
    pruned_channels = []

    W = W.clone()

    for group_size in group_sizes:
        while len(pruned_channels) < num_prune:
            # 计算所有未剪枝通道的误差
            errors = torch.full((num_channels,), float('inf'), device=W.device)

            for i in range(num_channels):
                if i not in pruned_channels and (W[:, i].abs() > 1e-10).any():
                    errors[i] = compute_column_error(W, H_inv, i)

            # 选择 top-k 最小误差的通道
            k = min(group_size, num_prune - len(pruned_channels))
            if k == 0:
                break

            _, indices = torch.topk(errors, k, largest=False)

            # 剪枝这些通道
            for idx in indices:
                idx = idx.item()
                if errors[idx] == float('inf'):
                    continue

                W = prune_single_column(W, H_inv, idx)
                pruned_channels.append(idx)

                if len(pruned_channels) >= num_prune:
                    break

        if len(pruned_channels) >= num_prune:
            break

    return W, pruned_channels


def compute_layer_pruning_ratios(
    num_layers: int,
    target_ratio: float,
    strategy: str = 'log'
) -> np.ndarray:
    """
    计算各层的剪枝率（对数增长策略）

    Args:
        num_layers: 层数
        target_ratio: 目标总剪枝率
        strategy: 策略 ('uniform', 'log')

    Returns:
        layer_ratios: 各层的剪枝率 [num_layers]
    """
    if strategy == 'uniform':
        return np.full(num_layers, target_ratio)

    elif strategy == 'log':
        # 对数增长：浅层剪得少，深层剪得多
        r0 = 0.5 * target_ratio
        rn = 1.5 * target_ratio

        layer_ratios = []
        for i in range(num_layers):
            if num_layers == 1:
                r = target_ratio
            else:
                r = r0 + (rn - r0) * np.log(i + 1) / np.log(num_layers)
            layer_ratios.append(r)

        # 归一化到目标剪枝率
        layer_ratios = np.array(layer_ratios)
        layer_ratios = layer_ratios * target_ratio / layer_ratios.mean()

        # 限制范围 [0, 1]
        layer_ratios = np.clip(layer_ratios, 0.0, 1.0)

        return layer_ratios

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
