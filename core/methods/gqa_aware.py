"""
GQA-aware Taylor Importance 计算

核心思想：
将"4个Q heads + 1个KV head"视为一个GQA组，计算每个组的总importance，
根据组的importance选择要剪枝的完整组。

这样确保：
1. 剪枝时保持4:1的GQA比例
2. 保留的Q heads与KV heads的语义对应关系正确
3. 剪枝基于importance，而不是简单截断
"""

import torch
import torch.nn as nn

def compute_gqa_group_importance(layer, head_dim=128, gqa_ratio=4):
    """
    计算每个GQA组的Taylor importance

    Args:
        layer: Attention layer
        head_dim: 每个head的维度 (128 for Llama-3)
        gqa_ratio: Q:KV比例 (4 for Llama-3)

    Returns:
        group_importance: Tensor of shape [num_kv_heads]
                         每个GQA组的总importance
    """
    # 1. 计算每个sub-layer的salience (weight * grad)
    salience = {}
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        sub_layer = getattr(layer.self_attn, name)
        salience[name] = (sub_layer.weight * sub_layer.weight.grad).abs()

    # 2. 计算每个通道的importance
    # q_proj, k_proj, v_proj: input channels (dim 0)
    # o_proj: output channels (dim 1, 但我们关心input channels对应的output)
    q_imp = salience['q_proj'].sum(1)  # [num_q_heads * head_dim]
    k_imp = salience['k_proj'].sum(1)  # [num_kv_heads * head_dim]
    v_imp = salience['v_proj'].sum(1)  # [num_kv_heads * head_dim]
    o_imp = salience['o_proj'].sum(0)  # [num_q_heads * head_dim] (o_proj输入来自concat的Q heads)

    # 3. 将通道importance转换为head importance
    num_q_heads = q_imp.shape[0] // head_dim
    num_kv_heads = k_imp.shape[0] // head_dim

    q_head_imp = q_imp.view(num_q_heads, head_dim).sum(1)  # [num_q_heads]
    k_head_imp = k_imp.view(num_kv_heads, head_dim).sum(1)  # [num_kv_heads]
    v_head_imp = v_imp.view(num_kv_heads, head_dim).sum(1)  # [num_kv_heads]
    o_head_imp = o_imp.view(num_q_heads, head_dim).sum(1)  # [num_q_heads]

    # 4. 计算每个GQA组的importance
    group_importance = torch.zeros(num_kv_heads, device=q_imp.device)

    for kv_idx in range(num_kv_heads):
        # 这个KV head对应的4个Q heads的索引
        q_start = kv_idx * gqa_ratio
        q_end = q_start + gqa_ratio

        # GQA组的总importance = 对应的Q heads + K head + V head
        group_imp = 0.0

        # Q heads的contribution (4个heads)
        group_imp += q_head_imp[q_start:q_end].sum()
        group_imp += o_head_imp[q_start:q_end].sum()

        # KV heads的contribution (1个head)
        group_imp += k_head_imp[kv_idx]
        group_imp += v_head_imp[kv_idx]

        group_importance[kv_idx] = group_imp

    return group_importance


def select_gqa_groups_to_prune(group_importance, target_num_kv_heads):
    """
    根据importance选择要剪枝的GQA组

    Args:
        group_importance: Tensor [num_kv_heads], 每个组的importance
        target_num_kv_heads: 目标保留的KV head数量

    Returns:
        keep_indices: 要保留的KV head索引
        prune_indices: 要剪枝的KV head索引
    """
    num_kv_heads = len(group_importance)

    # 按importance从大到小排序
    sorted_indices = torch.argsort(group_importance, descending=True)

    # 保留importance最高的N个组
    keep_indices = sorted(sorted_indices[:target_num_kv_heads].tolist())
    prune_indices = sorted(sorted_indices[target_num_kv_heads:].tolist())

    return keep_indices, prune_indices


def prune_attention_by_gqa_groups(layer, keep_kv_indices, head_dim=128, gqa_ratio=4):
    """
    根据选择的GQA组剪枝Attention层

    Args:
        layer: Attention layer
        keep_kv_indices: 要保留的KV head索引列表
        head_dim: 每个head的维度
        gqa_ratio: Q:KV比例
    """
    # 1. 计算要保留的Q head索引
    keep_q_indices = []
    for kv_idx in keep_kv_indices:
        q_start = kv_idx * gqa_ratio
        q_end = q_start + gqa_ratio
        keep_q_indices.extend(range(q_start, q_end))

    # 2. 转换为通道索引
    keep_q_channels = []
    for q_idx in keep_q_indices:
        start = q_idx * head_dim
        end = start + head_dim
        keep_q_channels.extend(range(start, end))

    keep_kv_channels = []
    for kv_idx in keep_kv_indices:
        start = kv_idx * head_dim
        end = start + head_dim
        keep_kv_channels.extend(range(start, end))

    keep_q_channels = torch.LongTensor(keep_q_channels)
    keep_kv_channels = torch.LongTensor(keep_kv_channels)

    # 3. 剪枝各个projection层
    # q_proj: output channels
    layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[keep_q_channels, :]
    if layer.self_attn.q_proj.bias is not None:
        layer.self_attn.q_proj.bias.data = layer.self_attn.q_proj.bias.data[keep_q_channels]

    # k_proj, v_proj: output channels
    layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[keep_kv_channels, :]
    if layer.self_attn.k_proj.bias is not None:
        layer.self_attn.k_proj.bias.data = layer.self_attn.k_proj.bias.data[keep_kv_channels]

    layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[keep_kv_channels, :]
    if layer.self_attn.v_proj.bias is not None:
        layer.self_attn.v_proj.bias.data = layer.self_attn.v_proj.bias.data[keep_kv_channels]

    # o_proj: input channels (它接收concat后的Q heads)
    layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data[:, keep_q_channels]

    # 4. 更新配置
    # 注意：不同版本的transformers可能有不同的属性名称
    # 无论这些属性是否存在，我们都设置它们（创建或更新）
    layer.self_attn.num_heads = len(keep_q_indices)
    layer.self_attn.num_key_value_heads = len(keep_kv_indices)
    layer.self_attn.num_key_value_groups = gqa_ratio

    # 5. 同步Linear层属性（避免LoRA维度不匹配）
    # 这些属性是nn.Linear固有的，必须更新
    layer.self_attn.q_proj.out_features = len(keep_q_channels)
    layer.self_attn.q_proj.in_features = layer.self_attn.q_proj.weight.shape[1]

    layer.self_attn.k_proj.out_features = len(keep_kv_channels)
    layer.self_attn.k_proj.in_features = layer.self_attn.k_proj.weight.shape[1]

    layer.self_attn.v_proj.out_features = len(keep_kv_channels)
    layer.self_attn.v_proj.in_features = layer.self_attn.v_proj.weight.shape[1]

    layer.self_attn.o_proj.in_features = len(keep_q_channels)
    layer.self_attn.o_proj.out_features = layer.self_attn.o_proj.weight.shape[0]

    return len(keep_q_indices), len(keep_kv_indices)


# ===================== 使用示例 =====================

def prune_layer_with_gqa_awareness(model, layer_idx, pruning_rate, example_prompts):
    """
    使用GQA-aware方法剪枝单个layer

    Args:
        model: Llama模型
        layer_idx: 层索引
        pruning_rate: 剪枝率 (e.g., 0.25 表示剪掉25%的GQA组)
        example_prompts: 用于计算梯度的样本

    Returns:
        (num_q_heads, num_kv_heads): 剪枝后的head数量
    """
    layer = model.model.layers[layer_idx]
    head_dim = 128
    gqa_ratio = 4

    # 1. 前向+反向传播计算梯度
    model.zero_grad()
    loss = model(example_prompts, labels=example_prompts).loss
    loss.backward()

    # 2. 计算每个GQA组的importance
    group_imp = compute_gqa_group_importance(layer, head_dim, gqa_ratio)

    # 3. 确定要保留的GQA组数量
    num_kv_heads = len(group_imp)
    num_groups_to_prune = int(num_kv_heads * pruning_rate)
    target_num_kv_heads = num_kv_heads - num_groups_to_prune

    # 确保至少保留1个组
    target_num_kv_heads = max(1, target_num_kv_heads)

    # 4. 选择要保留的组（importance最高的）
    keep_indices, prune_indices = select_gqa_groups_to_prune(group_imp, target_num_kv_heads)

    print(f"Layer {layer_idx}:")
    print(f"  Original: {num_kv_heads} KV heads ({num_kv_heads * gqa_ratio} Q heads)")
    print(f"  Pruning rate: {pruning_rate:.2%}")
    print(f"  Prune {len(prune_indices)} GQA groups: {prune_indices}")
    print(f"  Keep {len(keep_indices)} GQA groups: {keep_indices}")

    # 5. 执行剪枝
    num_q, num_kv = prune_attention_by_gqa_groups(layer, keep_indices, head_dim, gqa_ratio)

    print(f"  Result: {num_kv} KV heads ({num_q} Q heads), ratio = {num_q//num_kv}:1 ✓")

    return num_q, num_kv
