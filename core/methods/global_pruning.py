#!/usr/bin/env python3
"""
全局剪枝策略：基于 Score = Importance / Cost

核心思想：
1. 计算每个 group 的 importance（Taylor 或 Wanda）
2. 计算每个 group 的 cost（参数量）
3. 计算 Score = importance / cost
4. 全局排序，选择 Score 最低的 groups 剪枝
"""

import math
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GroupInfo:
    """Group 信息"""
    layer_idx: int
    group_type: str  # 'attention' or 'mlp'
    group_idx: int
    importance: float
    cost: int  # 参数量
    score: float  # importance / cost

    def __repr__(self):
        return f"Layer{self.layer_idx}-{self.group_type}-{self.group_idx}: score={self.score:.6f}"


def compute_attention_group_importance_taylor(layer, head_dim=128, gqa_ratio=4, hessian_diag=None, layer_idx=None):
    """
    计算 Attention 每个 GQA group 的 Taylor importance

    支持一阶和二阶泰勒展开：
    - 一阶: importance = |weight × gradient|
    - 二阶: importance = |weight × gradient| + 0.5 × |weight² × hessian_diag|

    Args:
        layer: Transformer层
        head_dim: head维度
        gqa_ratio: Q:KV比例
        hessian_diag: Hessian对角线（可选，用于二阶）
        layer_idx: 层索引（用于构建Hessian键名）

    Returns:
        group_importance: Tensor [num_kv_heads]
    """
    salience = {}
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        sub_layer = getattr(layer.self_attn, name)
        # 一阶项
        first_order = (sub_layer.weight * sub_layer.weight.grad).abs()

        # 二阶项（如果提供了 Hessian）
        if hessian_diag is not None and layer_idx is not None:
            full_name = f'model.layers.{layer_idx}.self_attn.{name}.weight'
            if full_name in hessian_diag:
                # Hessian 存储在CPU上，需要移动到与weight相同的设备
                hess = hessian_diag[full_name].to(sub_layer.weight.device)
                second_order = 0.5 * (sub_layer.weight ** 2 * hess).abs()
                salience[name] = first_order + second_order
            else:
                salience[name] = first_order
        else:
            salience[name] = first_order

    q_imp = salience['q_proj'].sum(1)
    k_imp = salience['k_proj'].sum(1)
    v_imp = salience['v_proj'].sum(1)
    o_imp = salience['o_proj'].sum(0)

    num_q_heads = q_imp.shape[0] // head_dim
    num_kv_heads = k_imp.shape[0] // head_dim

    q_head_imp = q_imp.view(num_q_heads, head_dim).sum(1)
    k_head_imp = k_imp.view(num_kv_heads, head_dim).sum(1)
    v_head_imp = v_imp.view(num_kv_heads, head_dim).sum(1)
    o_head_imp = o_imp.view(num_q_heads, head_dim).sum(1)

    group_importance = torch.zeros(num_kv_heads, device=q_imp.device)

    for kv_idx in range(num_kv_heads):
        q_start = kv_idx * gqa_ratio
        q_end = q_start + gqa_ratio

        group_imp = 0.0
        group_imp += q_head_imp[q_start:q_end].sum()
        group_imp += o_head_imp[q_start:q_end].sum()
        group_imp += k_head_imp[kv_idx]
        group_imp += v_head_imp[kv_idx]

        group_importance[kv_idx] = group_imp

    return group_importance


def compute_attention_group_importance_wanda(layer, activations, head_dim=128, gqa_ratio=4):
    """
    计算 Attention 每个 GQA group 的 Wanda importance (优化版：矩阵乘法)

    关键优化：
    1. 使用矩阵乘法代替广播，避免生成巨大中间矩阵
    2. Q/K/V 剪行: Score_i = (|W| @ A)_i
    3. O 剪列: Score_j = A_j * sum_i |W_ij|

    Args:
        layer: Attention layer
        activations: Dict with keys 'q_proj', 'k_proj', 'v_proj', 'o_proj'
                    每个是 Tensor [hidden_dim] (L2 Norm)
        head_dim: head 维度
        gqa_ratio: Q:KV 比例

    Returns:
        group_importance: Tensor [num_kv_heads]
    """
    salience = {}

    # 计算 Q, K, V (剪掉 Output Rows -> 对应 Head)
    # Score_Row_i = sum_j (|W_ij| * A_j) = (|W| @ A)_i
    for name in ['q_proj', 'k_proj', 'v_proj']:
        sub_layer = getattr(layer.self_attn, name)
        act = activations.get(name)

        if act is None:
            raise ValueError(f"Missing activation for {name}")

        # 移动到与 weight 相同的设备
        act = act.to(sub_layer.weight.device)

        # 矩阵向量乘法，得到每个 Output Channel 的得分
        # |W|: [Out, In], act: [In] -> [Out]
        score = torch.matmul(sub_layer.weight.abs(), act)
        salience[name] = score

    # 计算 O (剪掉 Input Cols -> 对应 Head)
    # Score_Col_j = sum_i (|W_ij| * A_j) = A_j * sum_i |W_ij|
    name = 'o_proj'
    sub_layer = getattr(layer.self_attn, name)
    act = activations.get(name)

    if act is None:
        raise ValueError(f"Missing activation for {name}")

    act = act.to(sub_layer.weight.device)

    # 先求权重的列和 [In], 再乘以激活值
    w_col_sum = sub_layer.weight.abs().sum(dim=0)  # [In]
    score = w_col_sum * act  # [In]
    salience[name] = score

    # 聚合到 Heads (逻辑不变)
    q_imp = salience['q_proj']  # [num_q_heads * head_dim]
    k_imp = salience['k_proj']  # [num_kv_heads * head_dim]
    v_imp = salience['v_proj']  # [num_kv_heads * head_dim]
    o_imp = salience['o_proj']  # [num_q_heads * head_dim]

    num_q_heads = q_imp.shape[0] // head_dim
    num_kv_heads = k_imp.shape[0] // head_dim

    # reshape 并求和，得到每个 Head 的总分
    q_head_imp = q_imp.view(num_q_heads, head_dim).sum(1)
    k_head_imp = k_imp.view(num_kv_heads, head_dim).sum(1)
    v_head_imp = v_imp.view(num_kv_heads, head_dim).sum(1)
    o_head_imp = o_imp.view(num_q_heads, head_dim).sum(1)

    group_importance = torch.zeros(num_kv_heads, device=q_imp.device)

    for kv_idx in range(num_kv_heads):
        q_start = kv_idx * gqa_ratio
        q_end = q_start + gqa_ratio

        # 累加一个 GQA Group 内所有相关参数的 Wanda Score
        group_imp = (
            q_head_imp[q_start:q_end].sum() +
            o_head_imp[q_start:q_end].sum() +
            k_head_imp[kv_idx] +
            v_head_imp[kv_idx]
        )
        group_importance[kv_idx] = group_imp

    return group_importance


def compute_mlp_group_importance_taylor(layer, hessian_diag=None, layer_idx=None):
    """
    计算 MLP 每个通道的 Taylor importance

    支持一阶和二阶泰勒展开：
    - 一阶: importance = |weight × gradient|
    - 二阶: importance = |weight × gradient| + 0.5 × |weight² × hessian_diag|

    Args:
        layer: Transformer层
        hessian_diag: Hessian对角线（可选，用于二阶）
        layer_idx: 层索引（用于构建Hessian键名）

    Returns:
        channel_importance: Tensor [intermediate_size]
    """
    # 一阶项
    gate_salience = (layer.mlp.gate_proj.weight * layer.mlp.gate_proj.weight.grad).abs().sum(1)
    up_salience = (layer.mlp.up_proj.weight * layer.mlp.up_proj.weight.grad).abs().sum(1)
    down_salience = (layer.mlp.down_proj.weight * layer.mlp.down_proj.weight.grad).abs().sum(0)

    # 二阶项（如果提供了 Hessian）
    if hessian_diag is not None and layer_idx is not None:
        for name, sal_var in [('gate_proj', 'gate_salience'),
                               ('up_proj', 'up_salience'),
                               ('down_proj', 'down_salience')]:
            full_name = f'model.layers.{layer_idx}.mlp.{name}.weight'
            if full_name in hessian_diag:
                sub_layer = getattr(layer.mlp, name)
                # Hessian 存储在CPU上，需要移动到与weight相同的设备
                hess = hessian_diag[full_name].to(sub_layer.weight.device)
                second_order = 0.5 * (sub_layer.weight ** 2 * hess).abs()

                # 累加到对应的 salience 变量
                if name == 'gate_proj':
                    gate_salience = gate_salience + second_order.sum(1)
                elif name == 'up_proj':
                    up_salience = up_salience + second_order.sum(1)
                else:  # down_proj
                    down_salience = down_salience + second_order.sum(0)

    channel_importance = gate_salience + up_salience + down_salience
    return channel_importance


def compute_mlp_group_importance_wanda(layer, activations):
    """
    计算 MLP 每个通道的 Wanda importance (优化版：矩阵乘法 + 正确 Hook)

    关键修正：
    1. 使用矩阵乘法代替广播
    2. down_proj 使用正确的 Hook 位置（包含 SwiGLU 作用）

    Args:
        layer: MLP layer
        activations: Dict with keys 'gate_proj', 'up_proj', 'down_proj'
                    每个是 Tensor [hidden_dim] (L2 Norm)

    Returns:
        channel_importance: Tensor [intermediate_size]
    """
    # 1. Gate & Up Proj (剪行 -> Output Rows)
    # 输入是 gate_proj (即 mlp_input)
    act_in = activations.get('gate_proj')
    if act_in is None:
        raise ValueError("Missing gate_proj activation")

    act_in = act_in.to(layer.mlp.gate_proj.weight.device)

    # |W| @ A -> [Intermediate_Size]
    gate_imp = torch.matmul(layer.mlp.gate_proj.weight.abs(), act_in)
    up_imp = torch.matmul(layer.mlp.up_proj.weight.abs(), act_in)

    # 2. Down Proj (剪列 -> Input Cols)
    # 【关键】输入是 down_proj (即 intermediate，包含 SwiGLU 作用)
    act_mid = activations.get('down_proj')
    if act_mid is None:
        raise ValueError("Missing down_proj activation")

    act_mid = act_mid.to(layer.mlp.down_proj.weight.device)

    # A * (|W|.sum(0)) -> [Intermediate_Size]
    w_col_sum = layer.mlp.down_proj.weight.abs().sum(dim=0)
    down_imp = w_col_sum * act_mid

    # 3. 聚合 (对应同一个神经元)
    channel_importance = gate_imp + up_imp + down_imp
    return channel_importance


def compute_attention_group_importance_magnitude(layer, head_dim=128, gqa_ratio=4):
    """
    计算 Attention 每个 GQA group 的 Magnitude importance（权重绝对值）

    Args:
        layer: Transformer层
        head_dim: head维度
        gqa_ratio: Q:KV比例

    Returns:
        group_importance: Tensor [num_kv_heads]
    """
    salience = {}
    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        sub_layer = getattr(layer.self_attn, name)
        # Magnitude: 只使用权重的绝对值
        salience[name] = sub_layer.weight.abs()

    # 计算每个输出通道的重要性
    q_imp = salience['q_proj'].sum(1)
    k_imp = salience['k_proj'].sum(1)
    v_imp = salience['v_proj'].sum(1)
    o_imp = salience['o_proj'].sum(0)

    num_q_heads = q_imp.shape[0] // head_dim
    num_kv_heads = k_imp.shape[0] // head_dim

    q_head_imp = q_imp.view(num_q_heads, head_dim).sum(1)
    k_head_imp = k_imp.view(num_kv_heads, head_dim).sum(1)
    v_head_imp = v_imp.view(num_kv_heads, head_dim).sum(1)
    o_head_imp = o_imp.view(num_q_heads, head_dim).sum(1)

    group_importance = torch.zeros(num_kv_heads, device=q_imp.device)

    for kv_idx in range(num_kv_heads):
        q_start = kv_idx * gqa_ratio
        q_end = q_start + gqa_ratio

        group_imp = 0.0
        group_imp += q_head_imp[q_start:q_end].sum()
        group_imp += o_head_imp[q_start:q_end].sum()
        group_imp += k_head_imp[kv_idx]
        group_imp += v_head_imp[kv_idx]

        group_importance[kv_idx] = group_imp

    return group_importance


def compute_mlp_group_importance_magnitude(layer):
    """
    计算 MLP 每个通道的 Magnitude importance（权重绝对值）

    Args:
        layer: Transformer层

    Returns:
        channel_importance: Tensor [intermediate_size]
    """
    # Magnitude: 只使用权重的绝对值
    gate_salience = layer.mlp.gate_proj.weight.abs().sum(1)
    up_salience = layer.mlp.up_proj.weight.abs().sum(1)
    down_salience = layer.mlp.down_proj.weight.abs().sum(0)

    channel_importance = gate_salience + up_salience + down_salience
    return channel_importance


def compute_attention_group_cost(layer, group_idx, head_dim=128, gqa_ratio=4):
    """
    计算单个 Attention GQA group 的参数量

    Returns:
        cost: 参数量
    """
    # 1个 KV head + 4个 Q heads
    # q_proj: hidden_dim → (4 * head_dim)
    # k_proj: hidden_dim → head_dim
    # v_proj: hidden_dim → head_dim
    # o_proj: (4 * head_dim) → hidden_dim

    hidden_dim = layer.self_attn.q_proj.in_features

    cost = 0
    # Q heads (4个)
    cost += hidden_dim * (gqa_ratio * head_dim)  # q_proj
    # KV heads (1个)
    cost += hidden_dim * head_dim  # k_proj
    cost += hidden_dim * head_dim  # v_proj
    # O proj (4个 Q heads 对应的部分)
    cost += (gqa_ratio * head_dim) * hidden_dim  # o_proj

    return cost


def compute_mlp_group_cost(layer):
    """
    计算单个 MLP 通道的参数量

    Returns:
        cost: 参数量
    """
    # 每个通道：
    # gate_proj: hidden_dim → 1
    # up_proj: hidden_dim → 1
    # down_proj: 1 → hidden_dim

    hidden_dim = layer.mlp.gate_proj.in_features

    cost = 0
    cost += hidden_dim  # gate_proj 的一行
    cost += hidden_dim  # up_proj 的一行
    cost += hidden_dim  # down_proj 的一列

    return cost


def build_global_group_table(
    model,
    importance_method='taylor',
    importance_info=None,
    layer_start=0,
    layer_end=None,
    head_dim=128,
    gqa_ratio=4,
    device='cuda',
    layer_removal_ppl=None,
    block_removal_ppl=None,
    temperature=1.0,
    tau=None
) -> pd.DataFrame:
    """
    构建全局 Group 分析表（H-GSP 统一算法）

    核心算法：
    1. 计算动态阈值 τ = 25th percentile(ppl_layer)
    2. 混合加权：
       - 如果 ppl_layer < τ: Layer-Dominant Mode, B = ln(1 + ppl_layer)
       - 如果 ppl_layer >= τ: Block-Dominant Mode, B = ln(1 + ppl_block)
       - M = B^T (温度调制)
       - S_final = S_base * M

    Args:
        model: LLaMA 模型
        importance_method: 'taylor', 'taylor_2nd', 'wanda' 或 'magnitude'
        importance_info: 重要性信息字典
        layer_start: 起始层
        layer_end: 结束层
        head_dim: head 维度
        gqa_ratio: Q:KV 比例
        device: 设备
        layer_removal_ppl: 层级重要度 Dict[int, float]
        block_removal_ppl: 块级重要度 Dict[str, Dict[int, float]]
        temperature: 温度参数 T
            - T=0: 退化为纯基础方法（Taylor/Magnitude）
            - T=1: 推荐，平衡基础方法与全局先验
            - T>1: 激进模式，强化首尾保护
        tau: 门控阈值（None则自动计算为25分位数）
            - tau=0: 纯 Block-wise
            - tau=inf: 纯 Layer-wise
            - tau=None: 自动计算（推荐）

    Returns:
        DataFrame with columns: layer_idx, group_type, group_idx, importance, cost, score
    """
    if layer_end is None:
        layer_end = len(model.model.layers)

    num_layers = layer_end - layer_start

    # 提取重要性信息
    activations = None
    hessian_diag = None

    if importance_info is not None:
        if importance_method == 'wanda':
            activations = importance_info.get('activations')
            if activations is None:
                raise ValueError("Wanda method requires 'activations' in importance_info")
        elif importance_method in ['taylor', 'taylor_2nd']:
            # Taylor 方法不需要额外提取（梯度已经在模型中）
            if importance_method == 'taylor_2nd':
                hessian_diag = importance_info.get('hessian_diag')
                if hessian_diag is None:
                    raise ValueError("Second-order Taylor requires 'hessian_diag' in importance_info")

    # ========== Step A: 动态阈值计算 (τ) ==========
    computed_tau = tau
    if tau is None and layer_removal_ppl is not None:
        # 自动计算 25 分位数
        import numpy as np
        ppl_values = list(layer_removal_ppl.values())
        computed_tau = np.percentile(ppl_values, 25)
        print(f"\n[H-GSP] 自动计算门控阈值 τ = {computed_tau:.4f} (25th percentile)")
    elif tau is not None:
        computed_tau = tau
        print(f"\n[H-GSP] 使用指定门控阈值 τ = {computed_tau:.4f}")
    else:
        computed_tau = None
        print(f"\n[H-GSP] 未提供层级重要度数据，使用基础 Taylor 评分")

    print(f"\n{'='*60}")
    print(f"构建全局 Group 分析表 (H-GSP)")
    print(f"{'='*60}")
    print(f"重要性方法: {importance_method}")
    if importance_method == 'taylor_2nd':
        print(f"  使用二阶泰勒展开")

    # 打印 H-GSP 配置
    if computed_tau is not None and layer_removal_ppl is not None:
        print(f"\nH-GSP 配置:")
        print(f"  温度 T = {temperature}")
        if temperature == 0:
            print(f"    → 退化为纯 Taylor (无先验)")
        elif temperature == 1:
            print(f"    → 推荐模式 (平衡)")
        else:
            print(f"    → 激进模式 (强化首尾保护)")

        print(f"  门控阈值 τ = {computed_tau:.4f}")
        if computed_tau == 0:
            print(f"    → 纯 Block-wise 模式")
        elif computed_tau == float('inf'):
            print(f"    → 纯 Layer-wise 模式")
        else:
            # 统计会触发 Layer-Dominant 的层数
            layer_dominant_count = sum(1 for ppl in layer_removal_ppl.values() if ppl < computed_tau)
            print(f"    → 混合模式: {layer_dominant_count}/{num_layers} 层触发 Layer-Dominant")

        if block_removal_ppl is not None:
            print(f"  块级数据: ✓ Attention + MLP")
        else:
            print(f"  块级数据: ✗ 仅使用层级数据")

    print(f"\n层范围: [{layer_start}, {layer_end})")
    print(f"总层数: {num_layers}")

    all_groups = []

    for layer_idx in range(layer_start, layer_end):
        layer = model.model.layers[layer_idx]

        # ========== Attention Groups ==========
        if importance_method in ['taylor', 'taylor_2nd']:
            attn_importance = compute_attention_group_importance_taylor(
                layer, head_dim, gqa_ratio, hessian_diag=hessian_diag, layer_idx=layer_idx
            )
        elif importance_method == 'magnitude':
            attn_importance = compute_attention_group_importance_magnitude(
                layer, head_dim, gqa_ratio
            )
        else:  # wanda
            layer_activations = activations.get(layer_idx, {})
            attn_importance = compute_attention_group_importance_wanda(
                layer, layer_activations, head_dim, gqa_ratio
            )

        num_kv_heads = attn_importance.shape[0]

        for group_idx in range(num_kv_heads):
            importance = attn_importance[group_idx].item()
            cost = compute_attention_group_cost(layer, group_idx, head_dim, gqa_ratio)
            score = importance / cost if cost > 0 else 0.0

            # ========== Step B: 混合加权评分 (H-GSP) ==========
            if computed_tau is not None and layer_removal_ppl is not None and layer_idx in layer_removal_ppl:
                ppl_layer = layer_removal_ppl[layer_idx]

                # 判断模式
                if ppl_layer < computed_tau:
                    # Layer-Dominant Mode: 强制压低得分，鼓励整层移除
                    B = math.log(1 + ppl_layer)
                else:
                    # Block-Dominant Mode: 根据 Attention 块具体表现精细评分
                    if block_removal_ppl is not None and layer_idx in block_removal_ppl.get('attention', {}):
                        ppl_block = block_removal_ppl['attention'][layer_idx]
                        B = math.log(1 + ppl_block)
                    else:
                        # Fallback: 使用层级数据
                        B = math.log(1 + ppl_layer)

                # 应用温度调制
                M = B ** temperature
                score = score * M

            group = GroupInfo(
                layer_idx=layer_idx,
                group_type='attention',
                group_idx=group_idx,
                importance=importance,
                cost=cost,
                score=score
            )
            all_groups.append(group)

        # ========== MLP Groups ==========
        if importance_method in ['taylor', 'taylor_2nd']:
            mlp_importance = compute_mlp_group_importance_taylor(layer, hessian_diag=hessian_diag, layer_idx=layer_idx)
        elif importance_method == 'magnitude':
            mlp_importance = compute_mlp_group_importance_magnitude(layer)
        else:  # wanda
            layer_activations = activations.get(layer_idx, {})
            mlp_importance = compute_mlp_group_importance_wanda(layer, layer_activations)

        intermediate_size = mlp_importance.shape[0]
        mlp_group_cost = compute_mlp_group_cost(layer)

        for group_idx in range(intermediate_size):
            importance = mlp_importance[group_idx].item()
            cost = mlp_group_cost  # 每个通道的成本相同
            score = importance / cost if cost > 0 else 0.0

            # ========== Step B: 混合加权评分 (H-GSP) ==========
            if computed_tau is not None and layer_removal_ppl is not None and layer_idx in layer_removal_ppl:
                ppl_layer = layer_removal_ppl[layer_idx]

                # 判断模式
                if ppl_layer < computed_tau:
                    # Layer-Dominant Mode: 强制压低得分，鼓励整层移除
                    B = math.log(1 + ppl_layer)
                else:
                    # Block-Dominant Mode: 根据 MLP 块具体表现精细评分
                    if block_removal_ppl is not None and layer_idx in block_removal_ppl.get('mlp', {}):
                        ppl_block = block_removal_ppl['mlp'][layer_idx]
                        B = math.log(1 + ppl_block)
                    else:
                        # Fallback: 使用层级数据
                        B = math.log(1 + ppl_layer)

                # 应用温度调制
                M = B ** temperature
                score = score * M

            group = GroupInfo(
                layer_idx=layer_idx,
                group_type='mlp',
                group_idx=group_idx,
                importance=importance,
                cost=cost,
                score=score
            )
            all_groups.append(group)

        print(f"Layer {layer_idx}: {num_kv_heads} attention groups, {intermediate_size} mlp groups")

    # 转换为 DataFrame
    df = pd.DataFrame([
        {
            'layer_idx': g.layer_idx,
            'group_type': g.group_type,
            'group_idx': g.group_idx,
            'importance': g.importance,
            'cost': g.cost,
            'score': g.score
        }
        for g in all_groups
    ])

    # 按 Score 排序（从小到大）
    df = df.sort_values('score').reset_index(drop=True)

    # 统计信息
    print(f"\n{'='*60}")
    print(f"统计信息")
    print(f"{'='*60}")
    print(f"总 Groups: {len(df)}")
    print(f"  Attention: {len(df[df['group_type']=='attention'])}")
    print(f"  MLP: {len(df[df['group_type']=='mlp'])}")
    print(f"\nScore 统计:")
    print(f"  最小值: {df['score'].min():.6e}")
    print(f"  最大值: {df['score'].max():.6e}")
    print(f"  平均值: {df['score'].mean():.6e}")
    print(f"  中位数: {df['score'].median():.6e}")

    return df


def select_groups_to_prune(df: pd.DataFrame, pruning_ratio: float, total_params: int) -> pd.DataFrame:
    """
    根据剪枝率选择要剪枝的 groups

    Args:
        df: Group 分析表（已按 score 排序）
        pruning_ratio: 剪枝率（相对于模型总参数）
        total_params: 模型总参数量

    Returns:
        要剪枝的 groups 的 DataFrame
    """
    target_pruned_params = int(total_params * pruning_ratio)

    print(f"\n{'='*60}")
    print(f"选择要剪枝的 Groups")
    print(f"{'='*60}")
    print(f"模型总参数: {total_params:,}")
    print(f"剪枝率: {pruning_ratio:.1%}")
    print(f"目标剪枝量: {target_pruned_params:,}")

    # 从 Score 最低的开始累加
    cumsum_cost = df['cost'].cumsum()

    # 找到累计成本刚好超过目标的位置
    prune_mask = cumsum_cost <= target_pruned_params

    # 如果没有超过，说明要剪掉所有
    if prune_mask.sum() == 0:
        print("⚠️ 警告：目标剪枝量太小，至少剪掉1个group")
        prune_mask.iloc[0] = True

    groups_to_prune = df[prune_mask].copy()
    actual_pruned_params = groups_to_prune['cost'].sum()

    print(f"\n选中 {len(groups_to_prune)} 个 groups:")
    print(f"  Attention: {len(groups_to_prune[groups_to_prune['group_type']=='attention'])}")
    print(f"  MLP: {len(groups_to_prune[groups_to_prune['group_type']=='mlp'])}")
    print(f"  实际剪枝量: {actual_pruned_params:,} ({actual_pruned_params/total_params:.2%})")

    return groups_to_prune


def save_group_table(df: pd.DataFrame, save_path: str):
    """保存 Group 分析表"""
    # 保存为 CSV
    csv_path = save_path.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Group 表已保存: {csv_path}")

    # 同时保存为 JSON（方便程序读取）
    df.to_json(save_path, orient='records', indent=2)
    print(f"✓ Group 表已保存: {save_path}")


if __name__ == '__main__':
    print("Global Pruning Module - 基于 Score = Importance / Cost")
    print("\n示例用法:")
    print("""
    # 1. 构建全局分析表
    df = build_global_group_table(
        model,
        importance_method='taylor',
        layer_start=0,
        layer_end=32
    )

    # 2. 保存表格
    save_group_table(df, 'group_analysis.json')

    # 3. 选择要剪枝的 groups
    groups_to_prune = select_groups_to_prune(df, pruning_ratio=0.25, total_params=8_000_000_000)

    # 4. 执行剪枝
    # ... (根据 groups_to_prune 剪枝模型)
    """)
    model = "/newdata/LLMs/Llama-3-8B-Instruct"
    df = build_global_group_table(
        model,
        importance_method='taylor',
        layer_start=0,
        layer_end=32
    )
    save_group_table(df, 'group_analysis.json')