#!/usr/bin/env python3
"""
全局剪枝策略：基于 Score = Importance / Cost

核心思想：
1. 计算每个 group 的 importance（Taylor 或 Wanda）
2. 计算每个 group 的 cost（参数量）
3. 计算 Score = importance / cost
4. 全局排序，选择 Score 最低的 groups 剪枝
"""

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
    计算 Attention 每个 GQA group 的 Wanda importance (weight × activation)

    Args:
        layer: Attention layer
        activations: Dict with keys 'q_proj', 'k_proj', 'v_proj', 'o_proj'
                    每个是 Tensor [batch, seq_len, hidden_dim] 或 [hidden_dim]
        head_dim: head 维度
        gqa_ratio: Q:KV 比例

    Returns:
        group_importance: Tensor [num_kv_heads]
    """
    salience = {}

    for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        sub_layer = getattr(layer.self_attn, name)
        act = activations.get(name)

        if act is None:
            raise ValueError(f"Missing activation for {name}")

        # 如果 activation 是多维的，取平均
        if act.dim() > 1:
            act = act.mean(dim=tuple(range(act.dim() - 1)))

        # weight × activation (按输入维度广播)
        # weight shape: [out_features, in_features]
        # act shape: [in_features]
        weight_act = sub_layer.weight.abs() * act.unsqueeze(0)
        salience[name] = weight_act

    # 计算每个输出通道的重要性
    q_imp = salience['q_proj'].sum(1)  # [num_q_heads * head_dim]
    k_imp = salience['k_proj'].sum(1)  # [num_kv_heads * head_dim]
    v_imp = salience['v_proj'].sum(1)  # [num_kv_heads * head_dim]
    o_imp = salience['o_proj'].sum(0)  # [num_q_heads * head_dim]

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
    计算 MLP 每个通道的 Wanda importance (weight × activation)

    Args:
        layer: MLP layer
        activations: Dict with keys 'gate_proj', 'up_proj', 'down_proj'

    Returns:
        channel_importance: Tensor [intermediate_size]
    """
    # gate_proj 和 up_proj: [intermediate_size, hidden_size]
    # down_proj: [hidden_size, intermediate_size]

    # 输入激活（MLP 的输入）
    mlp_input_act = activations.get('mlp_input')
    if mlp_input_act is None:
        raise ValueError("Missing mlp_input activation")

    if mlp_input_act.dim() > 1:
        mlp_input_act = mlp_input_act.mean(dim=tuple(range(mlp_input_act.dim() - 1)))

    # gate 和 up 的重要性（输入维度）
    gate_salience = (layer.mlp.gate_proj.weight.abs() * mlp_input_act.unsqueeze(0)).sum(1)
    up_salience = (layer.mlp.up_proj.weight.abs() * mlp_input_act.unsqueeze(0)).sum(1)

    # down 的输出激活（中间层激活）
    intermediate_act = activations.get('intermediate')
    if intermediate_act is None:
        raise ValueError("Missing intermediate activation")

    if intermediate_act.dim() > 1:
        intermediate_act = intermediate_act.mean(dim=tuple(range(intermediate_act.dim() - 1)))

    # down 的重要性（输入维度，即 intermediate_size）
    down_salience = (layer.mlp.down_proj.weight.abs() * intermediate_act.unsqueeze(0)).sum(0)

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
    device='cuda'
) -> pd.DataFrame:
    """
    构建全局 Group 分析表

    Args:
        model: LLaMA 模型
        importance_method: 'taylor', 'taylor_2nd' 或 'wanda'
        importance_info: 重要性信息字典
            - 对于 taylor/taylor_2nd: {'gradients': {...}, 'hessian_diag': {...}}
            - 对于 wanda: {'activations': {...}}
        layer_start: 起始层
        layer_end: 结束层
        head_dim: head 维度
        gqa_ratio: Q:KV 比例
        device: 设备

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

    print(f"\n{'='*60}")
    print(f"构建全局 Group 分析表")
    print(f"{'='*60}")
    print(f"重要性方法: {importance_method}")
    if importance_method == 'taylor_2nd':
        print(f"  使用二阶泰勒展开（包含 Hessian 对角线）")
    print(f"层范围: [{layer_start}, {layer_end})")
    print(f"总层数: {num_layers}")

    all_groups = []

    for layer_idx in range(layer_start, layer_end):
        layer = model.model.layers[layer_idx]

        # ========== Attention Groups ==========
        if importance_method in ['taylor', 'taylor_2nd']:
            attn_importance = compute_attention_group_importance_taylor(
                layer, head_dim, gqa_ratio, hessian_diag=hessian_diag, layer_idx=layer_idx
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
        else:  # wanda
            layer_activations = activations.get(layer_idx, {})
            mlp_importance = compute_mlp_group_importance_wanda(layer, layer_activations)

        intermediate_size = mlp_importance.shape[0]
        mlp_group_cost = compute_mlp_group_cost(layer)

        for group_idx in range(intermediate_size):
            importance = mlp_importance[group_idx].item()
            cost = mlp_group_cost  # 每个通道的成本相同
            score = importance / cost if cost > 0 else 0.0

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