"""
剪枝方法模块
包含各种神经网络剪枝算法的实现
"""

from .gqa_aware import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

from .global_pruning import (
    build_global_group_table,
    select_groups_to_prune,
    save_group_table,
    GroupInfo
)

__all__ = [
    # GQA-aware 剪枝
    'compute_gqa_group_importance',
    'select_gqa_groups_to_prune',
    'prune_attention_by_gqa_groups',
    # 全局剪枝
    'build_global_group_table',
    'select_groups_to_prune',
    'save_group_table',
    'GroupInfo',
]
