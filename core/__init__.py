"""
LLaMA Pruning 核心模块
"""

# 剪枝方法
from .methods.gqa_aware import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups,
)

from .methods.global_pruning import (
    build_global_group_table,
    select_groups_to_prune,
    save_group_table,
    compute_attention_group_importance_taylor,
    compute_attention_group_importance_wanda,
    compute_mlp_group_importance_taylor,
    compute_mlp_group_importance_wanda,
)

# 重要性分析
from .importance.layer_analyzer import (
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator,
)

# 数据集
from .datasets import (
    get_examples,
    get_examples_from_text,
    DatasetManager,
    create_dataset_manager,
)

# 训练
from .trainer.finetuner import FineTuner

# 工具
from .utils.logger import LoggerWithDepth

# 评估（从 evaluation 目录导入）
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation'))
from metrics.ppl import PPLMetric

__all__ = [
    # 剪枝方法
    'compute_gqa_group_importance',
    'select_gqa_groups_to_prune',
    'prune_attention_by_gqa_groups',
    'build_global_group_table',
    'select_groups_to_prune',
    'save_group_table',
    'compute_attention_group_importance_taylor',
    'compute_attention_group_importance_wanda',
    'compute_mlp_group_importance_taylor',
    'compute_mlp_group_importance_wanda',

    # 重要性分析
    'LayerImportanceAnalyzer',
    'UnbalancedStructuredPruningCalculator',

    # 数据集
    'get_examples',
    'get_examples_from_text',
    'DatasetManager',
    'create_dataset_manager',

    # 训练
    'FineTuner',

    # 工具
    'LoggerWithDepth',

    # 评估
    'PPLMetric',
]
