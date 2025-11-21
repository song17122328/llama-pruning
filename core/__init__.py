"""
Core - 大语言模型剪枝核心库

包含以下模块：
- methods: 剪枝方法（GQA-aware等）
- importance: 重要性分析
- datasets: 数据集加载
- trainer: 模型微调
- utils: 工具函数

注意：PPL 评估已移至 evaluation/metrics/ppl.py
"""

# 剪枝方法
from .methods import (
    compute_gqa_group_importance,
    select_gqa_groups_to_prune,
    prune_attention_by_gqa_groups
)

# 重要性分析
from .importance import (
    LayerImportanceAnalyzer,
    UnbalancedStructuredPruningCalculator
)

# 数据集工具
from .datasets import get_examples, get_examples_from_text

# 工具函数
from .utils.logger import LoggerWithDepth
from .utils.get_best_gpu import get_best_gpu

# 训练模块
from .trainer import FineTuner

__version__ = '0.2.0'

__all__ = [
    # Methods
    'compute_gqa_group_importance',
    'select_gqa_groups_to_prune',
    'prune_attention_by_gqa_groups',

    # Importance
    'LayerImportanceAnalyzer',
    'UnbalancedStructuredPruningCalculator',

    # Datasets
    'get_examples',
    'get_examples_from_text',

    # Trainer
    'FineTuner',

    # Utils
    'LoggerWithDepth',
    'get_best_gpu',
]
