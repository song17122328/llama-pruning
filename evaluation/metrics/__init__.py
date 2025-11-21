"""
评估指标模块

包含:
- PPL (Perplexity) 评估
- Zero-shot / Few-shot 评估
- 效率指标 (速度、显存)
"""

# PPL 评估 (核心实现)
from .ppl import PPLMetric, evaluate_perplexity

# 性能评估
from .performance import evaluate_ppl, evaluate_zeroshot, evaluate_fewshot

# 自定义 Zero-shot 评估 (不依赖 lm-eval)
from .zeroshot import (
    evaluate_zeroshot_custom,
    evaluate_multiple_choice_batched,
    compute_loglikelihood_batched
)

# 效率评估
from .efficiency import evaluate_efficiency, measure_inference_speed

__all__ = [
    # PPL
    'PPLMetric',
    'evaluate_perplexity',

    # Performance
    'evaluate_ppl',
    'evaluate_zeroshot',
    'evaluate_fewshot',
    'evaluate_zeroshot_custom',
    'evaluate_multiple_choice_batched',
    'compute_loglikelihood_batched',

    # Efficiency
    'evaluate_efficiency',
    'measure_inference_speed'
]
