#!/usr/bin/env python3
"""
层重要度分析工具 - 用于评估 Transformer 各层的重要性
结合结构化剪枝，实现非均衡剪枝
"""

import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


class LayerImportanceAnalyzer:
    """分析Transformer各层的重要性"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def compute_perplexity(self, texts: List[str]) -> float:
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for text in tqdm(texts, desc="计算困惑度"):
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                # 将inputs移动到模型第一层所在的设备
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)

        return np.exp(total_loss / total_tokens)

    def measure_layer_importance_by_removal(self, texts: List[str],
                                           num_layers: int) -> Dict[int, float]:
        """
        通过移除层来评估重要性（困惑度变化）
        重要性 = 移除该层后的困惑度增加量
        """
        baseline_ppl = self.compute_perplexity(texts)
        layer_importance = {}

        print(f"基准困惑度: {baseline_ppl:.4f}")

        for layer_idx in tqdm(range(num_layers), desc="分析层重要性"):
            # 保存原始forward函数
            original_forward = self.model.model.layers[layer_idx].forward

            # 定义恒等映射函数
            def identity_forward(hidden_states, *args, **kwargs):
                # 直接返回输入的hidden_states，跳过该层的计算
                # Llama 的 DecoderLayer forward 返回格式：
                # - 如果不返回额外信息：hidden_states
                # - 如果返回注意力权重：(hidden_states, self_attn_weights, present_key_value)

                # 检查是否需要返回额外信息
                output_attentions = kwargs.get('output_attentions', False)
                use_cache = kwargs.get('use_cache', False)

                if output_attentions or use_cache:
                    # 返回元组格式
                    outputs = (hidden_states,)
                    if output_attentions:
                        outputs += (None,)  # self_attn_weights
                    if use_cache:
                        outputs += (None,)  # present_key_value
                    return outputs
                else:
                    # 只返回 hidden_states
                    return hidden_states

            # 临时替换该层的forward
            self.model.model.layers[layer_idx].forward = identity_forward

            try:
                ppl = self.compute_perplexity(texts)
                importance = ppl - baseline_ppl  # 困惑度增加越多，该层越重要
                layer_importance[layer_idx] = importance

                # print(f"第 {layer_idx} 层: PPL 变化 = {importance:.4f}")
            finally:
                # 无论是否出错，都要恢复该层
                self.model.model.layers[layer_idx].forward = original_forward

        return layer_importance

    def measure_layer_importance_by_activation(self, texts: List[str]) -> Dict[int, float]:
        """通过激活值统计评估重要性"""
        layer_activations = {}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activation = output[0]
                else:
                    activation = output

                # 计算激活值的L2范数
                activation_norm = torch.norm(activation, p=2, dim=-1).mean().item()
                if layer_idx not in layer_activations:
                    layer_activations[layer_idx] = []
                layer_activations[layer_idx].append(activation_norm)
            return hook

        # 注册hooks
        hooks = []
        for idx, layer in enumerate(self.model.model.layers):
            hooks.append(layer.register_forward_hook(hook_fn(idx)))

        # 前向传播
        with torch.no_grad():
            for text in tqdm(texts, desc="收集激活值"):
                inputs = self.tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=512)
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
                self.model(**inputs)

        # 移除hooks
        for hook in hooks:
            hook.remove()

        # 计算平均激活值
        layer_importance = {idx: np.mean(acts) for idx, acts in layer_activations.items()}
        return layer_importance


class UnbalancedStructuredPruningCalculator:
    """
    非均衡结构化剪枝率计算器
    结合层重要度和目标剪枝率，计算每层的剪枝率
    """

    def __init__(self, layer_importance: Dict[int, float], num_layers: int):
        self.layer_importance = layer_importance
        self.num_layers = num_layers

    def compute_layer_pruning_rates(self,
                                    target_overall_rate: float,
                                    strategy: str = 'inverse',
                                    alpha: float = 1.0,
                                    min_rate: float = 0.0,
                                    max_rate: float = 0.8,
                                    use_log_transform: bool = True) -> Dict[int, float]:
        """
        根据层重要性计算各层剪枝率

        Args:
            target_overall_rate: 目标整体剪枝率（例如 0.25 表示减少25%的参数）
            strategy: 剪枝策略
                - 'inverse': 重要层剪少，不重要层剪多（默认）
                - 'proportional': 重要层剪多，不重要层剪少（反向）
                - 'uniform': 所有层使用相同剪枝率
            alpha: 重要性权重系数，越大差异越明显
            min_rate: 最小剪枝率
            max_rate: 最大剪枝率
            use_log_transform: 是否使用对数变换处理极端值（推荐）

        Returns:
            Dict[int, float]: 每层的剪枝率
        """
        if strategy == 'uniform':
            # 均匀剪枝
            return {idx: target_overall_rate for idx in range(self.num_layers)}

        importance_values = np.array(list(self.layer_importance.values()))

        # 对数变换处理极端值
        if use_log_transform:
            # 平移使所有值为正（最小值+1），然后取对数
            min_val = importance_values.min()
            shifted_importance = importance_values - min_val + 1.0
            log_importance = np.log(shifted_importance)
            importance_values = log_importance

        if strategy == 'inverse':
            # 重要性高 -> 剪枝率低
            # 归一化到 [0, 1]
            normalized_importance = (importance_values - importance_values.min()) / \
                                   (importance_values.max() - importance_values.min() + 1e-8)

            # 应用alpha系数增强差异
            normalized_importance = normalized_importance ** alpha

            # 反转：重要性高 -> 剪枝率低
            inverse_importance = 1 - normalized_importance

            # 缩放使得平均剪枝率等于目标剪枝率
            pruning_rates = inverse_importance * (target_overall_rate * self.num_layers / inverse_importance.sum())

        elif strategy == 'proportional':
            # 重要性高 -> 剪枝率高
            normalized_importance = (importance_values - importance_values.min()) / \
                                   (importance_values.max() - importance_values.min() + 1e-8)

            normalized_importance = normalized_importance ** alpha

            # 直接使用：重要性高 -> 剪枝率高
            pruning_rates = normalized_importance * (target_overall_rate * self.num_layers / normalized_importance.sum())

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 限制在合理范围内
        pruning_rates = np.clip(pruning_rates, min_rate, max_rate)

        # 重新归一化以确保平均剪枝率等于目标
        current_avg = pruning_rates.mean()
        if current_avg > 0:
            pruning_rates = pruning_rates * (target_overall_rate / current_avg)
            pruning_rates = np.clip(pruning_rates, min_rate, max_rate)

        return {idx: float(rate) for idx, rate in enumerate(pruning_rates)}

    def compute_layer_pruning_rates_by_target_params(
        self,
        layer_param_counts: Dict[int, int],
        target_total_pruned_params: int,
        strategy: str = 'inverse',
        alpha: float = 1.0,
        min_rate: float = 0.0,
        max_rate: float = 0.5,
        use_log_transform: bool = True
    ) -> Dict[int, float]:
        """
        根据目标总剪枝参数量计算各层剪枝率

        与 compute_layer_pruning_rates 的区别：
        - 输入是目标总剪枝参数量，而不是平均剪枝率
        - 考虑了每层的参数量差异

        Args:
            layer_param_counts: 每层的参数量 {layer_idx: param_count}
            target_total_pruned_params: 目标总剪枝参数量
            strategy: 剪枝策略 ('inverse', 'proportional', 'uniform')
            alpha: 重要性权重系数（越大层间差异越明显）
            min_rate: 单层最小剪枝率
            max_rate: 单层最大剪枝率
            use_log_transform: 是否对重要性使用对数变换

        Returns:
            Dict[int, float]: 各层剪枝率 {layer_idx: pruning_rate}
        """
        if not layer_param_counts:
            raise ValueError("layer_param_counts 不能为空")

        layer_indices = sorted(layer_param_counts.keys())
        param_counts = np.array([layer_param_counts[i] for i in layer_indices])
        total_params = param_counts.sum()

        if target_total_pruned_params <= 0:
            return {idx: 0.0 for idx in layer_indices}

        if target_total_pruned_params >= total_params:
            raise ValueError(f"目标剪枝参数量 ({target_total_pruned_params}) 超过总参数量 ({total_params})")

        # 获取层重要性
        importance_values = np.array([self.layer_importance.get(i, 1.0) for i in layer_indices])

        # 对数变换（可选）
        if use_log_transform:
            importance_values = np.log(importance_values + 1)

        # 根据策略计算权重
        if strategy == 'uniform':
            # 均匀分配
            weights = np.ones(self.num_layers)

        elif strategy == 'inverse':
            # 重要性高 -> 权重低 -> 剪枝少
            normalized_importance = (importance_values - importance_values.min()) / \
                                   (importance_values.max() - importance_values.min() + 1e-8)
            weights = 1.0 / (normalized_importance * alpha + 1.0)

        elif strategy == 'proportional':
            # 重要性高 -> 权重高 -> 剪枝多
            normalized_importance = (importance_values - importance_values.min()) / \
                                   (importance_values.max() - importance_values.min() + 1e-8)
            weights = normalized_importance ** alpha

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 归一化权重
        weights = weights / weights.sum()

        # 根据权重分配剪枝参数量到各层
        layer_pruned_params = weights * target_total_pruned_params

        # 计算各层剪枝率
        pruning_rates = layer_pruned_params / (param_counts + 1e-8)

        # 限制在合理范围内
        pruning_rates = np.clip(pruning_rates, min_rate, max_rate)

        # 重新调整以达到目标总剪枝参数量
        actual_pruned_params = (pruning_rates * param_counts).sum()
        if actual_pruned_params > 0:
            scale_factor = target_total_pruned_params / actual_pruned_params
            pruning_rates = pruning_rates * scale_factor
            pruning_rates = np.clip(pruning_rates, min_rate, max_rate)

        return {idx: float(rate) for idx, rate in zip(layer_indices, pruning_rates)}

    def verify_average_pruning_rate(self, layer_pruning_rates: Dict[int, float]) -> Dict[str, float]:
        """验证各层平均剪枝率"""
        rates = list(layer_pruning_rates.values())
        avg_rate = np.mean(rates)
        std_rate = np.std(rates)
        min_rate = np.min(rates)
        max_rate = np.max(rates)

        return {
            'average_pruning_rate': avg_rate,
            'std_pruning_rate': std_rate,
            'min_pruning_rate': min_rate,
            'max_pruning_rate': max_rate,
            'rate_range': max_rate - min_rate
        }

    def save_pruning_rates(self, layer_pruning_rates: Dict[int, float], filepath: str):
        """保存剪枝率配置到JSON文件"""
        config = {
            'layer_pruning_rates': {str(k): v for k, v in layer_pruning_rates.items()},
            'layer_importance': {str(k): v for k, v in self.layer_importance.items()},
            'statistics': self.verify_average_pruning_rate(layer_pruning_rates)
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"剪枝率配置已保存到: {filepath}")

    @staticmethod
    def load_pruning_rates(filepath: str) -> Dict[int, float]:
        """从JSON文件加载剪枝率配置"""
        with open(filepath, 'r') as f:
            config = json.load(f)

        return {int(k): v for k, v in config['layer_pruning_rates'].items()}

    def visualize_pruning_strategy(self, layer_pruning_rates: Dict[int, float],
                                   save_path: str = None):
        """可视化剪枝策略"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 绘制层重要性
        layers = sorted(self.layer_importance.keys())
        importance_values = [self.layer_importance[i] for i in layers]

        ax1.bar(layers, importance_values, alpha=0.7, color='steelblue')
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Importance Score', fontsize=12)
        ax1.set_title('Layer Importance Analysis', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 绘制剪枝率
        pruning_values = [layer_pruning_rates[i] for i in layers]

        ax2.bar(layers, pruning_values, alpha=0.7, color='coral')
        ax2.axhline(y=np.mean(pruning_values), color='r', linestyle='--',
                   label=f'Average: {np.mean(pruning_values):.4f}')
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Pruning Rate', fontsize=12)
        ax2.set_title('Layer-wise Pruning Rate Distribution', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"可视化图表已保存到: {save_path}")

        plt.show()



if __name__ == "__main__":
    print("""
    层重要度分析工具
    ================

    用法示例:

    1. 评估层重要性:
        analyzer = LayerImportanceAnalyzer(model, tokenizer)
        importance = analyzer.measure_layer_importance_by_removal(texts, num_layers=32)

    2. 计算非均衡剪枝率:
        calculator = UnbalancedStructuredPruningCalculator(importance, num_layers=32)
        pruning_rates = calculator.compute_layer_pruning_rates(
            target_overall_rate=0.25,
            strategy='inverse',
            alpha=1.0
        )
    """)
