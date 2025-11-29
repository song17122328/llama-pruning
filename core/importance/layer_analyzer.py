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
from transformers import AutoTokenizer, LlamaForCausalLM

import sys,os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.datasets.example_samples import get_examples


class LayerImportanceAnalyzer:
    """分析Transformer各层的重要性"""

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def compute_perplexity(self, texts: List[str], show_progress: bool = True) -> float:
        """计算困惑度

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条（默认True）
        """
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            iterator = tqdm(texts, desc="计算困惑度", disable=not show_progress)
            for text in iterator:
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
        baseline_ppl = self.compute_perplexity(texts, show_progress=False)
        layer_importance = {}

        # 使用单一进度条显示所有信息
        pbar = tqdm(range(num_layers), desc="分析层重要性", ncols=100)

        for layer_idx in pbar:
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
                ppl = self.compute_perplexity(texts, show_progress=False)
                importance = ppl - baseline_ppl  # 困惑度增加越多，该层越重要
                layer_importance[layer_idx] = importance

                # 在进度条上显示关键信息
                pbar.set_postfix({
                    'baseline': f'{baseline_ppl:.2f}',
                    'current': f'{ppl:.2f}',
                    'delta': f'+{importance:.2f}'
                })
            finally:
                # 无论是否出错，都要恢复该层
                self.model.model.layers[layer_idx].forward = original_forward

        pbar.close()
        return layer_importance

    def measure_block_importance_by_removal(self, texts: List[str],
                                           num_layers: int) -> Dict[str, Dict[int, float]]:
        """
        通过移除块（Attention或MLP）来评估重要性

        Args:
            texts: 测试文本列表
            num_layers: 层数

        Returns:
            Dict包含两个键:
            - 'attention': {layer_idx: importance, ...}
            - 'mlp': {layer_idx: importance, ...}
        """
        baseline_ppl = self.compute_perplexity(texts, show_progress=False)
        attention_importance = {}
        mlp_importance = {}

        # 评估每层的 Attention 重要性
        pbar = tqdm(range(num_layers), desc="分析 Attention 块", ncols=100)
        for layer_idx in pbar:
            layer = self.model.model.layers[layer_idx]

            # 保存原始 self_attn forward
            original_attn_forward = layer.self_attn.forward

            # 定义恒等映射（跳过 Attention）
            def identity_attn_forward(hidden_states, *args, **kwargs):
                # LlamaAttention.forward 总是返回固定格式：(attn_output, attn_weights)
                # 我们的恒等映射也返回相同格式
                return (hidden_states, None)

            # 替换 Attention forward
            layer.self_attn.forward = identity_attn_forward

            try:
                ppl = self.compute_perplexity(texts, show_progress=False)
                importance = ppl - baseline_ppl
                attention_importance[layer_idx] = importance

                # 在进度条上显示关键信息
                pbar.set_postfix({
                    'baseline': f'{baseline_ppl:.2f}',
                    'current': f'{ppl:.2f}',
                    'delta': f'+{importance:.2f}'
                })
            finally:
                layer.self_attn.forward = original_attn_forward

        pbar.close()

        # 评估每层的 MLP 重要性
        pbar = tqdm(range(num_layers), desc="分析 MLP 块", ncols=100)
        for layer_idx in pbar:
            layer = self.model.model.layers[layer_idx]

            # 保存原始 mlp forward
            original_mlp_forward = layer.mlp.forward

            # 定义恒等映射（跳过 MLP）
            def identity_mlp_forward(hidden_states, *args, **kwargs):
                return hidden_states

            # 替换 MLP forward
            layer.mlp.forward = identity_mlp_forward

            try:
                ppl = self.compute_perplexity(texts, show_progress=False)
                importance = ppl - baseline_ppl
                mlp_importance[layer_idx] = importance

                # 在进度条上显示关键信息
                pbar.set_postfix({
                    'baseline': f'{baseline_ppl:.2f}',
                    'current': f'{ppl:.2f}',
                    'delta': f'+{importance:.2f}'
                })
            finally:
                layer.mlp.forward = original_mlp_forward

        pbar.close()

        return {
            'attention': attention_importance,
            'mlp': mlp_importance
        }

    def measure_layer_importance_by_similarity(self, texts: List[str], num_layers: int = None) -> Dict[int, float]:
        """
        通过层输出相似度评估重要性（ShortGPT方法）

        核心思想：
        - 计算跳过某层后的输出与正常输出的相似度
        - 相似度越高，说明该层的变换越小，重要性越低
        - 重要性 = 1 - 余弦相似度

        优点：
        - 不需要移除层，避免兼容性问题
        - 计算高效，对所有模型通用
        - 无需重新计算PPL

        Args:
            texts: 测试文本列表
            num_layers: 层数（默认自动获取）

        Returns:
            Dict[int, float]: {layer_idx: importance, ...}
            importance越大表示该层越重要
        """
        if num_layers is None:
            num_layers = len(self.model.model.layers)

        layer_importance = {}

        print(f"\n{'='*60}")
        print(f"基于相似度的层重要性分析 (ShortGPT方法)")
        print(f"{'='*60}")
        print(f"方法: 计算跳过层后的输出相似度")
        print(f"指标: 1 - 余弦相似度")
        print(f"样本数: {len(texts)}")
        print(f"{'='*60}\n")

        pbar = tqdm(range(num_layers), desc="分析层重要性", ncols=100)

        for layer_idx in pbar:
            similarities = []

            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    first_device = next(self.model.parameters()).device
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}

                    # 1. 正常前向传播，收集该层的输入和输出
                    layer_outputs = {}

                    def hook_fn(module, input, output):
                        if isinstance(input, tuple):
                            layer_outputs['input'] = input[0].clone()
                        else:
                            layer_outputs['input'] = input.clone()

                        if isinstance(output, tuple):
                            layer_outputs['output'] = output[0].clone()
                        else:
                            layer_outputs['output'] = output.clone()

                    # 注册hook
                    hook = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)

                    try:
                        # 正常前向传播
                        _ = self.model(**inputs)

                        # 获取层的输入和输出
                        if 'input' in layer_outputs and 'output' in layer_outputs:
                            layer_input = layer_outputs['input']
                            layer_output = layer_outputs['output']

                            # 2. 计算相似度：如果跳过该层（输入直接作为输出），与实际输出的相似度
                            # 使用余弦相似度
                            cos_sim = torch.nn.functional.cosine_similarity(
                                layer_input.view(-1),
                                layer_output.view(-1),
                                dim=0
                            ).item()

                            similarities.append(cos_sim)
                    finally:
                        hook.remove()

            # 计算平均相似度
            if similarities:
                avg_similarity = np.mean(similarities)
                # 重要性 = 1 - 相似度
                # 相似度越高（层变换越小），重要性越低
                importance = 1.0 - avg_similarity
                layer_importance[layer_idx] = importance

                pbar.set_postfix({
                    'sim': f'{avg_similarity:.4f}',
                    'imp': f'{importance:.4f}'
                })
            else:
                layer_importance[layer_idx] = 0.0

        pbar.close()

        # 打印统计信息
        print(f"\n层重要性统计:")
        print(f"  最高重要性: {max(layer_importance.values()):.4f} (层 {max(layer_importance, key=layer_importance.get)})")
        print(f"  最低重要性: {min(layer_importance.values()):.4f} (层 {min(layer_importance, key=layer_importance.get)})")
        print(f"  平均重要性: {np.mean(list(layer_importance.values())):.4f}")

        return layer_importance

    def measure_block_importance_by_similarity(self, texts: List[str], num_layers: int = None) -> Dict[str, Dict[int, float]]:
        """
        通过块输出相似度评估Attention和MLP的重要性（ShortGPT方法）

        核心思想：
        - 分别计算跳过Attention块和MLP块后的输出相似度
        - 相似度越高，说明该块的变换越小，重要性越低
        - 重要性 = 1 - 余弦相似度

        Args:
            texts: 测试文本列表
            num_layers: 层数（默认自动获取）

        Returns:
            Dict包含两个键:
            - 'attention': {layer_idx: importance, ...}
            - 'mlp': {layer_idx: importance, ...}
        """
        if num_layers is None:
            num_layers = len(self.model.model.layers)

        attention_importance = {}
        mlp_importance = {}

        print(f"\n{'='*60}")
        print(f"基于相似度的块重要性分析 (ShortGPT方法)")
        print(f"{'='*60}")
        print(f"方法: 计算跳过块后的输出相似度")
        print(f"指标: 1 - 余弦相似度")
        print(f"样本数: {len(texts)}")
        print(f"{'='*60}\n")

        # 评估 Attention 块重要性
        pbar = tqdm(range(num_layers), desc="分析 Attention 块", ncols=100)
        for layer_idx in pbar:
            similarities = []
            layer = self.model.model.layers[layer_idx]

            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    first_device = next(self.model.parameters()).device
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}

                    # 收集 Attention 块的输入和输出
                    attn_outputs = {}

                    def hook_fn(module, input, output):
                        if isinstance(input, tuple):
                            attn_outputs['input'] = input[0].clone()
                        else:
                            attn_outputs['input'] = input.clone()

                        if isinstance(output, tuple):
                            attn_outputs['output'] = output[0].clone()
                        else:
                            attn_outputs['output'] = output.clone()

                    hook = layer.self_attn.register_forward_hook(hook_fn)

                    try:
                        _ = self.model(**inputs)

                        if 'input' in attn_outputs and 'output' in attn_outputs:
                            attn_input = attn_outputs['input']
                            attn_output = attn_outputs['output']

                            cos_sim = torch.nn.functional.cosine_similarity(
                                attn_input.view(-1),
                                attn_output.view(-1),
                                dim=0
                            ).item()

                            similarities.append(cos_sim)
                    finally:
                        hook.remove()

            if similarities:
                avg_similarity = np.mean(similarities)
                importance = 1.0 - avg_similarity
                attention_importance[layer_idx] = importance
                pbar.set_postfix({'sim': f'{avg_similarity:.4f}', 'imp': f'{importance:.4f}'})
            else:
                attention_importance[layer_idx] = 0.0

        pbar.close()

        # 评估 MLP 块重要性
        pbar = tqdm(range(num_layers), desc="分析 MLP 块", ncols=100)
        for layer_idx in pbar:
            similarities = []
            layer = self.model.model.layers[layer_idx]

            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt",
                                           truncation=True, max_length=512)
                    first_device = next(self.model.parameters()).device
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}

                    # 收集 MLP 块的输入和输出
                    mlp_outputs = {}

                    def hook_fn(module, input, output):
                        if isinstance(input, tuple):
                            mlp_outputs['input'] = input[0].clone()
                        else:
                            mlp_outputs['input'] = input.clone()

                        mlp_outputs['output'] = output.clone()

                    hook = layer.mlp.register_forward_hook(hook_fn)

                    try:
                        _ = self.model(**inputs)

                        if 'input' in mlp_outputs and 'output' in mlp_outputs:
                            mlp_input = mlp_outputs['input']
                            mlp_output = mlp_outputs['output']

                            cos_sim = torch.nn.functional.cosine_similarity(
                                mlp_input.view(-1),
                                mlp_output.view(-1),
                                dim=0
                            ).item()

                            similarities.append(cos_sim)
                    finally:
                        hook.remove()

            if similarities:
                avg_similarity = np.mean(similarities)
                importance = 1.0 - avg_similarity
                mlp_importance[layer_idx] = importance
                pbar.set_postfix({'sim': f'{avg_similarity:.4f}', 'imp': f'{importance:.4f}'})
            else:
                mlp_importance[layer_idx] = 0.0

        pbar.close()

        # 打印统计信息
        print(f"\nAttention 块重要性统计:")
        print(f"  最高重要性: {max(attention_importance.values()):.4f} (层 {max(attention_importance, key=attention_importance.get)})")
        print(f"  最低重要性: {min(attention_importance.values()):.4f} (层 {min(attention_importance, key=attention_importance.get)})")
        print(f"  平均重要性: {np.mean(list(attention_importance.values())):.4f}")

        print(f"\nMLP 块重要性统计:")
        print(f"  最高重要性: {max(mlp_importance.values()):.4f} (层 {max(mlp_importance, key=mlp_importance.get)})")
        print(f"  最低重要性: {min(mlp_importance.values()):.4f} (层 {min(mlp_importance, key=mlp_importance.get)})")
        print(f"  平均重要性: {np.mean(list(mlp_importance.values())):.4f}")

        return {
            'attention': attention_importance,
            'mlp': mlp_importance
        }

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
    try:
        from core.utils.get_best_gpu import get_best_gpu
        device = f"cuda:{get_best_gpu()}"
    except:
        device = "cuda:0"

    model = LlamaForCausalLM.from_pretrained(
        "/newdata/LLMs/Llama-3-8B-Instruct",
        device_map=device,
        torch_dtype=torch.float16,
    )
    model.half()
    tokenizer = AutoTokenizer.from_pretrained("/newdata/LLMs/Llama-3-8B-Instruct")
    analyzer = LayerImportanceAnalyzer(model, tokenizer)

    all_samples = get_examples('wikitext', tokenizer, num_samples=512, seq_len=512, split='test')
    eval_samples = all_samples[:50]
    eval_texts = [tokenizer.decode(sample, skip_special_tokens=True) for sample in eval_samples]

    # 层级重要度评估
    print("\n" + "="*60)
    print("层级重要度评估")
    print("="*60)
    layer_importance = analyzer.measure_layer_importance_by_removal(
        eval_texts, num_layers = 32
    )

    # 统一展示层级重要度结果
    print("\n" + "="*60)
    print("层级重要度汇总")
    print("="*60)
    print(f"\n{'Layer':<10} {'PPL 增加':<12}")
    print("-" * 30)

    for layer_idx in range(32):
        importance = layer_importance.get(layer_idx, 0.0)
        print(f"Layer {layer_idx:2d}   {importance:10.4f}")

    # 计算统计信息
    print("\n" + "-" * 30)
    layer_avg = sum(layer_importance.values()) / len(layer_importance)
    layer_total = sum(layer_importance.values())
    print(f"{'平均值':<10} {layer_avg:10.4f}")
    print(f"{'总和':<10} {layer_total:10.4f}")
    print("="*60)

    # 块级重要度评估
    print("\n" + "="*60)
    print("块级重要度评估")
    print("="*60)
    block_importance = analyzer.measure_block_importance_by_removal(
        eval_texts, num_layers = 32
    )

    # 统一展示结果
    print("\n" + "="*60)
    print("块级重要度汇总")
    print("="*60)
    print(f"\n{'Layer':<7} {'Attention':<12} {'MLP':<12} {'总和':<12}")
    print("-" * 60)

    for layer_idx in range(32):
        attn_imp = block_importance['attention'].get(layer_idx, 0.0)
        mlp_imp = block_importance['mlp'].get(layer_idx, 0.0)
        total_imp = attn_imp + mlp_imp
        print(f"Layer {layer_idx:2d}  {attn_imp:10.4f}   {mlp_imp:10.4f}   {total_imp:10.4f}")

    # 计算统计信息
    print("\n" + "-" * 60)
    attn_total = sum(block_importance['attention'].values())
    mlp_total = sum(block_importance['mlp'].values())
    attn_avg = attn_total / len(block_importance['attention'])
    mlp_avg = mlp_total / len(block_importance['mlp'])

    print(f"{'平均值':<7} {attn_avg:10.4f}   {mlp_avg:10.4f}   {(attn_avg + mlp_avg):10.4f}")
    print(f"{'总和':<7} {attn_total:10.4f}   {mlp_total:10.4f}   {(attn_total + mlp_total):10.4f}")
    print("="*60)
    
