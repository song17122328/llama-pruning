"""
模型分析模块

功能：
1. 统计模型的总参数量、每层参数量
2. 分析剪枝前后的差异
3. 生成详细的参数分析报告

可以作为独立脚本运行，也可以被其他模块导入使用。
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from collections import OrderedDict

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入模型加载器
from evaluation.utils.model_loader import load_model_and_tokenizer


class ModelAnalyzer:
    """模型参数分析器"""

    def __init__(self, model: nn.Module, model_name: str = "Model"):
        """
        初始化分析器

        Args:
            model: 要分析的模型
            model_name: 模型名称（用于报告）
        """
        self.model = model
        self.model_name = model_name
        self.analysis_result = None

    def analyze(self) -> Dict:
        """
        分析模型的参数结构

        Returns:
            包含详细参数信息的字典
        """
        result = {
            'model_name': self.model_name,
            'total_params': 0,
            'embedding_params': 0,
            'lm_head_params': 0,
            'layers': [],
            'layer_summary': {
                'num_layers': 0,
                'total_layer_params': 0
            }
        }

        # 统计总参数量
        result['total_params'] = sum(p.numel() for p in self.model.parameters())

        # 尝试获取 Llama 模型结构
        if hasattr(self.model, 'model'):
            # 这是一个 LlamaForCausalLM
            base_model = self.model.model

            # 统计 embedding 参数
            if hasattr(base_model, 'embed_tokens'):
                result['embedding_params'] = sum(
                    p.numel() for p in base_model.embed_tokens.parameters()
                )

            # 统计 lm_head 参数
            if hasattr(self.model, 'lm_head'):
                result['lm_head_params'] = sum(
                    p.numel() for p in self.model.lm_head.parameters()
                )

            # 统计每一层的参数
            if hasattr(base_model, 'layers'):
                layers = base_model.layers
                result['layer_summary']['num_layers'] = len(layers)

                for layer_idx, layer in enumerate(layers):
                    layer_info = self._analyze_layer(layer_idx, layer)
                    result['layers'].append(layer_info)
                    result['layer_summary']['total_layer_params'] += layer_info['total']
        else:
            # 直接是 LlamaModel
            if hasattr(self.model, 'embed_tokens'):
                result['embedding_params'] = sum(
                    p.numel() for p in self.model.embed_tokens.parameters()
                )

            if hasattr(self.model, 'layers'):
                layers = self.model.layers
                result['layer_summary']['num_layers'] = len(layers)

                for layer_idx, layer in enumerate(layers):
                    layer_info = self._analyze_layer(layer_idx, layer)
                    result['layers'].append(layer_info)
                    result['layer_summary']['total_layer_params'] += layer_info['total']

        self.analysis_result = result
        return result

    def _analyze_layer(self, layer_idx: int, layer: nn.Module) -> Dict:
        """
        分析单个 decoder layer 的参数

        Args:
            layer_idx: 层索引
            layer: LlamaDecoderLayer 实例

        Returns:
            该层的参数信息
        """
        layer_info = {
            'layer_idx': layer_idx,
            'total': 0,
            'attention': {},
            'mlp': {},
            'norm': 0,
            'is_zero_layer': False
        }

        # 检查是否是 ZeroAttention 或 ZeroMLP
        from core.models import ZeroAttention, ZeroMLP

        # 分析 Attention
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn

            if isinstance(attn, ZeroAttention):
                layer_info['attention'] = {
                    'type': 'ZeroAttention',
                    'total': 0,
                    'q_proj': 0,
                    'k_proj': 0,
                    'v_proj': 0,
                    'o_proj': 0,
                    'num_heads': 0,
                    'num_kv_heads': 0
                }
            else:
                # 正常的 LlamaAttention
                attn_params = {
                    'type': 'LlamaAttention',
                    'q_proj': 0,
                    'k_proj': 0,
                    'v_proj': 0,
                    'o_proj': 0
                }

                if hasattr(attn, 'q_proj'):
                    attn_params['q_proj'] = sum(p.numel() for p in attn.q_proj.parameters())
                if hasattr(attn, 'k_proj'):
                    attn_params['k_proj'] = sum(p.numel() for p in attn.k_proj.parameters())
                if hasattr(attn, 'v_proj'):
                    attn_params['v_proj'] = sum(p.numel() for p in attn.v_proj.parameters())
                if hasattr(attn, 'o_proj'):
                    attn_params['o_proj'] = sum(p.numel() for p in attn.o_proj.parameters())

                attn_params['total'] = sum(attn_params[k] for k in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

                # 记录头数信息
                if hasattr(attn, 'num_heads'):
                    attn_params['num_heads'] = attn.num_heads
                if hasattr(attn, 'num_key_value_heads'):
                    attn_params['num_kv_heads'] = attn.num_key_value_heads

                layer_info['attention'] = attn_params

        # 分析 MLP
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp

            if isinstance(mlp, ZeroMLP):
                layer_info['mlp'] = {
                    'type': 'ZeroMLP',
                    'total': 0,
                    'gate_proj': 0,
                    'up_proj': 0,
                    'down_proj': 0,
                    'intermediate_size': 0
                }
            else:
                # 正常的 LlamaMLP
                mlp_params = {
                    'type': 'LlamaMLP',
                    'gate_proj': 0,
                    'up_proj': 0,
                    'down_proj': 0
                }

                if hasattr(mlp, 'gate_proj'):
                    mlp_params['gate_proj'] = sum(p.numel() for p in mlp.gate_proj.parameters())
                if hasattr(mlp, 'up_proj'):
                    mlp_params['up_proj'] = sum(p.numel() for p in mlp.up_proj.parameters())
                if hasattr(mlp, 'down_proj'):
                    mlp_params['down_proj'] = sum(p.numel() for p in mlp.down_proj.parameters())

                mlp_params['total'] = sum(mlp_params[k] for k in ['gate_proj', 'up_proj', 'down_proj'])

                # 记录中间维度
                if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'out_features'):
                    mlp_params['intermediate_size'] = mlp.gate_proj.out_features

                layer_info['mlp'] = mlp_params

        # 分析 LayerNorm
        if hasattr(layer, 'input_layernorm'):
            layer_info['norm'] += sum(p.numel() for p in layer.input_layernorm.parameters())
        if hasattr(layer, 'post_attention_layernorm'):
            layer_info['norm'] += sum(p.numel() for p in layer.post_attention_layernorm.parameters())

        # 计算该层总参数
        layer_info['total'] = (
            layer_info['attention'].get('total', 0) +
            layer_info['mlp'].get('total', 0) +
            layer_info['norm']
        )

        # 检查是否是完全为零的层
        if (layer_info['attention'].get('type') == 'ZeroAttention' and
            layer_info['mlp'].get('type') == 'ZeroMLP'):
            layer_info['is_zero_layer'] = True

        return layer_info

    def print_report(self, verbose: bool = True):
        """
        打印分析报告

        Args:
            verbose: 是否打印详细信息（每层的参数）
        """
        if self.analysis_result is None:
            self.analyze()

        result = self.analysis_result

        print(f"\n{'='*80}")
        print(f"模型参数分析报告: {result['model_name']}")
        print(f"{'='*80}")

        print(f"\n总参数量: {result['total_params']:,}")
        print(f"  - Embedding 参数: {result['embedding_params']:,}")
        print(f"  - LM Head 参数: {result['lm_head_params']:,}")
        print(f"  - Decoder Layers 参数: {result['layer_summary']['total_layer_params']:,}")
        print(f"  - 层数: {result['layer_summary']['num_layers']}")

        if verbose and result['layers']:
            print(f"\n{'-'*80}")
            print("每层参数详情:")
            print(f"{'-'*80}")

            for layer_info in result['layers']:
                layer_idx = layer_info['layer_idx']
                total = layer_info['total']
                attn_total = layer_info['attention'].get('total', 0)
                mlp_total = layer_info['mlp'].get('total', 0)
                norm_total = layer_info['norm']

                # 标记特殊层
                special_marker = ""
                if layer_info['is_zero_layer']:
                    special_marker = " [完全剪空]"
                elif layer_info['attention'].get('type') == 'ZeroAttention':
                    special_marker = " [Attention剪空]"
                elif layer_info['mlp'].get('type') == 'ZeroMLP':
                    special_marker = " [MLP剪空]"

                print(f"\nLayer {layer_idx:2d}{special_marker}:")
                print(f"  总参数: {total:,}")
                print(f"  - Attention: {attn_total:,}")

                if layer_info['attention'].get('type') == 'ZeroAttention':
                    print(f"    类型: ZeroAttention (无参数)")
                else:
                    attn = layer_info['attention']
                    if 'num_heads' in attn and 'num_kv_heads' in attn:
                        print(f"    头数: {attn.get('num_heads', 0)}Q : {attn.get('num_kv_heads', 0)}KV")
                    print(f"    q_proj: {attn.get('q_proj', 0):,}")
                    print(f"    k_proj: {attn.get('k_proj', 0):,}")
                    print(f"    v_proj: {attn.get('v_proj', 0):,}")
                    print(f"    o_proj: {attn.get('o_proj', 0):,}")

                print(f"  - MLP: {mlp_total:,}")

                if layer_info['mlp'].get('type') == 'ZeroMLP':
                    print(f"    类型: ZeroMLP (无参数)")
                else:
                    mlp = layer_info['mlp']
                    if 'intermediate_size' in mlp:
                        print(f"    中间维度: {mlp.get('intermediate_size', 0)}")
                    print(f"    gate_proj: {mlp.get('gate_proj', 0):,}")
                    print(f"    up_proj: {mlp.get('up_proj', 0):,}")
                    print(f"    down_proj: {mlp.get('down_proj', 0):,}")

                print(f"  - LayerNorm: {norm_total:,}")

        print(f"\n{'='*80}\n")

    def save_report(self, save_path: str):
        """
        保存分析报告为 JSON 文件

        Args:
            save_path: 保存路径
        """
        if self.analysis_result is None:
            self.analyze()

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, indent=2, ensure_ascii=False)

        print(f"✓ 分析报告已保存至: {save_path}")


class ModelComparator:
    """模型对比分析器（对比剪枝前后的差异）"""

    def __init__(
        self,
        original_analysis: Dict,
        pruned_analysis: Dict,
        original_name: str = "原始模型",
        pruned_name: str = "剪枝后模型"
    ):
        """
        初始化对比分析器

        Args:
            original_analysis: 原始模型的分析结果
            pruned_analysis: 剪枝后模型的分析结果
            original_name: 原始模型名称
            pruned_name: 剪枝后模型名称
        """
        self.original = original_analysis
        self.pruned = pruned_analysis
        self.original_name = original_name
        self.pruned_name = pruned_name
        self.comparison_result = None

    def compare(self) -> Dict:
        """
        对比两个模型的差异

        Returns:
            对比结果字典
        """
        result = {
            'original_name': self.original_name,
            'pruned_name': self.pruned_name,
            'total_params': {
                'original': self.original['total_params'],
                'pruned': self.pruned['total_params'],
                'reduced': self.original['total_params'] - self.pruned['total_params'],
                'reduction_ratio': 0.0
            },
            'layer_params': {
                'original': self.original['layer_summary']['total_layer_params'],
                'pruned': self.pruned['layer_summary']['total_layer_params'],
                'reduced': 0,
                'reduction_ratio': 0.0
            },
            'layers': []
        }

        # 计算总参数的剪枝比例
        if self.original['total_params'] > 0:
            result['total_params']['reduction_ratio'] = (
                result['total_params']['reduced'] / self.original['total_params']
            )

        # 计算层参数的剪枝比例
        result['layer_params']['reduced'] = (
            self.original['layer_summary']['total_layer_params'] -
            self.pruned['layer_summary']['total_layer_params']
        )
        if self.original['layer_summary']['total_layer_params'] > 0:
            result['layer_params']['reduction_ratio'] = (
                result['layer_params']['reduced'] /
                self.original['layer_summary']['total_layer_params']
            )

        # 对比每一层
        num_layers = min(
            len(self.original['layers']),
            len(self.pruned['layers'])
        )

        for i in range(num_layers):
            orig_layer = self.original['layers'][i]
            pruned_layer = self.pruned['layers'][i]

            layer_comp = self._compare_layer(orig_layer, pruned_layer)
            result['layers'].append(layer_comp)

        self.comparison_result = result
        return result

    def _compare_layer(self, original: Dict, pruned: Dict) -> Dict:
        """
        对比单个层的差异

        Args:
            original: 原始层的信息
            pruned: 剪枝后层的信息

        Returns:
            该层的对比信息
        """
        comp = {
            'layer_idx': original['layer_idx'],
            'total': {
                'original': original['total'],
                'pruned': pruned['total'],
                'reduced': original['total'] - pruned['total'],
                'reduction_ratio': 0.0
            },
            'attention': {
                'original': original['attention'].get('total', 0),
                'pruned': pruned['attention'].get('total', 0),
                'reduced': 0,
                'reduction_ratio': 0.0
            },
            'mlp': {
                'original': original['mlp'].get('total', 0),
                'pruned': pruned['mlp'].get('total', 0),
                'reduced': 0,
                'reduction_ratio': 0.0
            },
            'is_zero_layer': pruned['is_zero_layer']
        }

        # 计算总参数减少比例
        if original['total'] > 0:
            comp['total']['reduction_ratio'] = comp['total']['reduced'] / original['total']

        # 计算 Attention 减少比例
        comp['attention']['reduced'] = (
            original['attention'].get('total', 0) -
            pruned['attention'].get('total', 0)
        )
        if original['attention'].get('total', 0) > 0:
            comp['attention']['reduction_ratio'] = (
                comp['attention']['reduced'] / original['attention'].get('total', 0)
            )

        # 计算 MLP 减少比例
        comp['mlp']['reduced'] = (
            original['mlp'].get('total', 0) -
            pruned['mlp'].get('total', 0)
        )
        if original['mlp'].get('total', 0) > 0:
            comp['mlp']['reduction_ratio'] = (
                comp['mlp']['reduced'] / original['mlp'].get('total', 0)
            )

        # 记录头数和维度的变化
        if 'num_heads' in original['attention'] and 'num_heads' in pruned['attention']:
            comp['attention']['num_heads'] = {
                'original': original['attention'].get('num_heads', 0),
                'pruned': pruned['attention'].get('num_heads', 0)
            }
            comp['attention']['num_kv_heads'] = {
                'original': original['attention'].get('num_kv_heads', 0),
                'pruned': pruned['attention'].get('num_kv_heads', 0)
            }

        if 'intermediate_size' in original['mlp'] and 'intermediate_size' in pruned['mlp']:
            comp['mlp']['intermediate_size'] = {
                'original': original['mlp'].get('intermediate_size', 0),
                'pruned': pruned['mlp'].get('intermediate_size', 0)
            }

        return comp

    def print_report(self, verbose: bool = True):
        """
        打印对比报告

        Args:
            verbose: 是否打印详细信息（每层的对比）
        """
        if self.comparison_result is None:
            self.compare()

        result = self.comparison_result

        print(f"\n{'='*80}")
        print(f"模型对比分析报告")
        print(f"{'='*80}")
        print(f"原始模型: {result['original_name']}")
        print(f"剪枝模型: {result['pruned_name']}")

        # 总参数对比
        total = result['total_params']
        print(f"\n总参数量:")
        print(f"  原始: {total['original']:,}")
        print(f"  剪枝后: {total['pruned']:,}")
        print(f"  减少: {total['reduced']:,} ({total['reduction_ratio']*100:.2f}%)")

        # Decoder Layers 参数对比
        layer_params = result['layer_params']
        print(f"\nDecoder Layers 参数:")
        print(f"  原始: {layer_params['original']:,}")
        print(f"  剪枝后: {layer_params['pruned']:,}")
        print(f"  减少: {layer_params['reduced']:,} ({layer_params['reduction_ratio']*100:.2f}%)")

        if verbose and result['layers']:
            print(f"\n{'-'*80}")
            print("每层参数对比:")
            print(f"{'-'*80}")

            for layer_comp in result['layers']:
                layer_idx = layer_comp['layer_idx']
                total_comp = layer_comp['total']
                attn_comp = layer_comp['attention']
                mlp_comp = layer_comp['mlp']

                # 标记特殊层
                special_marker = ""
                if layer_comp['is_zero_layer']:
                    special_marker = " [完全剪空]"

                print(f"\nLayer {layer_idx:2d}{special_marker}:")
                print(f"  总参数: {total_comp['original']:,} → {total_comp['pruned']:,} "
                      f"(-{total_comp['reduction_ratio']*100:.2f}%)")

                print(f"  Attention: {attn_comp['original']:,} → {attn_comp['pruned']:,} "
                      f"(-{attn_comp['reduction_ratio']*100:.2f}%)")
                if 'num_heads' in attn_comp:
                    orig_q = attn_comp['num_heads']['original']
                    pruned_q = attn_comp['num_heads']['pruned']
                    orig_kv = attn_comp['num_kv_heads']['original']
                    pruned_kv = attn_comp['num_kv_heads']['pruned']
                    print(f"    头数: {orig_q}Q:{orig_kv}KV → {pruned_q}Q:{pruned_kv}KV")

                print(f"  MLP: {mlp_comp['original']:,} → {mlp_comp['pruned']:,} "
                      f"(-{mlp_comp['reduction_ratio']*100:.2f}%)")
                if 'intermediate_size' in mlp_comp:
                    orig_size = mlp_comp['intermediate_size']['original']
                    pruned_size = mlp_comp['intermediate_size']['pruned']
                    print(f"    中间维度: {orig_size} → {pruned_size}")

        # 统计完全剪空的层
        zero_layers = [l['layer_idx'] for l in result['layers'] if l['is_zero_layer']]
        if zero_layers:
            print(f"\n完全剪空的层 ({len(zero_layers)}个): {zero_layers}")

        print(f"\n{'='*80}\n")

    def save_report(self, save_path: str):
        """
        保存对比报告为 JSON 文件

        Args:
            save_path: 保存路径
        """
        if self.comparison_result is None:
            self.compare()

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_result, f, indent=2, ensure_ascii=False)

        print(f"✓ 对比报告已保存至: {save_path}")


def analyze_model_from_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict:
    """
    从检查点文件加载模型并分析

    支持两种格式：
    1. HuggingFace 模型目录
    2. .bin 格式的剪枝模型

    Args:
        checkpoint_path: 模型路径（HF目录或.bin文件）
        device: 设备（cuda/cpu）
        verbose: 是否打印详细报告

    Returns:
        分析结果字典
    """
    # 使用路径名作为模型名称
    model_name = os.path.basename(checkpoint_path)
    if model_name.endswith('.bin'):
        # 如果是 .bin 文件，使用父目录名
        model_name = os.path.basename(os.path.dirname(checkpoint_path))

    print(f"正在加载模型: {checkpoint_path}")
    print(f"模型名称: {model_name}")

    # 使用统一的加载器，支持 HF 模型和 .bin checkpoint
    model, tokenizer = load_model_and_tokenizer(
        model_path=checkpoint_path,
        device=device,
        torch_dtype=torch.float16,
        force_single_device=True
    )

    print(f"✓ 模型加载完成")

    analyzer = ModelAnalyzer(model, model_name)
    result = analyzer.analyze()

    if verbose:
        analyzer.print_report(verbose=True)

    return result


def main():
    """
    独立运行示例

    使用方法:
        python core/analysis/model_analysis.py --model_path <path>
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="模型参数分析工具 - 支持 HuggingFace 模型和 .bin 格式的剪枝模型"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help="模型路径（HuggingFace 模型目录或 .bin 文件）"
    )
    from core.utils.get_best_gpu import get_best_gpu
    bestDevice = "cuda:"+str(get_best_gpu())
    parser.add_argument(
        '--device',
        type=str,
        default=bestDevice,
        help="设备 (cuda/cpu，默认: cuda)"
    )
    parser.add_argument(
        '--save_json',
        type=str,
        default=None,
        help="保存 JSON 报告的路径（可选）"
    )
    parser.add_argument(
        '--compare_with',
        type=str,
        default=None,
        help="对比的模型路径（可选，用于对比剪枝前后）"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help="显示详细信息（每层参数）"
    )

    args = parser.parse_args()

    # 分析主模型
    print(f"\n{'='*80}")
    print(f"开始分析模型...")
    print(f"{'='*80}\n")

    result = analyze_model_from_checkpoint(
        checkpoint_path=args.model_path,
        device=args.device,
        verbose=args.verbose
    )

    # 获取模型名称（从路径）
    model_name = os.path.basename(args.model_path)
    if model_name.endswith('.bin'):
        model_name = os.path.basename(os.path.dirname(args.model_path))

    # 保存 JSON 报告
    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✓ 分析报告已保存至: {args.save_json}")

    # 如果指定了对比模型
    if args.compare_with:
        print(f"\n{'='*80}")
        print(f"开始分析对比模型...")
        print(f"{'='*80}\n")

        compare_result = analyze_model_from_checkpoint(
            checkpoint_path=args.compare_with,
            device=args.device,
            verbose=args.verbose
        )

        # 获取对比模型名称（从路径）
        compare_name = os.path.basename(args.compare_with)
        if compare_name.endswith('.bin'):
            compare_name = os.path.basename(os.path.dirname(args.compare_with))

        # 生成对比报告
        comparator = ModelComparator(
            original_analysis=result,
            pruned_analysis=compare_result,
            original_name=model_name,
            pruned_name=compare_name
        )

        comparator.print_report(verbose=args.verbose)

        # 保存对比报告
        if args.save_json:
            compare_json_path = args.save_json.replace('.json', '_comparison.json')
            comparator.save_report(compare_json_path)


if __name__ == '__main__':
    # 使用示例
    print("""
使用示例:

1. 分析 HuggingFace 模型:
   python core/analysis/model_analysis.py \\
       --model_path meta-llama/Llama-2-7b-hf \\
       --save_json llama2_analysis.json

2. 分析剪枝后的 .bin 模型:
   python core/analysis/model_analysis.py \\
       --model_path results/my_pruning/models/pruned_model.bin \\
       --save_json pruned_analysis.json

3. 对比两个模型（剪枝前后）:
   python core/analysis/model_analysis.py \\
       --model_path meta-llama/Llama-2-7b-hf \\
       --compare_with results/my_pruning/models/pruned_model.bin \\
       --save_json comparison.json

4. 指定设备:
   python core/analysis/model_analysis.py \\
       --model_path /path/to/model \\
       --device cuda:0 \\
       --save_json analysis.json

5. 在代码中使用:
   from core.analysis.model_analysis import ModelAnalyzer, ModelComparator

   # 分析单个模型
   analyzer = ModelAnalyzer(model, "我的模型")
   result = analyzer.analyze()
   analyzer.print_report(verbose=True)

   # 对比两个模型
   comparator = ModelComparator(original_analysis, pruned_analysis)
   comparator.print_report(verbose=True)

注意：
- 模型名称会自动从路径中提取
- 支持 HuggingFace 模型目录和 .bin 格式的剪枝模型
- .bin 模型会使用父目录名作为模型名称
""")

    main()
