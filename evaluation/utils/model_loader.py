#!/usr/bin/env python3
"""
模型加载和信息提取工具

支持加载：
1. 原始HuggingFace模型
2. 剪枝后的checkpoint（pytorch_model.bin）
3. 微调后的模型
4. SliceGPT 模型（.pt + .json）
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Dict, Optional
import gc
import pathlib
import json

# 导入 IdentityDecoderLayer 以支持加载包含该层的剪枝模型
# 这个导入必须在 torch.load() 之前，否则 unpickle 会找不到类定义
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.models import IdentityDecoderLayer


def load_slicegpt_model(
    model_path: str,
    device: str,
    torch_dtype: torch.dtype,
    base_model: str = None,
    sparsity: float = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载 SliceGPT 模型 (.pt 格式)

    Args:
        model_path: .pt 文件路径，例如 path/to/Llama-3-8B-Instruct_0.2.pt
        device: 设备
        torch_dtype: 数据类型
        base_model: 基础模型路径（如果未指定，从文件名推断）
        sparsity: 稀疏度（如果未指定，从文件名推断）

    Returns:
        (model, tokenizer)
    """
    print(f"  检测到 SliceGPT 模型格式 (.pt)")

    # 尝试导入 SliceGPT
    try:
        from slicegpt import hf_utils as slicegpt_hf_utils
        from slicegpt.config import config as slicegpt_config
    except ImportError:
        raise ImportError(
            "无法导入 slicegpt 包。请先安装 SliceGPT:\n"
            "  git clone https://github.com/microsoft/TransformerCompression\n"
            "  cd TransformerCompression\n"
            "  pip install -e ."
        )

    # 从文件路径推断信息
    pt_path = pathlib.Path(model_path)
    sliced_model_dir = pt_path.parent

    # 从文件名推断 base_model 和 sparsity
    # 例如: Llama-3-8B-Instruct_0.2.pt -> base_model=Llama-3-8B-Instruct, sparsity=0.2
    if sparsity is None:
        try:
            # 提取 sparsity（假设格式为 <name>_<sparsity>.pt）
            sparsity = float(pt_path.stem.split('_')[-1])
            print(f"  从文件名推断 sparsity = {sparsity}")
        except:
            raise ValueError(
                f"无法从文件名推断 sparsity。请手动指定 --slicegpt-sparsity 参数。\n"
                f"文件名: {pt_path.name}"
            )

    if base_model is None:
        # 尝试从文件名推断
        model_name = pt_path.stem.rsplit('_', 1)[0]  # 移除 _0.2
        print(f"  从文件名推断 base_model = {model_name}")

        # 检查是否有同目录下的 config.json（说明是本地模型）
        if (sliced_model_dir / "config.json").exists():
            base_model = str(sliced_model_dir)
            print(f"  检测到本地模型配置，使用目录: {base_model}")
        else:
            base_model = model_name
            print(f"  将尝试从 HuggingFace 加载: {base_model}")

    # 设置 SliceGPT 的 device 和 dtype
    slicegpt_config.device = torch.device(device)
    slicegpt_config.dtype = torch_dtype

    print(f"  使用 SliceGPT 加载器...")
    print(f"    - Base model: {base_model}")
    print(f"    - Sparsity: {sparsity}")
    print(f"    - Model path: {sliced_model_dir}")

    # 使用 SliceGPT 的加载函数
    model_adapter, tokenizer = slicegpt_hf_utils.load_sliced_model(
        base_model,
        str(sliced_model_dir),
        sparsity=sparsity,
        round_interval=8,  # 默认值
        token=None
    )

    # 提取原始模型（去掉 adapter）
    model = model_adapter.model
    model.eval()

    # 移动到指定设备
    if device.startswith('cuda'):
        print(f"  移动模型到 {device}...")
        model = model.to(device)

    print(f"✓ SliceGPT 模型加载完成")
    return model, tokenizer


def load_model_and_tokenizer(
    model_path: str,
    device: str = 'cuda',
    load_in_8bit: bool = False,
    torch_dtype: Optional[torch.dtype] = torch.float16,
    force_single_device: bool = True,
    slicegpt_base_model: str = None,
    slicegpt_sparsity: float = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和tokenizer

    Args:
        model_path: 模型路径
            - HuggingFace模型目录: /path/to/Llama-3-8B-Instruct
            - 剪枝checkpoint: prune_log/xxx/pytorch_model.bin
            - SliceGPT模型: path/to/Llama-3-8B-Instruct_0.2.pt
        device: 设备 (cuda/cpu)
        load_in_8bit: 是否使用8bit量化加载
        torch_dtype: 数据类型
        force_single_device: 是否强制使用单个设备（避免multi-GPU问题）
        slicegpt_base_model: SliceGPT 的基础模型路径（仅用于 .pt 模型）
        slicegpt_sparsity: SliceGPT 的稀疏度（仅用于 .pt 模型）

    Returns:
        (model, tokenizer)
    """
    print(f"加载模型: {model_path}")

    # 将device转换为字符串（支持 torch.device 对象和字符串）
    device_str = str(device)

    # 检测是否是 SliceGPT 模型（.pt 文件）
    if model_path.endswith('.pt'):
        return load_slicegpt_model(
            model_path,
            device_str,
            torch_dtype,
            slicegpt_base_model,
            slicegpt_sparsity
        )

    # 判断是checkpoint还是目录
    elif model_path.endswith('.bin'):
        # 剪枝checkpoint - 直接加载到目标设备
        target_device = device_str if device_str.startswith('cuda') and force_single_device else 'cpu'
        print(f"  直接加载checkpoint到 {target_device}...")
        checkpoint = torch.load(model_path, map_location=target_device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # 完整保存格式（包含model, tokenizer, config等）
            model = checkpoint['model']
            tokenizer = checkpoint.get('tokenizer')

            if tokenizer is None:
                # 从配置中获取base_model路径
                config = checkpoint.get('config', {})
                base_model = config.get('base_model')
                if base_model and os.path.exists(base_model):
                    tokenizer = AutoTokenizer.from_pretrained(base_model)
                else:
                    raise ValueError(f"无法找到tokenizer，请提供base_model路径")
        else:
            # 只保存了state_dict的格式
            raise ValueError(f"不支持的checkpoint格式。请确保保存时包含完整的model对象。")

    else:
        # HuggingFace模型目录
        # 决定device_map策略
        if force_single_device:
            # 强制单设备：避免PPL计算时的multi-GPU问题
            device_map = None
            print(f"  使用单设备模式: {device_str}")
        else:
            # 自动分布：适合生成任务
            device_map = 'auto' if device_str == 'cuda' else None
            print(f"  使用自动设备分布模式")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 设置padding token和padding方向
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only模型（如LLaMA）必须使用left padding
    # 原因：生成时需要从右边开始，左边是padding不参与计算
    if hasattr(model.config, 'is_decoder') and model.config.is_decoder:
        tokenizer.padding_side = 'left'
    else:
        # 保险起见，对于Causal LM也设为left
        tokenizer.padding_side = 'left'

    # 移动到设备（仅当没有使用device_map且是HF模型时）
    # 注意：checkpoint已经在上面移动过了
    if not load_in_8bit and device_str.startswith('cuda') and force_single_device and not model_path.endswith('.bin'):
        print(f"  移动HF模型到 {device_str}...")
        model = model.to(device_str)

    model.eval()

    print(f"✓ 模型加载完成")
    return model, tokenizer


def get_model_info(model: torch.nn.Module) -> Dict:
    """
    获取模型的详细信息

    Args:
        model: PyTorch模型

    Returns:
        包含模型信息的字典
    """
    info = {}

    # 1. 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info['total_params'] = total_params
    info['trainable_params'] = trainable_params
    info['total_params_M'] = total_params / 1e6
    info['total_params_B'] = total_params / 1e9

    # 2. Attention和MLP参数（针对LLaMA架构）
    attention_params = 0
    mlp_params = 0

    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer in model.model.layers:
                # Attention参数
                if hasattr(layer, 'self_attn'):
                    attention_params += sum(p.numel() for p in layer.self_attn.parameters())
                # MLP参数
                if hasattr(layer, 'mlp'):
                    mlp_params += sum(p.numel() for p in layer.mlp.parameters())

        info['attention_params'] = attention_params
        info['mlp_params'] = mlp_params
        info['attention_params_M'] = attention_params / 1e6
        info['mlp_params_M'] = mlp_params / 1e6
        info['attention_ratio'] = attention_params / total_params if total_params > 0 else 0
        info['mlp_ratio'] = mlp_params / total_params if total_params > 0 else 0
    except:
        info['attention_params'] = None
        info['mlp_params'] = None

    # 3. 层数
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            info['num_layers'] = len(model.model.layers)
        else:
            info['num_layers'] = None
    except:
        info['num_layers'] = None

    # 4. 模型占用内存（近似）
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024**2
    info['model_size_mb'] = total_size_mb
    info['model_size_gb'] = total_size_mb / 1024

    return info


def print_model_info(info: Dict, name: str = "Model"):
    """
    打印模型信息

    Args:
        info: 模型信息字典
        name: 模型名称
    """
    print(f"\n{'='*60}")
    print(f"{name} 信息:")
    print(f"{'='*60}")
    print(f"总参数量: {info['total_params']:,} ({info['total_params_B']:.2f}B)")

    if info['attention_params'] is not None:
        print(f"Attention参数: {info['attention_params']:,} ({info['attention_params_M']:.1f}M, {info['attention_ratio']*100:.1f}%)")
        print(f"MLP参数: {info['mlp_params']:,} ({info['mlp_params_M']:.1f}M, {info['mlp_ratio']*100:.1f}%)")

    if info['num_layers'] is not None:
        print(f"层数: {info['num_layers']}")

    print(f"模型大小: {info['model_size_mb']:.1f} MB ({info['model_size_gb']:.2f} GB)")
    print(f"{'='*60}\n")


def cleanup_model(model, tokenizer=None):
    """
    清理模型释放显存

    Args:
        model: 模型对象
        tokenizer: tokenizer对象（可选）
    """
    del model
    if tokenizer is not None:
        del tokenizer

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✓ 模型已清理，显存已释放")


if __name__ == '__main__':
    # 测试
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        model, tokenizer = load_model_and_tokenizer(model_path)
        info = get_model_info(model)
        print_model_info(info, name=model_path)
        cleanup_model(model, tokenizer)
    else:
        print("Usage: python model_loader.py <model_path>")
