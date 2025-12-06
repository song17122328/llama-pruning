#!/usr/bin/env python3
"""
检查 Qwen2.5 模型中哪些层有 bias 参数
"""
import torch
from transformers import AutoModelForCausalLM

print("加载 Qwen2.5-7B 模型...")
model = AutoModelForCausalLM.from_pretrained(
    "/newdata/LLMs/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map='cpu',  # 使用 CPU 避免显存占用
    low_cpu_mem_usage=True
)

print("\n检查第一层的结构...")
layer = model.model.layers[0]

print("\n=== Attention 层 ===")
print(f"q_proj.bias: {layer.self_attn.q_proj.bias is not None}")
print(f"k_proj.bias: {layer.self_attn.k_proj.bias is not None}")
print(f"v_proj.bias: {layer.self_attn.v_proj.bias is not None}")
print(f"o_proj.bias: {layer.self_attn.o_proj.bias is not None}")

print("\n=== MLP 层 ===")
print(f"gate_proj.bias: {layer.mlp.gate_proj.bias is not None}")
print(f"up_proj.bias: {layer.mlp.up_proj.bias is not None}")
print(f"down_proj.bias: {layer.mlp.down_proj.bias is not None}")

print("\n=== Layer Norm ===")
print(f"input_layernorm.bias: {hasattr(layer.input_layernorm, 'bias') and layer.input_layernorm.bias is not None}")
print(f"post_attention_layernorm.bias: {hasattr(layer.post_attention_layernorm, 'bias') and layer.post_attention_layernorm.bias is not None}")

print("\n详细信息:")
if layer.self_attn.q_proj.bias is not None:
    print(f"q_proj.bias shape: {layer.self_attn.q_proj.bias.shape}")
if layer.self_attn.k_proj.bias is not None:
    print(f"k_proj.bias shape: {layer.self_attn.k_proj.bias.shape}")
if layer.self_attn.v_proj.bias is not None:
    print(f"v_proj.bias shape: {layer.self_attn.v_proj.bias.shape}")
if layer.self_attn.o_proj.bias is not None:
    print(f"o_proj.bias shape: {layer.self_attn.o_proj.bias.shape}")
