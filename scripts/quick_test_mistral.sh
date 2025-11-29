#!/bin/bash
# Mistral 7B v0.3 - 快速测试（10% 剪枝）

CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/quick_test \
  --pruning_ratio 0.1 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 1.0 \
  --layer_importance_method removal \
  --layer_importance_samples 10 \
  --channel_importance_samples 5 \
  --taylor_seq_len 128 \
  --nsamples 32 \
  --device cuda:0 \
  --save_model
