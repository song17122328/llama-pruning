#!/bin/bash
# Mistral 7B v0.3 - 20% 剪枝率

CUDA_VISIBLE_DEVICES=0 python layer_pruning.py \
  --base_model mistralai/Mistral-7B-v0.3 \
  --save_ckpt_log_name Mistral-7B-v0.3/prune_20 \
  --pruning_ratio 0.2 \
  --pruning_distribution 5:5 \
  --pruning_strategy inverse \
  --layer_importance_weight 1.0 \
  --layer_importance_method removal \
  --layer_importance_samples 50 \
  --channel_importance_samples 10 \
  --taylor_seq_len 128 \
  --nsamples 128 \
  --device cuda:0 \
  --save_model
