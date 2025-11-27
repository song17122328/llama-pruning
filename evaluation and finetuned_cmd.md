建议使用这个版本评估：pip install datasets==2.18.0 lm-eval==0.4.2
评估前需要清理缓存：
rm -rf ~/.cache/huggingface/datasets/*

# 评估sliceGPT
python evaluation/run_evaluation.py     \
    --model_path results/SliceGPT_PCA_2000/Llama-3-8B-Instruct_0.2.pt     \
    --slicegpt_base_model /newdata/LLMs/Llama-3-8B-Instruct     \
    --metrics ppl,zeroshot,speed,memory     \
    --output results/SliceGPT_PCA_2000/evaluation/evaluation_results.json

# 评估 LLM-Puner
python evaluation/run_evaluation.py \
    --model_path results/LLM-Pruner_1937/pruned_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --output results/LLM-Pruner_1937/evaluation/evaluation_results.json 

# 评估shortGPT
python evaluation/run_evaluation.py \
    --model_path results/ShortGPT_remove_8/pruned_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --output results/ShortGPT_remove_8/evaluation/evaluation_results.json 

# 评估Wanda
python evaluation/run_evaluation.py \
    --model_path results/Wanda_2000/pruned_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --output results/Wanda_2000/evaluation/evaluation_results.json 

# 评估global_taylor
python evaluation/run_evaluation.py \
    --model_path results/taylor_only_2000/pruned_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --output results/taylor_only_2000/evaluation/evaluation_results.json 

# 评估 Magnitude
python evaluation/run_evaluation.py \
    --model_path results/Magnitude_2000/pruned_model.bin \
    --metrics ppl,zeroshot,speed,memory \
    --output results/Magnitude_2000/evaluation/evaluation_results.json 

微调taylor_only_2000
CUDA_VISIBLE_DEVICES=0 python finetune_lora.py \
    --pruned_model results/taylor_only_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --data_path /newdata/DataSets/alpaca-cleaned/alpaca_data_cleaned.json \
    --device cuda:0 \
    --wandb_project Taylo_only_finetune_lora


微调LLM-Pruner
CUDA_VISIBLE_DEVICES=1 python finetune_lora.py \
    --pruned_model results/LLM-Pruner_1937/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --data_path /newdata/DataSets/alpaca-cleaned/alpaca_data_cleaned.json \
    --device cuda:0 \
    --wandb_project LLM-Pruner_1937_finetune_lora


微调layer_wise
CUDA_VISIBLE_DEVICES=2 python finetune_lora.py \
    --pruned_model results/layerwise_only_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --data_path /newdata/DataSets/alpaca-cleaned/alpaca_data_cleaned.json \
    --device cuda:0 \
    --wandb_project layerwise_only_2000_finetune_lora

微调block_wise
CUDA_VISIBLE_DEVICES=3 python finetune_lora.py \
    --pruned_model results/blockwise_only_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --data_path /newdata/DataSets/alpaca-cleaned/alpaca_data_cleaned.json \
    --device cuda:0 \
    --wandb_project blockwise_only_2000_finetune_lora

微调 HGSP_2000
CUDA_VISIBLE_DEVICES=4 python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --data_path /newdata/DataSets/alpaca-cleaned/alpaca_data_cleaned.json \
    --device cuda:0 \
    --wandb_project HGSP_2000_finetune_lora

微调原来的Llama模型 Llama-3-8B-Instruct
CUDA_VISIBLE_DEVICES=4 python finetune_lora.py \
    --pruned_model results/HGSP_2000/pruned_model.bin \
    --data_path yahma/alpaca-cleaned \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --micro_batch_size 4 \
    --data_path /newdata/DataSets/alpaca-cleaned/alpaca_data_cleaned.json \
    --device cuda:0 \
    --wandb_project HGSP_2000_finetune_lora