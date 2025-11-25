#!/usr/bin/env python3
"""
LoRA 微调脚本 - 基于 LLM-Pruner 改进

用于对剪枝后的模型进行 LoRA 微调，提升性能

用法:
    单GPU训练的时候，必须添加 CUDA_VISIBLE_DEVICES = X,否则会出现数据分配在多GPU的情况,模型在单GPU，导致报错
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
功能:
    1. 加载剪枝后的模型
    2. 使用 LoRA 微调
    3. 自动评估微调后的模型
    4. 保存结果到 results/<model_name>_finetuned/
"""

import os
import sys
import argparse
import subprocess
import gc
from typing import List
from pathlib import Path

import torch
import transformers
from datasets import load_dataset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        prepare_model_for_kbit_training,
    )
except ImportError:
    print("错误: 未安装 peft 库")
    print("请运行: pip install peft")
    sys.exit(1)

from core.utils.get_best_gpu import get_best_gpu
device = "cuda:"+str(get_best_gpu()) if torch.cuda.is_available() else "cpu"

class Prompter:
    """Alpaca 风格的提示词生成器"""

    def __init__(self, template_name: str = "alpaca"):
        self.template_name = template_name
        self._verbose = False

        if template_name == "alpaca":
            self.template = {
                "description": "Template used by Alpaca-LoRA.",
                "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
                "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
                "response_split": "### Response:"
            }
        else:
            raise ValueError(f"Unknown template: {template_name}")

    def generate_prompt(
        self,
        instruction: str,
        input: str = None,
        label: str = None,
    ) -> str:
        """生成提示词"""
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res


def main(args):
    # 确保使用正确的 device
    # 如果用户没有指定 --device，使用全局的 device (基于 get_best_gpu)
    if not hasattr(args, 'device') or args.device is None:
        args.device = device  # 使用全局 device（get_best_gpu）

    print(f"\n{'='*80}")
    print(f"LoRA 微调脚本")
    print(f"{'='*80}\n")
    print(f"使用设备: {args.device}")

    # 设置 WandB (可选)
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        print(f"WandB 项目: {args.wandb_project}")
    else:
        os.environ["WANDB_DISABLED"] = "true"
        print("WandB 已禁用")

    # 加载剪枝后的模型
    print(f"\n加载剪枝模型: {args.pruned_model}")
    pruned_dict = torch.load(args.pruned_model, map_location=args.device,weights_only=False)
    tokenizer = pruned_dict['tokenizer']
    model = pruned_dict['model']
    print(f"✓ 模型加载成功")

    # 获取原模型名称（用于保存）
    pruned_model_path = Path(args.pruned_model)
    original_model_name = pruned_model_path.parent.name
    output_model_name = f"{original_model_name}_finetuned"

    # 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"results/{output_model_name}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 计算梯度累积步数
    gradient_accumulation_steps = args.batch_size // args.micro_batch_size

    # Prompter
    prompter = Prompter(args.prompt_template_name)

    # DDP 设置
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        print(f"DDP 模式: world_size={world_size}")

    # 准备模型
    if 'cuda' in args.device:
        model.half()
        print(f"✓ 模型转换为 FP16")

    # 设置 tokenizer
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        """Tokenize 函数"""
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        """生成并 tokenize 提示词"""
        # Alpaca 数据集格式
        if 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point.get("input", ""),
                data_point.get("output", ""),
            )
        else:
            raise NotImplementedError(f"不支持的数据集: {args.data_path}")

        tokenized_full_prompt = tokenize(full_prompt)

        if not args.train_on_inputs:
            # 只计算 response 部分的 loss
            user_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point.get("input", "")
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=args.add_eos_token)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            )

        return tokenized_full_prompt

    # 准备 LoRA
    print(f"\n配置 LoRA:")
    print(f"  r: {args.lora_r}")
    print(f"  alpha: {args.lora_alpha}")
    print(f"  dropout: {args.lora_dropout}")
    print(f"  target_modules: {args.lora_target_modules}")

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # 加载训练数据集
    print(f"\n加载数据集: {args.data_path}")
        # 检查是否是本地 JSON 文件

    if os.path.isfile(args.data_path) and args.data_path.endswith('.json'):
        print(f"✓ 从本地 JSON 文件加载数据")
        data = load_dataset('json', data_files=args.data_path)

    elif os.path.isdir(args.data_path):
        # 本地目录，查找 JSON 文件
        json_files = list(Path(args.data_path).glob('*.json'))
        if json_files:
            print(f"✓ 从本地目录加载数据: {json_files[0]}")
            data = load_dataset('json', data_files=str(json_files[0]))
        else:
            raise ValueError(f"在 {args.data_path} 中未找到 JSON 文件")
    else:
        # HuggingFace 数据集
        print(f"✓ 从 HuggingFace 加载数据")
        data = load_dataset(args.data_path)

    # 划分训练集和验证集
    train_val = data["train"].train_test_split(
        test_size=args.val_set_size, shuffle=True, seed=42
    )

    print(f"准备数据集...")
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

    print(f"✓ 训练样本: {len(train_data)}")
    print(f"✓ 验证样本: {len(val_data)}")

    # Trainer
    print(f"\n训练配置:")
    print(f"  batch_size: {args.batch_size}")
    print(f"  micro_batch_size: {args.micro_batch_size}")
    print(f"  gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"  num_epochs: {args.num_epochs}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  cutoff_len: {args.cutoff_len}")

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            optim="adamw_torch",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=100,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None if not ddp else False,
            group_by_length=args.group_by_length,
            report_to="wandb" if args.wandb_project else "none",
            run_name=output_model_name,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # 开始训练
    print(f"\n{'='*80}")
    print(f"开始训练...")
    print(f"{'='*80}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 恢复 state_dict
    model.state_dict = old_state_dict

    # 保存 LoRA adapter
    lora_output_dir = output_path / "lora_adapter"
    lora_output_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))
    print(f"\n✓ LoRA adapter 保存到: {lora_output_dir}")

    # 合并 LoRA 权重到基础模型
    print(f"\n合并 LoRA 权重到基础模型...")
    model = model.merge_and_unload()
    print(f"✓ LoRA 权重已合并")

    # 保存完整的微调模型 (pruned_model.bin 格式)
    model.half()
    finetuned_model_path = output_path / "pruned_model.bin"
    save_dict = {
        'model': model,
        'tokenizer': tokenizer,
        'layer_pruning_rates': pruned_dict.get('layer_pruning_rates', {}),
        'layer_importance': pruned_dict.get('layer_importance', {}),
        'pruning_method': pruned_dict.get('pruning_method', 'unknown'),
        'finetuned': True,
        'finetune_config': {
            'data_path': args.data_path,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
        }
    }

    torch.save(save_dict, finetuned_model_path)
    print(f"✓ 微调模型保存到: {finetuned_model_path}")

    # 清理GPU缓存，避免评估时OOM
    print(f"\n清理GPU缓存...")
    del model
    del tokenizer
    del pruned_dict
    del save_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"✓ GPU缓存已清理")

    # 自动评估
    if not args.skip_evaluation:
        print(f"\n{'='*80}")
        print(f"开始自动评估...")
        print(f"{'='*80}\n")

        evaluation_output_dir = output_path / "evaluation"
        evaluation_output_dir.mkdir(exist_ok=True)
        evaluation_output_file = evaluation_output_dir / "evaluation_results.json"

        # 调用评估脚本
        eval_cmd = [
            "python", "evaluation/run_evaluation.py",
            "--model_path", str(finetuned_model_path),
            "--metrics", "all",
            "--output", str(evaluation_output_file)
        ]

        print(f"执行评估命令: {' '.join(eval_cmd)}")
        try:
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            print(f"\n✓ 评估完成，结果保存到: {evaluation_output_file}")
        except subprocess.CalledProcessError as e:
            print(f"⚠ 评估失败:")
            print(e.stdout)
            print(e.stderr)
            print(f"\n您可以稍后手动运行评估:")
            print(f"  python evaluation/run_evaluation.py --model_path {finetuned_model_path} --metrics all --output {evaluation_output_file}")

    print(f"\n{'='*80}")
    print(f"✓ 微调完成！")
    print(f"{'='*80}")
    print(f"\n微调后的模型保存在: {output_dir}")
    print(f"  - LoRA adapter: {lora_output_dir}")
    print(f"  - 完整模型: {finetuned_model_path}")
    if not args.skip_evaluation:
        print(f"  - 评估结果: {evaluation_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LoRA 微调剪枝后的模型')

    # 模型和数据路径
    parser.add_argument('--pruned_model', type=str, required=True,
                       help='剪枝模型路径 (pruned_model.bin)')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned",
                       help='训练数据集路径')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录 (默认: results/<model_name>_finetuned)')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='总batch size (默认: 64)')
    parser.add_argument('--micro_batch_size', type=int, default=4,
                       help='每个GPU的batch size (默认: 4)')
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='训练轮数 (默认: 2)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--cutoff_len', type=int, default=256,
                       help='最大序列长度 (默认: 256)')
    parser.add_argument('--val_set_size', type=int, default=2000,
                       help='验证集大小 (默认: 2000)')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca",
                       help="提示词模板 (默认: alpaca)")

    # LoRA 配置
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA 秩 r (默认: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha (默认: 16)')
    parser.add_argument('--lora_dropout', type=float, default=0.05,
                       help='LoRA dropout (默认: 0.05)')
    parser.add_argument('--lora_target_modules', type=str,
                       default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj",
                       help='LoRA 目标模块')

    # 其他选项
    parser.add_argument('--train_on_inputs', default=False, action="store_true",
                       help='是否在输入部分计算loss (默认: False)')
    parser.add_argument('--add_eos_token', default=False, action="store_true",
                       help='添加EOS token')
    parser.add_argument('--group_by_length', default=False, action="store_true",
                       help="按长度分组 (更快但loss曲线奇怪)")
    parser.add_argument('--skip_evaluation', default=False, action="store_true",
                       help='跳过自动评估')

    # WandB
    parser.add_argument('--wandb_project', type=str, default="",
                       help="WandB 项目名称")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help="从检查点恢复训练")

    # 设备选择
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (默认: 自动选择最佳GPU)')

    # DDP
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='DDP local rank')

    args = parser.parse_args()

    main(args)
