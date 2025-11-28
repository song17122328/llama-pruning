#!/usr/bin/env python3
"""
LoRA 微调模块

提供可复用的 LoRA 微调类，用于：
1. 剪枝后模型的性能恢复
2. 指令微调（Alpaca 等）

可以在剪枝流程中集成使用，也可以单独调用
"""

import os
import sys
import torch
import torch.nn as nn
import transformers
from typing import Optional, Dict, Any, List
from pathlib import Path

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("警告: 未安装 peft 库，LoRA 微调功能不可用")


class LoRAFineTuner:
    """LoRA 微调器 - 简化版，专注于集成到剪枝流程"""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        logger = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None
    ):
        """
        初始化 LoRA 微调器

        Args:
            model: 要微调的模型
            tokenizer: tokenizer
            device: 设备
            logger: 日志记录器（可选）
            lora_r: LoRA 秩
            lora_alpha: LoRA 缩放系数
            lora_dropout: LoRA dropout 率
            lora_target_modules: 目标模块列表（默认为 attention + mlp）
        """
        if not HAS_PEFT:
            raise ImportError("需要安装 peft 库: pip install peft")

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger

        # LoRA 配置
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # 默认目标模块
        if lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "down_proj", "up_proj"
            ]
        else:
            self.lora_target_modules = lora_target_modules

        self.lora_model = None

    def log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def prepare_lora_model(self):
        """准备 LoRA 模型"""
        self.log(f"\n配置 LoRA:")
        self.log(f"  r: {self.lora_r}")
        self.log(f"  alpha: {self.lora_alpha}")
        self.log(f"  dropout: {self.lora_dropout}")
        self.log(f"  target_modules: {self.lora_target_modules}")

        # 准备模型
        self.model = prepare_model_for_kbit_training(self.model)

        # LoRA 配置
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 应用 LoRA
        self.lora_model = get_peft_model(self.model, config)
        self.lora_model.print_trainable_parameters()

        return self.lora_model

    def finetune(
        self,
        dataset_name: str = 'wikitext',
        num_samples: int = 500,
        seq_len: int = 512,
        lr: float = 1e-5,
        epochs: int = 1,
        batch_size: int = 1,
        split: str = 'train',
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        weight_decay: float = 0.01
    ) -> Dict[str, Any]:
        """
        使用简单数据进行 LoRA 微调（用于剪枝后的性能恢复）

        Args:
            dataset_name: 数据集名称
            num_samples: 样本数量
            seq_len: 序列长度
            lr: 学习率
            epochs: 训练轮数
            batch_size: batch 大小
            split: 数据集划分
            gradient_accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪阈值
            warmup_steps: 预热步数
            weight_decay: 权重衰减

        Returns:
            微调统计信息字典
        """
        self.log(f"\n从 {dataset_name} {split} 集加载 {num_samples} 个样本进行微调...")

        # 准备 LoRA 模型
        if self.lora_model is None:
            self.prepare_lora_model()

        # 加载数据（使用项目内部的数据加载器）
        try:
            from core.datasets.example_samples import get_examples
            finetune_data = get_examples(
                dataset_name,
                self.tokenizer,
                nsamples=num_samples,
                seqlen=seq_len,
                split=split
            )
        except ImportError:
            self.log("警告: 无法导入 core.datasets.example_samples，使用简单数据")
            # 创建简单的虚拟数据作为备用
            from datasets import load_dataset
            dataset = load_dataset(dataset_name, split=split)
            finetune_data = dataset.select(range(min(num_samples, len(dataset))))

        # 准备 tokenizer
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        # 简单的 tokenize 函数
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'] if 'text' in examples else str(examples),
                truncation=True,
                max_length=seq_len,
                padding="max_length"
            )

        # Tokenize 数据
        if hasattr(finetune_data, 'map'):
            tokenized_data = finetune_data.map(tokenize_function, batched=True)
        else:
            # 如果是简单列表，转换为数据集
            from datasets import Dataset
            tokenized_data = Dataset.from_dict({'text': [str(x) for x in finetune_data]})
            tokenized_data = tokenized_data.map(tokenize_function, batched=True)

        # 配置 Trainer
        training_args = transformers.TrainingArguments(
            output_dir="./lora_finetuning_output",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=10,
            max_grad_norm=max_grad_norm,
            weight_decay=weight_decay,
            save_strategy="no",  # 不保存中间checkpoint
            report_to="none",  # 禁用 wandb
        )

        # 创建 Trainer
        trainer = transformers.Trainer(
            model=self.lora_model,
            args=training_args,
            train_dataset=tokenized_data,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            ),
        )

        # 开始训练
        self.log(f"\n开始 LoRA 微调...")
        self.log(f"  样本数: {len(tokenized_data)}")
        self.log(f"  Epochs: {epochs}")
        self.log(f"  学习率: {lr}")
        self.log(f"  Batch size: {batch_size}")

        trainer.train()

        self.log(f"✓ LoRA 微调完成")

        # 合并 LoRA 权重回主模型
        self.model = self.lora_model.merge_and_unload()

        return {
            'final_loss': trainer.state.log_history[-1].get('loss', 0.0) if trainer.state.log_history else 0.0,
            'num_samples': len(tokenized_data),
            'epochs': epochs
        }


__all__ = ['LoRAFineTuner']
