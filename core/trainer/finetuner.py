#!/usr/bin/env python3
"""
模型微调模块
用于剪枝后模型的性能恢复

支持两种微调方式：
1. 全参数微调（默认，推荐用于剪枝后的模型）
2. LoRA微调（低显存环境的备选方案）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from ..datasets.example_samples import get_examples


class FineTuner:
    """剪枝后模型的微调器"""

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda',
        logger = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_attention: bool = True,
        lora_target_mlp: bool = True
    ):
        """
        初始化微调器

        Args:
            model: 要微调的模型
            tokenizer: tokenizer
            device: 设备
            logger: 日志记录器（可选）
            use_lora: 是否使用LoRA微调（默认False，使用全参数微调）
            lora_r: LoRA秩（默认8）
            lora_alpha: LoRA缩放系数（默认16）
            lora_dropout: LoRA dropout率（默认0.05）
            lora_target_attention: 是否对Attention层应用LoRA（默认True）
            lora_target_mlp: 是否对MLP层应用LoRA（默认True）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.logger = logger
        self.use_lora = use_lora
        self.lora_config = None

        # 存储 LoRA 配置参数
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_attention = lora_target_attention
        self.lora_target_mlp = lora_target_mlp

    def log(self, message: str):
        """记录日志"""
        if self.logger:
            self.logger.log(message)
        else:
            print(message)

    def finetune(
        self,
        dataset_name: str = 'wikitext',
        num_samples: int = 500,
        seq_len: int = 512,
        lr: float = 1e-5,
        epochs: int = 1,
        batch_size: int = 1,
        split: str = 'train',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        use_scheduler: bool = True
    ) -> Dict[str, Any]:
        """
        微调模型（支持全参数微调和LoRA微调）

        Args:
            dataset_name: 数据集名称
            num_samples: 样本数量
            seq_len: 序列长度
            lr: 学习率
            epochs: 训练轮数
            batch_size: batch大小
            split: 数据集划分
            gradient_accumulation_steps: 梯度累积步数（模拟更大batch）
            max_grad_norm: 梯度裁剪阈值
            warmup_steps: 学习率预热步数
            weight_decay: 权重衰减系数
            use_scheduler: 是否使用学习率调度器

        Returns:
            微调统计信息字典
        """
        self.log(f"从 {dataset_name} {split} 集加载 {num_samples} 个样本...")

        # 加载训练数据
        finetune_data = get_examples(
            dataset_name,
            self.tokenizer,
            num_samples=num_samples,
            seq_len=seq_len,
            split=split
        )
        self.log(f"✅ 微调数据加载完成，shape: {finetune_data.shape}")

        # 如果使用LoRA，先配置LoRA
        if self.use_lora:
            self._setup_lora()

        # 准备优化器
        self.model.train()

        # 检查模型健康状态
        self.log("\n检查模型权重健康状态...")
        model_healthy, health_info = self._check_model_health()
        if not model_healthy:
            self.log("❌ 模型存在数值问题:")
            self.log(f"  NaN参数数量: {health_info['nan_count']:,}")
            self.log(f"  Inf参数数量: {health_info['inf_count']:,}")
            self.log("\n建议:")
            self.log("  1. 重新运行剪枝流程")
            self.log("  2. 检查剪枝率是否过高")
            self.log("  3. 运行诊断脚本: python diagnose_model.py --model_path <path>")
            raise ValueError("模型包含NaN或Inf值，无法进行微调")
        self.log("✅ 模型权重正常")

        # 只优化需要训练的参数
        if self.use_lora:
            # LoRA模式：只优化LoRA参数
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.log(f"LoRA模式：训练 {sum(p.numel() for p in trainable_params):,} 个参数")
        else:
            # 全参数微调模式
            trainable_params = self.model.parameters()
            total_params = sum(p.numel() for p in self.model.parameters())
            self.log(f"全参数微调模式：训练 {total_params:,} 个参数")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # 学习率调度器
        num_batches = (len(finetune_data) + batch_size - 1) // batch_size
        total_steps = num_batches * epochs // gradient_accumulation_steps

        scheduler = None
        if use_scheduler:
            try:
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
            except ImportError:
                self.log("⚠️ transformers版本较旧，使用线性学习率调度")
                # 使用PyTorch内置的线性调度器作为备选
                from torch.optim.lr_scheduler import LambdaLR
                def lr_lambda(current_step: int):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
                scheduler = LambdaLR(optimizer, lr_lambda)

        # 打印配置
        self.log(f"\n微调配置:")
        self.log(f"  模式: {'LoRA微调' if self.use_lora else '全参数微调'}")
        self.log(f"  学习率: {lr}")
        self.log(f"  轮数: {epochs}")
        self.log(f"  样本数: {num_samples}")
        self.log(f"  Batch size: {batch_size}")
        self.log(f"  梯度累积步数: {gradient_accumulation_steps}")
        self.log(f"  有效Batch size: {batch_size * gradient_accumulation_steps}")
        self.log(f"  序列长度: {seq_len}")
        self.log(f"  梯度裁剪: {max_grad_norm}")
        self.log(f"  权重衰减: {weight_decay}")
        self.log(f"  学习率调度: {'启用' if use_scheduler else '禁用'}")
        if use_scheduler and warmup_steps > 0:
            self.log(f"  预热步数: {warmup_steps}")

        # 训练统计
        epoch_losses = []
        step_losses = []
        global_step = 0

        # 微调循环
        for epoch in range(epochs):
            self.log(f"\n开始第 {epoch + 1}/{epochs} 轮微调...")

            total_loss = 0
            num_batches = (len(finetune_data) + batch_size - 1) // batch_size
            optimizer.zero_grad()

            for batch_idx in range(num_batches):
                # 获取batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(finetune_data))
                batch = finetune_data[start_idx:end_idx].to(self.device)

                # 前向传播
                outputs = self.model(batch, labels=batch)
                loss = outputs.loss

                # 检查loss是否有效
                loss_value = loss.item()
                if torch.isnan(loss) or torch.isinf(loss) or loss_value > 1e6:
                    self.log(f"\n❌ 检测到异常Loss: {loss_value}")
                    self.log(f"  批次索引: {batch_idx}/{num_batches}")
                    self.log(f"  轮次: {epoch + 1}/{epochs}")
                    self.log(f"  当前学习率: {lr if scheduler is None else scheduler.get_last_lr()[0]:.2e}")
                    self.log("\n可能原因:")
                    self.log("  1. 剪枝后的模型PPL异常高（>50），模型已经损坏")
                    self.log("  2. 学习率过高")
                    self.log("  3. 梯度爆炸")
                    self.log("\n建议解决方案:")
                    self.log("  1. 检查剪枝后PPL，如果>50需要重新剪枝")
                    self.log("  2. 降低学习率: --finetune_lr 1e-6 或 5e-7")
                    self.log("  3. 增强梯度裁剪: --finetune_max_grad_norm 0.5")
                    self.log("  4. 减少剪枝率: --pruning_ratio 0.15")
                    raise ValueError(f"训练出现异常Loss={loss_value}，已停止")

                # 梯度累积：除以累积步数
                loss = loss / gradient_accumulation_steps

                # 反向传播
                loss.backward()

                total_loss += loss_value

                # 梯度累积：每N步更新一次参数
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == num_batches:
                    # 梯度裁剪
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_grad_norm
                        )

                    # 更新参数
                    optimizer.step()

                    # 更新学习率
                    if scheduler is not None:
                        scheduler.step()

                    optimizer.zero_grad()
                    global_step += 1

                    # 记录步骤loss
                    step_losses.append(total_loss / (batch_idx + 1))

                # 每10%进度打印一次
                if (batch_idx + 1) % max(1, num_batches // 10) == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    progress = (batch_idx + 1) / num_batches * 100
                    if scheduler is not None:
                        current_lr = scheduler.get_last_lr()[0]
                    else:
                        current_lr = lr
                    self.log(f"  进度: {progress:.0f}% | 平均Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")

            avg_epoch_loss = total_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            self.log(f"✅ 第 {epoch + 1} 轮完成，平均Loss: {avg_epoch_loss:.4f}")

        self.log("\n✅ 微调完成！")

        # 如果使用LoRA，合并权重
        if self.use_lora:
            self.log("合并LoRA权重到模型...")
            self._merge_lora_weights()

        # 返回统计信息
        return {
            'method': 'lora' if self.use_lora else 'full',
            'lr': lr,
            'epochs': epochs,
            'samples': num_samples,
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': batch_size * gradient_accumulation_steps,
            'seq_len': seq_len,
            'max_grad_norm': max_grad_norm,
            'weight_decay': weight_decay,
            'total_steps': global_step,
            'epoch_losses': epoch_losses,
            'step_losses': step_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else None
        }

    def _setup_lora(self):
        """
        配置LoRA（如果PEFT可用）
        使用 __init__ 中存储的 LoRA 配置参数
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # 使用存储在 self 中的配置
            lora_r = self.lora_r
            lora_alpha = self.lora_alpha
            lora_dropout = self.lora_dropout
            target_attention = self.lora_target_attention
            target_mlp = self.lora_target_mlp

            # 构建target_modules列表
            target_modules = []

            if target_attention:
                # Attention层：q_proj, k_proj, v_proj, o_proj
                target_modules.extend(["q_proj", "k_proj", "v_proj", "o_proj"])

            if target_mlp:
                # MLP层：gate_proj, up_proj, down_proj
                target_modules.extend(["gate_proj", "up_proj", "down_proj"])

            if not target_modules:
                self.log("⚠️ 未指定任何target_modules，回退到全参数微调")
                self.use_lora = False
                return

            self.log(f"配置LoRA:")
            self.log(f"  秩(r): {lora_r}")
            self.log(f"  缩放(alpha): {lora_alpha}")
            self.log(f"  Dropout: {lora_dropout}")
            self.log(f"  目标模块: {', '.join(target_modules)}")

            # 配置LoRA
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
            )

            # 应用LoRA
            self.model = get_peft_model(self.model, self.lora_config)
            self.model.print_trainable_parameters()

        except ImportError:
            self.log("⚠️ PEFT库未安装，回退到全参数微调")
            self.log("安装命令: pip install peft")
            self.use_lora = False

    def _merge_lora_weights(self):
        """合并LoRA权重到基础模型"""
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                self.log("合并LoRA权重...")
                self.model = self.model.merge_and_unload()
                self.log("✅ LoRA权重已合并")
        except:
            self.log("⚠️ 无法合并LoRA权重，可能未使用PEFT")

    def _check_model_health(self) -> tuple[bool, dict]:
        """
        检查模型权重的健康状态

        Returns:
            (is_healthy, info): 布尔值表示是否健康，字典包含详细信息
        """
        nan_count = 0
        inf_count = 0

        for param in self.model.parameters():
            nan_count += torch.isnan(param).sum().item()
            inf_count += torch.isinf(param).sum().item()

        is_healthy = (nan_count == 0 and inf_count == 0)

        return is_healthy, {
            'nan_count': nan_count,
            'inf_count': inf_count
        }

    def save_finetuned_model(
        self,
        save_path: str,
        layer_pruning_rates: Dict[int, float],
        layer_importance: Dict[int, float],
        finetune_stats: Dict[str, Any],
        extra_info: Optional[Dict[str, Any]] = None
    ):
        """
        保存微调后的模型

        Args:
            save_path: 保存路径
            layer_pruning_rates: 层剪枝率
            layer_importance: 层重要性
            finetune_stats: 微调统计信息
            extra_info: 额外信息（可选）
        """
        self.log(f"保存微调后的模型到: {save_path}")

        self.model.half()

        save_dict = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'layer_pruning_rates': layer_pruning_rates,
            'layer_importance': layer_importance,
            'pruning_method': 'gqa_aware_taylor',
            'finetuned': True,
            'finetune_config': finetune_stats,
        }

        if extra_info:
            save_dict['config'] = extra_info

        torch.save(save_dict, save_path)
        self.log(f"✅ 微调后模型已保存")
