#!/usr/bin/env python3
"""
自定义 Zero-shot 评估器

不依赖 lm-eval，直接从本地 jsonl 文件加载数据并评估。
通过计算每个选项的 log-likelihood 来选择答案。

支持的任务:
- piqa: 物理常识推理
- boolq: 是非问答
- hellaswag: 常识推理
- winogrande: 代词消歧
- arc_easy/arc_challenge: 科学问答
- openbookqa: 科学推理
"""

import os
import json
import re
import html
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import gc


def get_project_root():
    """获取项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_jsonl(file_path: str) -> List[dict]:
    """加载 jsonl 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_loglikelihood(
    model,
    tokenizer,
    context: str,
    continuation: str,
    device: str
) -> float:
    """
    计算给定上下文后续文本的 log-likelihood

    Args:
        model: 语言模型
        tokenizer: tokenizer
        context: 上下文文本
        continuation: 续写文本
        device: 设备

    Returns:
        log-likelihood 值
    """
    # 编码上下文和完整文本
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_text = context + continuation
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # 获取续写部分的 token ids
    continuation_ids = full_ids[len(context_ids):]

    if len(continuation_ids) == 0:
        return float('-inf')

    # 准备输入
    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # 计算续写部分的 log-likelihood
    # logits shape: [1, seq_len, vocab_size]
    # 我们需要从 context 结束位置开始计算
    start_pos = len(context_ids) - 1  # -1 因为我们用前一个 token 预测当前 token

    log_likelihood = 0.0
    for i, token_id in enumerate(continuation_ids):
        pos = start_pos + i
        if pos >= logits.shape[1]:
            break

        # 获取该位置的 logits 并计算 log_softmax
        token_logits = logits[0, pos, :]
        log_probs = F.log_softmax(token_logits, dim=-1)
        log_likelihood += log_probs[token_id].item()

    return log_likelihood


def compute_loglikelihood_batched(
    model,
    tokenizer,
    contexts: List[str],
    continuations: List[str],
    device: str
) -> List[Tuple[float, int]]:
    """
    批量计算多个 (context, continuation) 对的 log-likelihood

    Args:
        model: 语言模型
        tokenizer: tokenizer
        contexts: 上下文列表
        continuations: 续写文本列表
        device: 设备

    Returns:
        List of (log_likelihood, num_tokens) tuples for length normalization
    """
    if len(contexts) != len(continuations):
        raise ValueError("contexts 和 continuations 长度必须相同")

    batch_size = len(contexts)
    if batch_size == 0:
        return []

    # 编码所有样本
    all_context_ids = []
    all_full_ids = []
    all_continuation_lengths = []

    for context, continuation in zip(contexts, continuations):
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        full_text = context + continuation
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        continuation_ids = full_ids[len(context_ids):]

        all_context_ids.append(context_ids)
        all_full_ids.append(full_ids)
        all_continuation_lengths.append(len(continuation_ids))

    # 找到最大长度并填充
    max_len = max(len(ids) for ids in all_full_ids)

    # 创建填充后的张量
    padded_input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for i, full_ids in enumerate(all_full_ids):
        seq_len = len(full_ids)
        padded_input_ids[i, :seq_len] = torch.tensor(full_ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    # 批量前向传播
    with torch.no_grad():
        outputs = model(input_ids=padded_input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # 计算每个样本的 log-likelihood
    results = []

    for i in range(batch_size):
        context_len = len(all_context_ids[i])
        full_len = len(all_full_ids[i])
        continuation_len = all_continuation_lengths[i]

        if continuation_len == 0:
            results.append((float('-inf'), 1))
            continue

        # 计算续写部分的 log-likelihood
        start_pos = context_len - 1
        log_likelihood = 0.0

        for j in range(continuation_len):
            pos = start_pos + j
            if pos >= full_len:
                break

            token_id = all_full_ids[i][pos + 1] if pos + 1 < full_len else 0
            token_logits = logits[i, pos, :]
            log_probs = F.log_softmax(token_logits, dim=-1)
            log_likelihood += log_probs[token_id].item()

        results.append((log_likelihood, continuation_len))

    return results


def evaluate_multiple_choice_batched(
    model,
    tokenizer,
    questions: List[dict],
    format_fn,
    device: str,
    task_name: str = "",
    batch_size: int = 8
) -> Tuple[float, int, int]:
    """
    批量评估多选题任务（更快）

    Args:
        model: 语言模型
        tokenizer: tokenizer
        questions: 问题列表
        format_fn: 格式化函数，返回 (context, choices, label)
        device: 设备
        task_name: 任务名称（用于显示）
        batch_size: 批次大小

    Returns:
        (accuracy, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0

    desc = f"评估 {task_name}" if task_name else "评估中"

    # 预处理所有问题
    all_items = []
    for item in questions:
        context, choices, label = format_fn(item)
        all_items.append((context, choices, label))

    # 按批次处理
    for batch_start in tqdm(range(0, len(all_items), batch_size), desc=desc):
        batch_items = all_items[batch_start:batch_start + batch_size]

        # 收集这个批次中所有的 (context, continuation) 对
        batch_contexts = []
        batch_continuations = []
        batch_indices = []  # (item_idx, choice_idx)

        for item_idx, (context, choices, label) in enumerate(batch_items):
            for choice_idx, choice in enumerate(choices):
                batch_contexts.append(context)
                batch_continuations.append(choice)
                batch_indices.append((item_idx, choice_idx))

        # 批量计算 log-likelihood
        ll_results = compute_loglikelihood_batched(
            model, tokenizer, batch_contexts, batch_continuations, device
        )

        # 整理结果并判断
        item_scores = {}  # item_idx -> [(ll, num_tokens) for each choice]
        for (item_idx, choice_idx), (ll, num_tokens) in zip(batch_indices, ll_results):
            if item_idx not in item_scores:
                item_scores[item_idx] = []
            item_scores[item_idx].append((ll, num_tokens))

        # 计算准确率 (使用 length-normalized scores，即 acc_norm)
        for item_idx, (context, choices, label) in enumerate(batch_items):
            scores_with_len = item_scores[item_idx]
            # 使用长度归一化的分数 (log_likelihood / num_tokens)
            normalized_scores = [ll / num_tokens for ll, num_tokens in scores_with_len]
            pred = normalized_scores.index(max(normalized_scores))

            if pred == label:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_multiple_choice(
    model,
    tokenizer,
    questions: List[dict],
    format_fn,
    device: str,
    task_name: str = ""
) -> Tuple[float, int, int]:
    """
    评估多选题任务

    Args:
        model: 语言模型
        tokenizer: tokenizer
        questions: 问题列表
        format_fn: 格式化函数，返回 (context, choices, label)
        device: 设备
        task_name: 任务名称（用于显示）

    Returns:
        (accuracy, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0

    desc = f"评估 {task_name}" if task_name else "评估中"

    for item in tqdm(questions, desc=desc):
        context, choices, label = format_fn(item)

        # 计算每个选项的 log-likelihood
        scores = []
        for choice in choices:
            ll = compute_loglikelihood(model, tokenizer, context, choice, device)
            scores.append(ll)

        # 选择得分最高的选项
        pred = scores.index(max(scores))

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


# ============== 任务格式化函数 ==============

def format_piqa(item: dict) -> Tuple[str, List[str], int]:
    """PIQA 格式化"""
    context = f"Question: {item['goal']}\nAnswer:"
    choices = [f" {item['sol1']}", f" {item['sol2']}"]  # 添加前导空格
    label = item['label']
    return context, choices, label


def format_boolq(item: dict) -> Tuple[str, List[str], int]:
    """BoolQ 格式化"""
    context = f"{item['passage']}\nQuestion: {item['question']}?\nAnswer:"
    choices = [" no", " yes"]  # 添加前导空格
    label = 1 if item['answer'] else 0
    return context, choices, label


def format_hellaswag(item: dict) -> Tuple[str, List[str], int]:
    """HellaSwag 格式化

    lm-eval 使用 ctx_a + ctx_b 作为 context，并对 endings 进行预处理。
    包括 HTML 实体解码和特殊标记处理。
    """
    def preprocess_text(text):
        """预处理文本：解码 HTML 实体，处理特殊标记"""
        # 解码 HTML 实体 (如 &amp; -> &, &quot; -> ")
        text = html.unescape(text)
        # 去除 [header] [title] 等标记
        text = re.sub(r'\[header\]\s*', '', text)
        text = re.sub(r'\[title\]\s*', '', text)
        text = re.sub(r'\[step\]\s*', '', text)
        text = re.sub(r'\[substeps\]\s*', '', text)
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    activity = item.get('activity_label', '')

    # lm-eval 使用 ctx_a 和 ctx_b
    ctx_a = item.get('ctx_a', '')
    ctx_b = item.get('ctx_b', '')

    # 如果没有 ctx_a/ctx_b，回退到 ctx
    if ctx_a and ctx_b:
        ctx_a = preprocess_text(ctx_a)
        ctx_b = preprocess_text(ctx_b)
        # ctx_b 首字母大写并添加到 ctx_a 后
        if ctx_b:
            ctx = ctx_a + " " + ctx_b[0].upper() + ctx_b[1:]
        else:
            ctx = ctx_a
    else:
        ctx = preprocess_text(item.get('ctx', ''))

    # 构建 context
    if activity:
        context = f"{activity}: {ctx}"
    else:
        context = ctx

    # 预处理 endings
    choices = []
    for ending in item['endings']:
        ending = preprocess_text(ending)
        # 确保有前导空格
        choices.append(f" {ending}")

    label = int(item['label'])
    return context, choices, label


def format_winogrande(item: dict) -> Tuple[str, List[str], int]:
    """Winogrande 格式化

    Winogrande 的句子包含 '_' 占位符，需要用选项替换。
    评估方式：计算替换后句子中选项部分的 log-likelihood。
    """
    sentence = item['sentence']
    option1 = item['option1']
    option2 = item['option2']

    # 找到 '_' 的位置，分割句子
    if '_' in sentence:
        parts = sentence.split('_')
        context = parts[0]  # '_' 之前的部分作为 context
        suffix = parts[1] if len(parts) > 1 else ''  # '_' 之后的部分

        # 选项 + 后缀作为 continuation
        choices = [f"{option1}{suffix}", f"{option2}{suffix}"]
    else:
        # 如果没有 '_'，回退到简单模式
        context = sentence
        choices = [option1, option2]

    # answer 是 "1" 或 "2"，转换为 0 或 1
    label = int(item['answer']) - 1
    return context, choices, label


def format_arc(item: dict) -> Tuple[str, List[str], int]:
    """ARC (Easy/Challenge) 格式化"""
    context = f"Question: {item['question']}\nAnswer:"
    # 添加前导空格
    choices = [f" {text}" for text in item['choices']['text']]
    labels = item['choices']['label']
    answer_key = item['answerKey']
    label = labels.index(answer_key)
    return context, choices, label


def format_openbookqa(item: dict) -> Tuple[str, List[str], int]:
    """OpenBookQA 格式化

    lm-eval 使用 fact1 作为额外上下文。
    """
    # 获取 fact1（如果存在）
    fact1 = item.get('fact1', '')

    # 构建 context
    if fact1:
        context = f"{fact1}\nQuestion: {item['question_stem']}\nAnswer:"
    else:
        context = f"Question: {item['question_stem']}\nAnswer:"

    # 添加前导空格
    choices = [f" {text}" for text in item['choices']['text']]
    labels = item['choices']['label']
    answer_key = item['answerKey']
    label = labels.index(answer_key)
    return context, choices, label


# ============== 主评估函数 ==============

TASK_CONFIGS = {
    'piqa': {
        'file': 'piqa/validation.jsonl',
        'format_fn': format_piqa
    },
    'boolq': {
        'file': 'boolq/validation.jsonl',
        'format_fn': format_boolq
    },
    'hellaswag': {
        'file': 'hellaswag/validation.jsonl',
        'format_fn': format_hellaswag
    },
    'winogrande': {
        'file': 'winogrande/validation.jsonl',
        'format_fn': format_winogrande
    },
    'arc_easy': {
        'file': 'arc_easy/validation.jsonl',
        'format_fn': format_arc
    },
    'arc_challenge': {
        'file': 'arc_challenge/validation.jsonl',
        'format_fn': format_arc
    },
    'openbookqa': {
        'file': 'openbookqa/validation.jsonl',
        'format_fn': format_openbookqa
    }
}


def evaluate_zeroshot_custom(
    model,
    tokenizer,
    tasks: List[str] = None,
    device: str = 'cuda',
    data_dir: str = None,
    batch_size: int = 8,
    use_batched: bool = True
) -> Dict[str, Dict]:
    """
    自定义 Zero-shot 评估（不使用 lm-eval）

    Args:
        model: 语言模型
        tokenizer: tokenizer
        tasks: 任务列表，默认所有任务
        device: 设备
        data_dir: 数据目录，默认 data/zeroshot
        batch_size: 批次大小（仅在 use_batched=True 时有效）
        use_batched: 是否使用批处理（更快，默认True）

    Returns:
        {task_name: {'accuracy': float, 'correct': int, 'total': int}}
    """
    if tasks is None:
        tasks = ['boolq', 'piqa', 'hellaswag', 'winogrande',
                 'arc_easy', 'arc_challenge', 'openbookqa']

    if data_dir is None:
        data_dir = os.path.join(get_project_root(), 'data', 'zeroshot')

    print(f"\n{'='*60}")
    print(f"自定义 Zero-shot 评估")
    print(f"{'='*60}")
    print(f"任务: {', '.join(tasks)}")
    print(f"数据目录: {data_dir}")
    if use_batched:
        print(f"批处理模式: batch_size={batch_size}")
    else:
        print(f"单样本模式")
    print()

    results = {}

    for task in tasks:
        if task not in TASK_CONFIGS:
            print(f"⚠ 未知任务: {task}，跳过")
            continue

        config = TASK_CONFIGS[task]
        file_path = os.path.join(data_dir, config['file'])

        if not os.path.exists(file_path):
            print(f"⚠ 数据文件不存在: {file_path}，跳过")
            continue

        # 加载数据
        questions = load_jsonl(file_path)
        print(f"\n{task}: 加载 {len(questions)} 个样本")

        # 评估（选择批处理或单样本模式）
        if use_batched:
            accuracy, correct, total = evaluate_multiple_choice_batched(
                model, tokenizer, questions,
                config['format_fn'],
                device,
                task_name=task,
                batch_size=batch_size
            )
        else:
            accuracy, correct, total = evaluate_multiple_choice(
                model, tokenizer, questions,
                config['format_fn'],
                device,
                task_name=task
            )

        # 每个任务后清理显存
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()

        results[task] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }

        print(f"✓ {task}: {accuracy*100:.2f}% ({correct}/{total})")

    # 计算平均准确率
    if results:
        avg_acc = sum(r['accuracy'] for r in results.values()) / len(results)
        print(f"\n平均准确率: {avg_acc*100:.2f}%")

    return results


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from evaluation.utils.model_loader import load_model_and_tokenizer

    parser = argparse.ArgumentParser(description='自定义 Zero-shot 评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--tasks', type=str, default=None,
                       help='任务列表，逗号分隔')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录，默认 data/zeroshot')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小（默认8）')
    parser.add_argument('--no_batch', action='store_true',
                       help='禁用批处理，使用单样本模式')
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        force_single_device=True
    )

    # 解析任务
    tasks = args.tasks.split(',') if args.tasks else None

    # 评估
    results = evaluate_zeroshot_custom(
        model, tokenizer,
        tasks=tasks,
        device=args.device,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        use_batched=not args.no_batch
    )

    print(f"\n完整结果: {json.dumps(results, indent=2)}")
