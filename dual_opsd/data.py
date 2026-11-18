# `data.py` 负责读写数据，并把不同长度的提示与回答整理成模型输入，
# 标记有效 token （attention_mask）和损失计算位置(response_mask)，
# 保证学生与教师对同一回答逐 token 对齐。

import hashlib
import json
from pathlib import Path

import torch


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_view(prompt_ids, response_ids, device):
    """A common continuation with role-specific prompt offsets. Batch size is one."""
    if not prompt_ids or not response_ids:
        raise ValueError("Empty prompt or continuation")
    ids = torch.tensor([list(prompt_ids) + list(response_ids)], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "start": len(prompt_ids) - 1, "length": len(response_ids)}


def response_logits(model_output, view):
    start, length = view["start"], view["length"]
    return model_output.logits[:, start:start + length, :]


def model_inputs(view):
    return {key: view[key] for key in ("input_ids", "attention_mask", "position_ids") if key in view}


def make_batch_view(prompts, responses, device, pad_token_id):
    """Left-pad prompts to align continuation positions; right-pad responses."""
    if len(prompts) != len(responses) or not responses or any(not r for r in responses):
        raise ValueError("Invalid batch")
    width = max(map(len, prompts))
    length = max(map(len, responses))
    ids, attention, masks = [], [], []
    for prompt, response in zip(prompts, responses):
        left, right = width - len(prompt), length - len(response)
        # ids
        ids.append([pad_token_id] * left + prompt + response + [pad_token_id] * right)
        # attention_mask补全
        attention.append([0] * left + [1] * (len(prompt) + len(response)) + [0] * right)
        # response_mask补全
        masks.append([True] * len(response) + [False] * right)
    attention = torch.tensor(attention, device=device)
    positions = (attention.cumsum(-1) - 1).clamp_min(0)
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": attention,
            "position_ids": positions, "start": width - 1, "length": length,
            "response_mask": torch.tensor(masks, device=device)}