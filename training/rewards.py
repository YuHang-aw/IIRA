"""奖励与工具函数
- Brier 奖励（终局）
- 熵正则（目标熵）
- 动作 token id 辅助
"""
from __future__ import annotations
from typing import Dict
import torch

ACTION_TOKENS = ["CLAIM", "CHECK", "ABSTAIN", "STOP"]

def extract_action_token_ids(tok) -> Dict[str, int]:
    ids = {}
    for a in ACTION_TOKENS:
        ids[a] = tok.convert_tokens_to_ids(a) if tok.convert_tokens_to_ids(a) is not None else tok(a, add_special_tokens=False).input_ids[0]
    return ids

def brier_reward(p: float, g: float) -> float:
    return - (p - g) ** 2

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return - (probs * torch.log(probs + 1e-12)).sum(dim=-1)
