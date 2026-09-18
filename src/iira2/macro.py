"""The reproducible three-action Macro policy used over precomputed evidence.

The controller consumes detached scalar evidence features. It does not load or
own a language model, image encoder, or evidence extractor during RL training.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

import torch
from torch import Tensor, nn


class MacroAction(IntEnum):
    DIRECT_COMMIT = 0
    QUERY_AND_REVISE = 1
    ABSTAIN = 2


ACTIONS = tuple(MacroAction)
STATE_DIM = 9


def _prob(value: float | None) -> float:
    if value is None or not math.isfinite(float(value)):
        return 0.5
    return min(max(float(value), 1e-7), 1.0 - 1e-7)


def _entropy(probability: float) -> float:
    p = _prob(probability)
    return (-p * math.log(p) - (1.0 - p) * math.log1p(-p)) / math.log(2.0)


def build_state(
    qp: float | None,
    kp: float | None,
    qroi_p: float | None,
    loc_score: float | None,
) -> Tensor:
    """Build the fixed state ``[qp, q_ent, ...]`` in the declared order."""
    q = _prob(qp)
    k = _prob(kp)
    roi = _prob(qroi_p)
    loc = 0.0 if loc_score is None or not math.isfinite(float(loc_score)) else float(loc_score)
    values = [q, _entropy(q), abs(q - 0.5) * 2.0, k, abs(k - 0.5) * 2.0,
              abs(q - k), roi, abs(q - roi), loc]
    return torch.tensor(values, dtype=torch.float32)


def fuse_probability(
    action: MacroAction | int,
    state: Tensor,
    alpha_logit: Tensor | float,
    gamma_logit: Tensor | float,
) -> Tensor:
    """Apply the three-action probability contract."""
    action = MacroAction(int(action))
    qp, kp, qroi_p = state[..., 0], state[..., 3], state[..., 6]
    if action is MacroAction.DIRECT_COMMIT:
        return qp
    if action is MacroAction.ABSTAIN:
        return torch.full_like(qp, 0.5)
    alpha = torch.sigmoid(torch.as_tensor(alpha_logit, dtype=state.dtype, device=state.device))
    gamma = torch.sigmoid(torch.as_tensor(gamma_logit, dtype=state.dtype, device=state.device))
    return alpha * qp + (1.0 - alpha) * (gamma * kp + (1.0 - gamma) * qroi_p)


def compute_reward(
    probability: Tensor | float,
    label: Tensor | float,
    action: MacroAction | int,
    *,
    query_cost: float = 0.10,
    abstain_cost: float = 0.50,
    discrim_weight: float = 0.5,
) -> Tensor:
    """Return ``-Brier + discrim_weight*log_likelihood - action_cost``."""
    p = torch.as_tensor(probability, dtype=torch.float32)
    y = torch.as_tensor(label, dtype=p.dtype, device=p.device)
    p_safe = p.clamp(1e-7, 1.0 - 1e-7)
    reward = -(p - y).square() + discrim_weight * (y * p_safe.log() + (1.0 - y) * (1.0 - p_safe).log())
    action = MacroAction(int(action))
    if action is MacroAction.QUERY_AND_REVISE:
        reward = reward - query_cost
    elif action is MacroAction.ABSTAIN:
        reward = reward - abstain_cost
    return reward


class MacroPolicy(nn.Module):
    """9-to-64-to-64 MLP with action, alpha, and gamma heads."""

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64, n_actions: int = 3):
        super().__init__()
        if state_dim != STATE_DIM or n_actions != 3:
            raise ValueError("the public Macro contract is fixed at 9 state features and 3 actions")
        self.trunk = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.action_head = nn.Linear(hidden, n_actions)
        self.alpha_head = nn.Linear(hidden, 1)
        self.gamma_head = nn.Linear(hidden, 1)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        h = self.trunk(state)
        return self.action_head(h), self.alpha_head(h).squeeze(-1), self.gamma_head(h).squeeze(-1)


@dataclass(frozen=True)
class CISPOLoss:
    total: Tensor
    policy: Tensor
    entropy: Tensor
    kl_uniform: Tensor
    alpha_loss: Tensor
    gamma_loss: Tensor
    valid_group_count: int
    zero_advantage_group_count: int


def _group_advantage(rewards: Tensor, group_ids: Tensor) -> tuple[Tensor, int, int]:
    advantage = torch.zeros_like(rewards)
    valid = zero = 0
    for group in torch.unique(group_ids):
        mask = group_ids == group
        group_rewards = rewards[mask]
        if torch.all(group_rewards == group_rewards[0]):
            zero += 1
            continue
        advantage[mask] = group_rewards - group_rewards.mean()
        valid += 1
    return advantage, valid, zero


def macro_cispo_loss(
    current_logits: Tensor,
    behavior_logits: Tensor,
    actions: Tensor,
    rewards: Tensor,
    group_ids: Tensor,
    *,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.01,
    entropy_coef: float = 0.01,
) -> CISPOLoss:
    """Compute the terminal-action CISPO-style loss over sampled groups."""
    current_logp = torch.log_softmax(current_logits, dim=-1).gather(1, actions[:, None]).squeeze(1)
    behavior_logp = torch.log_softmax(behavior_logits.detach(), dim=-1).gather(1, actions[:, None]).squeeze(1)
    advantage, valid, zero = _group_advantage(rewards.detach(), group_ids)
    ratio = torch.exp(current_logp - behavior_logp).clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    if valid:
        policy = -(ratio * advantage * current_logp).mean()
    else:
        policy = current_logp.sum() * 0.0
    probabilities = current_logits.softmax(dim=-1)
    log_probabilities = torch.log_softmax(current_logits, dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    uniform_logp = math.log(1.0 / current_logits.shape[-1])
    kl_uniform = (probabilities * (log_probabilities - uniform_logp)).sum(dim=-1).mean()
    # Both reported pathwise terms use the same differentiable reward and
    # full-batch denominator. Preserve the factor of two explicitly.
    query_mask = (actions == int(MacroAction.QUERY_AND_REVISE)).to(rewards.dtype)
    alpha_loss = -(rewards * query_mask).mean()
    gamma_loss = -(rewards * query_mask).mean()
    total = policy + alpha_loss + gamma_loss - entropy_coef * entropy + kl_coef * kl_uniform
    return CISPOLoss(total, policy, entropy, kl_uniform, alpha_loss, gamma_loss, valid, zero)
