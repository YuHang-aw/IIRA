import pytest
import torch

from iira2.macro import (
    ACTIONS,
    build_state,
    compute_reward,
    macro_cispo_loss,
    fuse_probability,
)


def test_state_order_and_entropy_are_stable():
    state = build_state(0.72, 0.85, 0.91, 0.72)
    assert state.shape == (9,)
    assert state.tolist() == pytest.approx(
        [0.72, 0.855451, 0.44, 0.85, 0.70, 0.13, 0.91, 0.19, 0.72],
        abs=1e-5,
    )


def test_fusion_matches_three_action_contract():
    state = build_state(0.72, 0.85, 0.91, 0.72)
    alpha_logit = torch.tensor(0.05)
    gamma_logit = torch.tensor(-0.02)
    assert fuse_probability(ACTIONS[0], state, alpha_logit, gamma_logit) == pytest.approx(0.72)
    assert fuse_probability(ACTIONS[1], state, alpha_logit, gamma_logit) == pytest.approx(0.7982, abs=1e-4)
    assert fuse_probability(ACTIONS[2], state, alpha_logit, gamma_logit) == pytest.approx(0.5)


def test_reward_uses_brier_loglik_and_action_cost():
    reward = compute_reward(0.7982, 1, ACTIONS[1], query_cost=0.10, abstain_cost=0.50)
    assert reward == pytest.approx(-0.2534, abs=2e-3)


def test_cispo_uses_group_baseline_and_only_current_logits_grad():
    current = torch.zeros(3, 3, requires_grad=True)
    behavior = torch.zeros(3, 3)
    result = macro_cispo_loss(
        current, behavior, torch.tensor([0, 1, 2]), torch.tensor([1.0, 0.0, -1.0]),
        torch.zeros(3, dtype=torch.long), clip_ratio=0.2, kl_coef=0.01, entropy_coef=0.01,
    )
    result.total.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()
    assert current.grad.norm() > 0
    assert result.valid_group_count == 1


def test_missing_probabilities_are_neutral_and_clipped():
    state = build_state(None, None, None, None)
    assert torch.equal(state, torch.tensor([0.5, 1.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0]))


def test_query_pathwise_reward_has_two_copies_and_detached_policy_advantage():
    reward = torch.tensor([0.2, 0.8, -0.5], requires_grad=True)
    logits = torch.zeros(3, 3, requires_grad=True)
    result = macro_cispo_loss(logits, logits.detach(), torch.tensor([0, 1, 2]),
                             reward, torch.zeros(3, dtype=torch.long),
                             kl_coef=0, entropy_coef=0)
    result.total.backward()
    assert reward.grad is not None
    assert reward.grad.tolist() == pytest.approx([0, -2/3, 0])


def test_small_nonzero_advantage_is_not_suppressed():
    logits = torch.zeros(2, 3, requires_grad=True)
    result = macro_cispo_loss(logits, logits.detach(), torch.tensor([0, 2]),
                             torch.tensor([1.0, 1.000001]), torch.zeros(2, dtype=torch.long),
                             kl_coef=0, entropy_coef=0)
    result.policy.backward()
    assert logits.grad.norm() > 0
