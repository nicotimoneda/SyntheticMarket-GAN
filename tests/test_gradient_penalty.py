"""Tests for the gradient penalty term."""

import torch

from syntheticmarket.models.critic import Critic
from syntheticmarket.training.gradient_penalty import compute_gradient_penalty

BATCH = 8
SEQ_LEN = 24


def test_gradient_penalty_finite_and_nonnegative():
    torch.manual_seed(0)
    critic = Critic()
    real = torch.rand(BATCH, SEQ_LEN, 1)
    fake = torch.rand(BATCH, SEQ_LEN, 1)
    gp = compute_gradient_penalty(critic, real, fake, device="cpu")
    assert torch.isfinite(gp)
    assert gp.item() >= 0.0


def test_gradient_penalty_when_real_equals_fake():
    # Degenerate case: real == fake. The interpolation collapses onto a single
    # point regardless of alpha, so the penalty is still well-defined and >= 0.
    torch.manual_seed(0)
    critic = Critic()
    real = torch.rand(BATCH, SEQ_LEN, 1)
    gp = compute_gradient_penalty(critic, real, real.clone(), device="cpu")
    assert torch.isfinite(gp)
    assert gp.item() >= 0.0
