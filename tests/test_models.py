"""Tests for the Generator and Critic shapes and output ranges."""

import torch

from syntheticmarket.models.critic import Critic
from syntheticmarket.models.generator import NOISE_DIM, Generator

BATCH = 8
SEQ_LEN = 24


def test_generator_output_shape():
    gen = Generator()
    z = torch.randn(BATCH, SEQ_LEN, NOISE_DIM)
    out = gen(z)
    assert out.shape == (BATCH, SEQ_LEN, 1)


def test_generator_output_in_unit_range():
    gen = Generator()
    z = torch.randn(BATCH, SEQ_LEN, NOISE_DIM)
    out = gen(z)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_critic_output_shape():
    critic = Critic()
    x = torch.rand(BATCH, SEQ_LEN, 1)
    out = critic(x)
    assert out.shape == (BATCH, 1)


def test_critic_output_is_unbounded_real_score():
    # No sigmoid: feeding large inputs should be able to produce values outside [0, 1].
    critic = Critic()
    with torch.no_grad():
        critic.linear.bias.fill_(5.0)
    x = torch.rand(BATCH, SEQ_LEN, 1)
    out = critic(x)
    assert out.dtype == torch.float32
    assert (out > 1.0).any()
