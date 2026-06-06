"""Gradient penalty for the WGAN-GP 1-Lipschitz constraint (Gulrajani et al. 2017).

Penalises the critic when the norm of its gradient at a random interpolation
``x_hat = alpha * real + (1 - alpha) * fake`` deviates from 1:

    GP = E[(||grad_{x_hat} C(x_hat)||_2 - 1)^2]
"""

from __future__ import annotations

import torch
from torch import nn


def compute_gradient_penalty(
    critic: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute the scalar gradient penalty for a batch of real/fake windows."""
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)  # broadcast over (B, T, F)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    score = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=score,
        inputs=interpolated,
        grad_outputs=torch.ones_like(score),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_norm = gradients.view(batch_size, -1).norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()
