"""WGAN-GP training loop.

Faithful migration of the notebook training loop. Hyperparameter defaults match
the published paper exactly (Adam beta1=0.0 is the decision that matters most):

    lr=1e-4, betas=(0.0, 0.9), lambda_gp=10, n_critic=5, batch_size=64,
    num_epochs=200, seq_len=24, noise_dim=10.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from syntheticmarket import config as paper
from syntheticmarket.models.critic import Critic
from syntheticmarket.models.generator import Generator
from syntheticmarket.training.gradient_penalty import compute_gradient_penalty


@dataclass
class TrainConfig:
    """WGAN-GP hyperparameters (paper defaults)."""

    seq_len: int = paper.SEQ_LEN
    noise_dim: int = paper.NOISE_DIM
    hidden_dim: int = paper.HIDDEN_DIM
    feature_dim: int = paper.FEATURE_DIM
    batch_size: int = paper.BATCH_SIZE
    num_epochs: int = paper.NUM_EPOCHS
    lr: float = paper.LR
    beta1: float = paper.BETA1
    beta2: float = paper.BETA2
    lambda_gp: float = paper.LAMBDA_GP
    n_critic: int = paper.N_CRITIC


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    windows: np.ndarray,
    config: TrainConfig | None = None,
    *,
    seed: int = 42,
    device: torch.device | str | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[Generator, Critic, dict[str, list[float]]]:
    """Train the WGAN-GP on ``windows`` of shape ``(N, seq_len, 1)``.

    Returns the trained ``(generator, critic, history)``. If ``output_dir`` is
    given, the generator weights are saved to ``output_dir/generator_wgan.pth``.
    """
    config = config or TrainConfig()
    set_seed(seed)
    device = torch.device(device) if device is not None else torch.device("cpu")

    data = torch.as_tensor(np.asarray(windows, dtype="float32"))
    loader = DataLoader(
        TensorDataset(data),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    generator = Generator(
        noise_dim=config.noise_dim,
        hidden_dim=config.hidden_dim,
        feature_dim=config.feature_dim,
    ).to(device)
    critic = Critic(feature_dim=config.feature_dim, hidden_dim=config.hidden_dim).to(
        device
    )

    betas = (config.beta1, config.beta2)
    opt_g = optim.Adam(generator.parameters(), lr=config.lr, betas=betas)
    opt_c = optim.Adam(critic.parameters(), lr=config.lr, betas=betas)

    history: dict[str, list[float]] = {"loss_c": [], "loss_g": []}

    for epoch in range(config.num_epochs):
        loss_c = torch.tensor(0.0)
        loss_g = torch.tensor(0.0)
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size = real_batch.size(0)

            # --- N_CRITIC critic updates per generator update ---
            for _ in range(config.n_critic):
                z = torch.randn(
                    batch_size, config.seq_len, config.noise_dim, device=device
                )
                fake = generator(z).detach()
                gp = compute_gradient_penalty(critic, real_batch, fake, device)
                loss_c = (
                    -(critic(real_batch).mean() - critic(fake).mean())
                    + config.lambda_gp * gp
                )
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()

            # --- 1 generator update ---
            z = torch.randn(batch_size, config.seq_len, config.noise_dim, device=device)
            fake = generator(z)
            loss_g = -critic(fake).mean()
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        history["loss_c"].append(float(loss_c.item()))
        history["loss_g"].append(float(loss_g.item()))
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(
                f"Epoch [{epoch + 1}/{config.num_epochs}] "
                f"C: {loss_c.item():.4f} | G: {loss_g.item():.4f}"
            )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        weights_path = output_dir / "generator_wgan.pth"
        torch.save(generator.state_dict(), weights_path)
        if verbose:
            print(f"Saved generator weights to {weights_path}")

    return generator, critic, history
