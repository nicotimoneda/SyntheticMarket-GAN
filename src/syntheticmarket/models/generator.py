"""WGAN-GP generator.

A 2-layer *unidirectional* LSTM that maps a noise sequence to a 24-day price
window in ``[0, 1]``. The final ``Sigmoid`` is valid because the training data is
MinMax-scaled to ``[0, 1]``; it keeps the generator output inside the data range.

The attribute names (``lstm``, ``linear``) match the published checkpoint
``models/generator_wgan.pth`` so the paper weights load without remapping.
"""

from __future__ import annotations

import torch
from torch import nn

from syntheticmarket import config

# Paper defaults (do not change) — re-exported from :mod:`syntheticmarket.config`.
NOISE_DIM = config.NOISE_DIM
HIDDEN_DIM = config.HIDDEN_DIM
FEATURE_DIM = config.FEATURE_DIM
NUM_LAYERS = config.NUM_LAYERS
DROPOUT = config.DROPOUT


class Generator(nn.Module):
    """Noise sequence -> synthetic price window of shape ``(batch, seq_len, 1)``."""

    def __init__(
        self,
        noise_dim: int = NOISE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        feature_dim: int = FEATURE_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, seq_len, noise_dim)
        lstm_out, _ = self.lstm(z)  # (batch, seq_len, hidden_dim)
        out = self.linear(lstm_out)  # (batch, seq_len, feature_dim)
        return self.sigmoid(out)  # scaled to [0, 1]
