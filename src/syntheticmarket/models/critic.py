"""WGAN-GP critic.

A 2-layer *unidirectional* LSTM that summarises a 24-day window through the final
hidden state of the last layer (``h_n[-1]``) and emits a single real-valued score.
There is deliberately **no Sigmoid**: in the Wasserstein formulation the critic
estimates a distance, not a probability.
"""

from __future__ import annotations

import torch
from torch import nn

# Paper defaults (do not change).
FEATURE_DIM = 1
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2


class Critic(nn.Module):
    """Price window ``(batch, seq_len, 1)`` -> scalar score ``(batch, 1)``."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature_dim)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim) — last layer only
        return self.linear(last_hidden)  # (batch, 1)
