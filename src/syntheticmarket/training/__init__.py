"""Training utilities: gradient penalty and the WGAN-GP training loop."""

from syntheticmarket.training.gradient_penalty import compute_gradient_penalty
from syntheticmarket.training.trainer import TrainConfig, set_seed, train

__all__ = ["compute_gradient_penalty", "TrainConfig", "train", "set_seed"]
