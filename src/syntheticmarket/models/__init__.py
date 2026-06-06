"""Generator and Critic networks for the WGAN-GP."""

from syntheticmarket.models.critic import Critic
from syntheticmarket.models.generator import Generator

__all__ = ["Generator", "Critic"]
