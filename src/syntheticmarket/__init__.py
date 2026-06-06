"""SyntheticMarket-GAN: synthetic financial time series with a WGAN-GP.

A small, faithful refactor of the original notebook into an installable package.
See the companion blog post for the full write-up and validation results:
https://nicotimoneda.substack.com/p/wgan-gp-on-aapl-when-pca-looks-fine
"""

from syntheticmarket.models.critic import Critic
from syntheticmarket.models.generator import Generator

__version__ = "0.1.0"

__all__ = ["Generator", "Critic", "__version__"]
