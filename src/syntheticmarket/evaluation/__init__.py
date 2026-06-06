"""Evaluation: dimensionality reduction (PCA/t-SNE) and step-to-step metrics."""

from syntheticmarket.evaluation.dimreduction import pca_2d, tsne_2d
from syntheticmarket.evaluation.metrics import (
    step_to_step_ratio,
    step_to_step_std,
    summary_metrics,
)

__all__ = [
    "pca_2d",
    "tsne_2d",
    "step_to_step_std",
    "step_to_step_ratio",
    "summary_metrics",
]
