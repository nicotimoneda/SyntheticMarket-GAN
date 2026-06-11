"""PCA and t-SNE projections of price windows into 2-D.

Each window of shape ``(seq_len, 1)`` is flattened to a ``seq_len`` vector before
projection, matching the validation procedure in the blog post (Section 6).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from syntheticmarket import config


def _flatten(sequences: np.ndarray) -> np.ndarray:
    arr = np.asarray(sequences, dtype="float32")
    return arr.reshape(arr.shape[0], -1)


def pca_analysis(sequences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project sequences to 2-D with PCA.

    Returns ``(coords, explained_variance_ratio)`` where ``coords`` is
    ``(N, 2)`` and ``explained_variance_ratio`` is ``(2,)``.
    """
    flat = _flatten(sequences)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(flat)
    return coords, pca.explained_variance_ratio_


def tsne_analysis(
    sequences: np.ndarray,
    perplexity: float = config.TSNE_PERPLEXITY,
    max_iter: int = config.TSNE_MAX_ITER,
    seed: int = config.SEED,
) -> np.ndarray:
    """Project sequences to 2-D with t-SNE (perplexity=30, max_iter=1000)."""
    flat = _flatten(sequences)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=seed,
        init="pca",
    )
    return tsne.fit_transform(flat)
