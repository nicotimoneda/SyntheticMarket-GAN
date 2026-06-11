"""Paper constants — single source of truth for the published configuration.

Every value here reproduces the results reported in the blog post. Changing any
of them means the run is no longer the paper baseline.
"""

from __future__ import annotations

# Data pipeline (blog post, Section 3)
TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2025-11-29"
SEQ_LEN = 24
WINDOW_STEP = 1
FEATURE_RANGE = (0.0, 1.0)

# Architecture (blog post, Section 4)
NOISE_DIM = 10
HIDDEN_DIM = 64
FEATURE_DIM = 1
NUM_LAYERS = 2
DROPOUT = 0.2

# Training (blog post, Section 5 — Gulrajani et al. 2017)
BATCH_SIZE = 64
NUM_EPOCHS = 200
LR = 1e-4
BETA1 = 0.0
BETA2 = 0.9
LAMBDA_GP = 10.0
N_CRITIC = 5

# Evaluation (blog post, Section 6)
SEED = 42
N_EVAL_SAMPLES = 500
TSNE_PERPLEXITY = 30.0
TSNE_MAX_ITER = 1000
