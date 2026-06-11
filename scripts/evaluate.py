"""Evaluate a trained generator: PCA + t-SNE figures and step-to-step metrics.

Loads generator weights, draws ``--n-samples`` real and synthetic windows, then
writes ``pca.png``, ``tsne.png`` and ``metrics.csv`` to the output directory.

Example
-------
    python scripts/evaluate.py --weights-dir models
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from syntheticmarket.data.loader import SEQ_LEN, load_price_windows  # noqa: E402
from syntheticmarket.evaluation.dimreduction import (  # noqa: E402
    pca_analysis,
    tsne_analysis,
)
from syntheticmarket.evaluation.metrics import summary_metrics  # noqa: E402
from syntheticmarket.models.generator import NOISE_DIM, Generator  # noqa: E402
from syntheticmarket.training.trainer import set_seed  # noqa: E402

REAL_COLOR = "#2E86DE"
SYNTH_COLOR = "#EE4C2C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained generator.")
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing generator_wgan.pth.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=500, help="Real/synthetic sample count."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where to write figures + CSV.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu/cuda)."
    )
    return parser.parse_args()


def _scatter(ax, coords, n, title) -> None:
    ax.scatter(
        coords[:n, 0], coords[:n, 1], s=12, c=REAL_COLOR, alpha=0.5, label="Real"
    )
    ax.scatter(
        coords[n:, 0], coords[n:, 1], s=12, c=SYNTH_COLOR, alpha=0.5, label="Synthetic"
    )
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    weights_path = args.weights_dir / "generator_wgan.pth"
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(weights_path, map_location=device))
    generator.eval()

    windows, _ = load_price_windows()
    n = min(args.n_samples, windows.shape[0])
    rng = np.random.default_rng(args.seed)
    real = windows[rng.choice(windows.shape[0], n, replace=False)]

    with torch.no_grad():
        z = torch.randn(n, SEQ_LEN, NOISE_DIM, device=device)
        synthetic = generator(z).cpu().numpy()

    combined = np.concatenate([real, synthetic], axis=0)
    pca_coords, explained = pca_analysis(combined)
    tsne_coords = tsne_analysis(combined, seed=args.seed)
    metrics = summary_metrics(real, synthetic)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, pca_coords, n, f"PCA — PC1 {explained[0] * 100:.1f}% var")
    fig.tight_layout()
    fig.savefig(args.output_dir / "pca.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter(ax, tsne_coords, n, "t-SNE (perplexity=30)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "tsne.png", dpi=130)
    plt.close(fig)

    csv_path = args.output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["pca_pc1_explained_variance", f"{explained[0]:.4f}"])
        writer.writerow(["pca_pc2_explained_variance", f"{explained[1]:.4f}"])
        for key, value in metrics.items():
            writer.writerow([key, f"{value:.4f}"])

    print(f"PCA PC1 explained variance: {explained[0] * 100:.1f}%")
    print(
        f"Step-to-step std  real: {metrics['step_to_step_std_real']:.4f}  "
        f"synthetic: {metrics['step_to_step_std_synthetic']:.4f}  "
        f"ratio: {metrics['step_to_step_ratio']:.2f}x"
    )
    print(f"Wrote figures + metrics to {args.output_dir}/")


if __name__ == "__main__":
    main()
