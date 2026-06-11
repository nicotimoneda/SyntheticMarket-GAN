"""Train the WGAN-GP on AAPL and save the generator weights.

Examples
--------
    python scripts/train.py                          # paper run: 200 epochs, seed 42
    python scripts/train.py --epochs 2 --seed 42     # smoke test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from syntheticmarket.data.loader import load_price_windows
from syntheticmarket.training.trainer import TrainConfig, train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the WGAN-GP on AAPL price windows."
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save generator_wgan.pth.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Torch device (cpu/cuda)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading AAPL windows... (seed={args.seed}, epochs={args.epochs})")
    windows, _ = load_price_windows()
    print(f"Loaded {windows.shape[0]} windows of shape {windows.shape[1:]}.")

    config = TrainConfig(num_epochs=args.epochs)
    train(
        windows,
        config=config,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
