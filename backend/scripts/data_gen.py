from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.ml.data import augment_samples, generate_base_samples, split_base_samples
from app.ml.schema import (
    ARTIFACTS_DIR,
    DEFAULT_2X2_BASE_DATASET,
    DEFAULT_2X3_BASE_DATASET,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, default=ARTIFACTS_DIR / "graphs_train.pt")
    parser.add_argument("--test-output", type=Path, default=ARTIFACTS_DIR / "graphs_test.pt")
    parser.add_argument("--two-by-two-output", type=Path, default=DEFAULT_2X2_BASE_DATASET)
    parser.add_argument("--two-by-three-output", type=Path, default=DEFAULT_2X3_BASE_DATASET)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    max_nodes = 6
    base_2x2 = generate_base_samples(Lx=2, Ly=2, max_nodes=max_nodes)
    base_2x3 = generate_base_samples(Lx=2, Ly=3, max_nodes=max_nodes)
    base_samples = base_2x2 + base_2x3
    train_base, test_base = split_base_samples(base_samples)

    train_samples = augment_samples(train_base)
    test_samples = augment_samples(test_base)

    torch.save(train_samples, args.train_output)
    torch.save(test_samples, args.test_output)
    torch.save(base_2x2, args.two_by_two_output)
    torch.save(base_2x3, args.two_by_three_output)

    print(f"saved {len(train_samples)} train samples to {args.train_output}")
    print(f"saved {len(test_samples)} test samples to {args.test_output}")
    print(f"saved {len(base_2x2)} base 2x2 samples to {args.two_by_two_output}")
    print(f"saved {len(base_2x3)} base 2x3 samples to {args.two_by_three_output}")


if __name__ == "__main__":
    main()
