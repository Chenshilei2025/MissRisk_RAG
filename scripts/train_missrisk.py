from __future__ import annotations

import argparse
from pathlib import Path

from common import not_implemented_yet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model C: joint miss-risk estimator.")
    parser.add_argument("--train-path", type=Path, default=Path("data_missrisk/processed/missrisk_train.jsonl"))
    parser.add_argument("--dev-path", type=Path, default=Path("data_missrisk/processed/missrisk_dev.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/missrisk"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Train: {args.train_path}")
    print(f"Dev: {args.dev_path}")
    print(f"Output: {args.output_dir}")
    not_implemented_yet("Joint miss-risk model training")


if __name__ == "__main__":
    main()
