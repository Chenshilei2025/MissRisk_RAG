from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_parent, not_implemented_yet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label detectability for observation states.")
    parser.add_argument(
        "--states-path",
        type=Path,
        default=Path("data_missrisk/processed/detectability_states.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data_missrisk/processed/missrisk_labeled.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_parent(args.output_path)
    print(f"States: {args.states_path}")
    print(f"Output: {args.output_path}")
    not_implemented_yet("Detectability labeling")


if __name__ == "__main__":
    main()
