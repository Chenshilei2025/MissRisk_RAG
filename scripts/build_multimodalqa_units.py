from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ensure_parent, not_implemented_yet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MultiModalQA examples into MissRisk units.")
    parser.add_argument("--input-path", type=Path, default=Path("data_missrisk/raw/multimodalqa"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data_missrisk/processed/multimodalqa_units.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_parent(args.output_path)
    print(f"Input: {args.input_path}")
    print(f"Output: {args.output_path}")
    not_implemented_yet("MultiModalQA unit conversion")


if __name__ == "__main__":
    main()
