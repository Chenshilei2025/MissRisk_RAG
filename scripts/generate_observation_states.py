from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_parent, not_implemented_yet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled observation states.")
    parser.add_argument("--units-path", type=Path, default=Path("data_missrisk/processed/units.jsonl"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data_missrisk/processed/detectability_states.jsonl"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_parent(args.output_path)
    print(f"Units: {args.units_path}")
    print(f"Output: {args.output_path}")
    not_implemented_yet("Observation state generation")


if __name__ == "__main__":
    main()
