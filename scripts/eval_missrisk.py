from __future__ import annotations

import argparse
from pathlib import Path

from common import ensure_parent, not_implemented_yet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate miss-risk calibration and policies.")
    parser.add_argument("--test-path", type=Path, default=Path("data_missrisk/processed/missrisk_test.jsonl"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/figures/missrisk_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_parent(args.report_path)
    print(f"Test: {args.test_path}")
    print(f"Report: {args.report_path}")
    not_implemented_yet("MissRisk evaluation")


if __name__ == "__main__":
    main()
