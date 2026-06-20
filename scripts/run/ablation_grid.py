from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print nohup commands for MissRisk-RAG pre-submission ablation grid."
    )
    parser.add_argument("--bc-dir", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/models/ablation_grid"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--python", default="/root/autodl-tmp/envs/missrisk-vl/bin/python")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def quote(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def command(name: str, parts: list[str], log_dir: Path) -> str:
    log = log_dir / f"{name}.log"
    return " ".join(["nohup", *parts, ">", quote(log), "2>&1", "&"])


def main() -> None:
    args = parse_args()
    commands: list[str] = [
        f"mkdir -p {quote(args.out_root)} {quote(args.log_dir)}",
    ]
    for profile in ("full", "no_state", "no_content", "state_only"):
        name = f"detectability_{profile}_seed{args.seed}"
        parts = [
            quote(args.python),
            "scripts/train/model_b.py",
            "--train",
            quote(args.bc_dir / "detectability_train.jsonl"),
            "--dev",
            quote(args.bc_dir / "detectability_dev.jsonl"),
            "--calib",
            quote(args.bc_dir / "detectability_dev.jsonl"),
            "--encoder",
            quote(args.encoder),
            "--out",
            quote(args.out_root / name),
            "--epochs",
            str(args.epochs),
            "--bs",
            str(args.bs),
            "--max-len",
            str(args.max_len),
            "--seed",
            str(args.seed),
            "--input-profile",
            profile,
        ]
        commands.append(command(name, parts, args.log_dir))

    c_grid = [
        ("full_all", "full", 0.3, 0.5, None),
        ("full_no_aux", "full", 0.0, 0.0, None),
        ("full_bear_only", "full", 0.3, 0.0, None),
        ("full_detect_only", "full", 0.0, 0.5, None),
        ("full_low_pos_weight", "full", 0.3, 0.5, 1.0),
        ("no_state_all", "no_state", 0.3, 0.5, None),
        ("no_content_all", "no_content", 0.3, 0.5, None),
        ("state_only_all", "state_only", 0.3, 0.5, None),
    ]
    for label, profile, lambda_b, lambda_d, pos_weight in c_grid:
        name = f"missrisk_{label}_seed{args.seed}"
        parts = [
            quote(args.python),
            "scripts/train/model_c.py",
            "--train",
            quote(args.bc_dir / "missrisk_train.jsonl"),
            "--dev",
            quote(args.bc_dir / "missrisk_dev.jsonl"),
            "--calib",
            quote(args.bc_dir / "missrisk_dev.jsonl"),
            "--test",
            quote(args.bc_dir / "missrisk_test.jsonl"),
            "--encoder",
            quote(args.encoder),
            "--out",
            quote(args.out_root / name),
            "--epochs",
            str(args.epochs),
            "--bs",
            str(args.bs),
            "--max-len",
            str(args.max_len),
            "--seed",
            str(args.seed),
            "--input-profile",
            profile,
            "--lambda-b",
            str(lambda_b),
            "--lambda-d",
            str(lambda_d),
        ]
        if pos_weight is not None:
            parts.extend(["--miss-pos-weight", str(pos_weight)])
        commands.append(command(name, parts, args.log_dir))

    print("\n".join(commands))


if __name__ == "__main__":
    main()
