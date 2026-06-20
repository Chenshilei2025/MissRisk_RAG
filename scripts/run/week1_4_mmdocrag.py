from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print commands for the strict Week1-4 MMDocRAG pipeline.")
    parser.add_argument("--python", default="/root/autodl-tmp/envs/missrisk-vl/bin/python")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--data-root", type=Path, default=Path("data_missrisk/processed/mmdocrag_week1_4"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/models/mmdocrag_week1_4"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--epochs-a", type=int, default=3)
    parser.add_argument("--epochs-b", type=int, default=3)
    parser.add_argument("--epochs-c", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=384)
    return parser.parse_args()


def q(value: object) -> str:
    text = str(value)
    return "'" + text.replace("'", "'\"'\"'") + "'"


def main() -> None:
    args = parse_args()
    model_a_dir = args.data_root / "model_a"
    bc_dir = args.data_root / "model_bc"
    out_a = args.out_root / "answer_bearing_cross_encoder"
    out_b = args.out_root / "detectability_warmup"
    out_c = args.out_root / "joint_missrisk_direct"
    commands = [
        "set -e",
        f"mkdir -p {q(args.data_root)} {q(args.out_root)} {q(args.log_dir)} outputs/eval",
        "echo '[week1] build MMDocRAG pilot units'",
        " ".join(
            [
                q(args.python),
                "scripts/build/mmdocrag.py",
                "--input-dir data_missrisk/raw/mmdocrag",
                "--output-dir",
                q(model_a_dir),
                "--max-examples",
                str(args.max_examples),
                "--max-negatives-per-question 6",
                "--train-ratio 0.7 --calib-ratio 0.1 --dev-ratio 0.1",
            ]
        ),
        "echo '[week2] train Model A cross-encoder'",
        " ".join(
            [
                q(args.python),
                "scripts/train/model_a.py",
                "--train",
                q(model_a_dir / "train_pairs.jsonl"),
                "--dev",
                q(model_a_dir / "dev_pairs.jsonl"),
                "--test",
                q(model_a_dir / "test_pairs.jsonl"),
                "--encoder",
                q(args.encoder),
                "--out",
                q(out_a),
                "--epochs",
                str(args.epochs_a),
                "--bs",
                str(args.bs),
                "--max-len",
                str(args.max_len),
            ]
        ),
        "echo '[week3] generate controlled observation states'",
        " ".join(
            [
                q(args.python),
                "scripts/build/observation_states.py",
                "--model-a-dir",
                q(model_a_dir),
                "--output-dir",
                q(bc_dir),
            ]
        ),
        "echo '[week3] train Model B warm-up'",
        " ".join(
            [
                q(args.python),
                "scripts/train/model_b.py",
                "--train",
                q(bc_dir / "detectability_train.jsonl"),
                "--dev",
                q(bc_dir / "detectability_dev.jsonl"),
                "--calib",
                q(bc_dir / "detectability_calib.jsonl"),
                "--encoder",
                q(args.encoder),
                "--out",
                q(out_b),
                "--epochs",
                str(args.epochs_b),
                "--bs",
                str(args.bs),
                "--max-len",
                str(args.max_len),
                "--input-profile full",
            ]
        ),
        "echo '[week4] train Model C direct miss-risk head'",
        " ".join(
            [
                q(args.python),
                "scripts/train/model_c.py",
                "--train",
                q(bc_dir / "missrisk_train.jsonl"),
                "--dev",
                q(bc_dir / "missrisk_dev.jsonl"),
                "--calib",
                q(bc_dir / "missrisk_calib.jsonl"),
                "--test",
                q(bc_dir / "missrisk_test.jsonl"),
                "--encoder",
                q(args.encoder),
                "--out",
                q(out_c),
                "--epochs",
                str(args.epochs_c),
                "--bs",
                str(args.bs),
                "--max-len",
                str(args.max_len),
                "--lambda-b 0.0 --lambda-d 0.0",
                "--input-profile full",
            ]
        ),
        "echo '[week4] audit source leakage and shortcut baselines'",
        " ".join(
            [
                q(args.python),
                "scripts/eval/audit_submission.py",
                "--model-a-dir",
                q(model_a_dir),
                "--bc-dir",
                q(bc_dir),
                "--model-b-dir",
                q(out_b),
                "--model-c-dir",
                q(out_c),
                "--output outputs/eval/mmdocrag_week1_4_audit.json",
                "> /tmp/mmdocrag_week1_4_audit.stdout.json",
            ]
        ),
        "echo '[done] strict Week1-4 pipeline complete'",
    ]
    print("\n".join(commands))


if __name__ == "__main__":
    main()
