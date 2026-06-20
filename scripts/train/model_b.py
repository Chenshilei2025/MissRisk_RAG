from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from scripts.lib.missrisk_common import (
    INPUT_PROFILES,
    MissRiskDataset,
    brier,
    classification_metrics,
    ece,
    fit_temperature,
    make_collate,
    pos_weight_from_labels,
    row_metadata,
    set_seed,
    sigmoid_np,
    state_feat_dim_for_profile,
    summarize_rows,
)
from scripts.lib.missrisk_models import DetectabilityModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model B: conditional detectability.")
    parser.add_argument("--train", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_bc_pilot/detectability_train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_bc_pilot/detectability_dev.jsonl"))
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=Path, default=Path("outputs/models/detectability_deberta"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-state-feats", action="store_true")
    parser.add_argument(
        "--input-profile",
        choices=INPUT_PROFILES,
        default="full",
        help=(
            "Input ablation profile. Use no_content/state_only for shortcut probes; "
            "use no_state to remove explicit state/channel hints."
        ),
    )
    parser.add_argument("--include-unit-content", action="store_true", help="Ablation only; may leak full evidence.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def forward_batch(
    model: DetectabilityModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    use_state_feats: bool,
) -> torch.Tensor:
    state_feats = batch.get("state_feats")
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        state_feats.to(device) if use_state_feats and state_feats is not None else None,
    )


def run_eval(
    model: DetectabilityModel,
    loader: DataLoader,
    dataset: MissRiskDataset,
    device: torch.device,
    use_state_feats: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    labels_all = []
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = forward_batch(model, batch, device, use_state_feats)
            logits_all.extend(logits.detach().float().cpu().tolist())
            labels_all.extend(batch["y_detect"].detach().float().cpu().tolist())
            for index in batch["row_index"].detach().cpu().tolist():
                rows.append(dataset.rows[int(index)])
    logits = np.asarray(logits_all, dtype=float)
    labels = np.asarray(labels_all, dtype=float)
    probs = sigmoid_np(logits)
    metrics = classification_metrics(labels, probs)
    predictions = [
        {
            **row_metadata(row),
            "logit_detectable": float(logit),
            "p_detectable_given_bearing": float(prob),
            "y_detect": int(label),
        }
        for row, logit, prob, label in zip(rows, logits, probs, labels, strict=True)
    ]
    return metrics, predictions, logits, labels


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_state_feats = not args.no_state_feats
    n_state_feats = state_feat_dim_for_profile(args.input_profile) if use_state_feats else 0
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    collate = make_collate(tokenizer, use_state_feats=use_state_feats)
    train_ds = MissRiskDataset(
        args.train,
        tokenizer,
        args.max_len,
        oracle_only=True,
        use_state_feats=use_state_feats,
        include_unit_content=args.include_unit_content,
        input_profile=args.input_profile,
        limit=args.max_train_samples,
    )
    dev_ds = MissRiskDataset(
        args.dev,
        tokenizer,
        args.max_len,
        oracle_only=True,
        use_state_feats=use_state_feats,
        include_unit_content=args.include_unit_content,
        input_profile=args.input_profile,
        limit=args.max_dev_samples,
    )
    save_json(
        args.out / "data_audit.json",
        {
            "train": str(args.train),
            "dev": str(args.dev),
            "train_summary": summarize_rows(train_ds.rows),
            "dev_summary": summarize_rows(dev_ds.rows),
            "include_unit_content": args.include_unit_content,
            "input_profile": args.input_profile,
            "use_state_feats": use_state_feats,
            "state_feat_dim": n_state_feats,
        },
    )
    if args.dry_run:
        print(json.dumps(summarize_rows(train_ds.rows), indent=2, sort_keys=True))
        return

    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=collate)
    dev_dl = DataLoader(dev_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
    model = DetectabilityModel(args.encoder, n_state_feats, args.dropout).to(device)
    labels = [float(row.get("label_detectable", 0) or 0) for row in train_ds.rows]
    pos_weight = torch.tensor([pos_weight_from_labels(labels)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_dl) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )
    best_auroc = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, start=1):
            logits = forward_batch(model, batch, device, use_state_feats)
            loss = loss_fn(logits, batch["y_detect"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu())
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "loss": running / step}), flush=True)
        metrics, predictions, _, _ = run_eval(model, dev_dl, dev_ds, device, use_state_feats)
        report = {"epoch": epoch, "train_loss": running / max(1, len(train_dl)), "dev": metrics}
        history.append(report)
        print("[dev] " + json.dumps(report, sort_keys=True), flush=True)
        score = metrics["auroc"]
        if not np.isnan(score) and score > best_auroc:
            best_auroc = float(score)
            torch.save(model.state_dict(), args.out / "model_B.pt")
            save_jsonl(args.out / "dev_predictions.best.jsonl", predictions)

    tokenizer.save_pretrained(args.out)
    save_json(args.out / "metrics.json", {"history": history, "best_dev_auroc": best_auroc, "args": vars(args)})

    model.load_state_dict(torch.load(args.out / "model_B.pt", map_location=device))
    calib_path = args.calib or args.dev
    calib_ds = MissRiskDataset(
        calib_path,
        tokenizer,
        args.max_len,
        oracle_only=True,
        use_state_feats=use_state_feats,
        include_unit_content=args.include_unit_content,
        input_profile=args.input_profile,
    )
    calib_dl = DataLoader(calib_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
    calib_metrics, calib_predictions, logits, labels_np = run_eval(
        model,
        calib_dl,
        calib_ds,
        device,
        use_state_feats,
    )
    temperature = fit_temperature(logits, labels_np)
    calibrated = sigmoid_np(logits / temperature)
    save_json(
        args.out / "temperature.json",
        {
            "temperature": temperature,
            "best_dev_auroc": best_auroc,
            "calib_path": str(calib_path),
            "ece_equal_mass_before": calib_metrics["ece_equal_mass"],
            "ece_equal_mass_after": ece(calibrated, labels_np, scheme="equal_mass"),
            "brier_after": brier(calibrated, labels_np),
        },
    )
    save_jsonl(args.out / "calib_predictions.jsonl", calib_predictions)


if __name__ == "__main__":
    main()
