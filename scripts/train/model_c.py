from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
    safe_auroc,
    set_seed,
    sigmoid_np,
    state_feat_dim_for_profile,
    summarize_rows,
)
from scripts.lib.missrisk_models import DetectabilityModel, JointMissRiskModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Model C: joint residual miss-risk estimator."
    )
    parser.add_argument("--train", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_bc_pilot/missrisk_train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data_missrisk/processed/mmdocrag_model_bc_pilot/missrisk_dev.jsonl"))
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=Path, default=Path("outputs/models/missrisk_deberta"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-b", type=float, default=0.3)
    parser.add_argument("--lambda-d", type=float, default=0.5)
    parser.add_argument(
        "--miss-pos-weight",
        type=float,
        default=None,
        help="Override positive class weight for the joint miss loss. Default uses neg/pos.",
    )
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
    parser.add_argument(
        "--include-unit-content",
        action="store_true",
        help="Ablation only: include full unit_content. This may leak oracle content for partial states.",
    )
    parser.add_argument("--warm-start-b", type=Path, default=None)
    parser.add_argument("--no-copy-b-encoder", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def forward_batch(
    model: JointMissRiskModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    use_state_feats: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_feats = batch.get("state_feats")
    return model(
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        state_feats.to(device) if use_state_feats and state_feats is not None else None,
    )


def multitask_loss(
    bear_logits: torch.Tensor,
    detect_logits: torch.Tensor,
    miss_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    miss_pos_weight: torch.Tensor,
    lambda_b: float,
    lambda_d: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_bear = batch["y_bear"].to(device)
    y_detect = batch["y_detect"].to(device)
    y_miss = batch["y_miss"].to(device)
    detect_mask = batch["detect_mask"].to(device)
    loss_miss = F.binary_cross_entropy_with_logits(
        miss_logits,
        y_miss,
        pos_weight=miss_pos_weight,
    )
    loss_bear = F.binary_cross_entropy_with_logits(bear_logits, y_bear)
    detect_per_row = F.binary_cross_entropy_with_logits(
        detect_logits,
        y_detect,
        reduction="none",
    )
    loss_detect = (detect_per_row * detect_mask).sum() / detect_mask.sum().clamp_min(1.0)
    total = loss_miss + lambda_b * loss_bear + lambda_d * loss_detect
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_miss": float(loss_miss.detach().cpu()),
        "loss_bear": float(loss_bear.detach().cpu()),
        "loss_detect": float(loss_detect.detach().cpu()),
    }


def run_eval(
    model: JointMissRiskModel,
    loader: DataLoader,
    dataset: MissRiskDataset,
    device: torch.device,
    use_state_feats: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[float]] = {
        "bear_logits": [],
        "detect_logits": [],
        "miss_logits": [],
        "y_bear": [],
        "y_detect": [],
        "y_miss": [],
        "detect_mask": [],
    }
    with torch.no_grad():
        for batch in loader:
            bear_logits, detect_logits, miss_logits = forward_batch(
                model,
                batch,
                device,
                use_state_feats,
            )
            for key, tensor in (
                ("bear_logits", bear_logits),
                ("detect_logits", detect_logits),
                ("miss_logits", miss_logits),
                ("y_bear", batch["y_bear"]),
                ("y_detect", batch["y_detect"]),
                ("y_miss", batch["y_miss"]),
                ("detect_mask", batch["detect_mask"]),
            ):
                arrays[key].extend(tensor.detach().float().cpu().tolist())
            for index in batch["row_index"].detach().cpu().tolist():
                rows.append(dataset.rows[int(index)])
    arr = {key: np.asarray(value, dtype=float) for key, value in arrays.items()}
    p_bear = sigmoid_np(arr["bear_logits"])
    p_detect = sigmoid_np(arr["detect_logits"])
    p_miss = sigmoid_np(arr["miss_logits"])
    detect_valid = arr["detect_mask"] == 1
    metrics: dict[str, Any] = {
        "miss": classification_metrics(arr["y_miss"], p_miss),
        "bear": classification_metrics(arr["y_bear"], p_bear),
        "detect": classification_metrics(arr["y_detect"][detect_valid], p_detect[detect_valid])
        if detect_valid.any()
        else {"count": 0},
        "product_ablation": {
            "auroc": safe_auroc(arr["y_miss"], p_bear * (1.0 - p_detect)),
            "brier": brier(p_bear * (1.0 - p_detect), arr["y_miss"]),
            "ece_equal_mass": ece(p_bear * (1.0 - p_detect), arr["y_miss"], scheme="equal_mass"),
        },
    }
    predictions = []
    for row, bear_logit, detect_logit, miss_logit, y_bear, y_detect, y_miss, mask in zip(
        rows,
        arr["bear_logits"],
        arr["detect_logits"],
        arr["miss_logits"],
        arr["y_bear"],
        arr["y_detect"],
        arr["y_miss"],
        arr["detect_mask"],
        strict=True,
    ):
        p_b = float(1.0 / (1.0 + np.exp(-bear_logit)))
        p_d = float(1.0 / (1.0 + np.exp(-detect_logit)))
        p_m = float(1.0 / (1.0 + np.exp(-miss_logit)))
        predictions.append(
            {
                **row_metadata(row),
                "logit_answer_bearing": float(bear_logit),
                "logit_detectable": float(detect_logit),
                "logit_joint_miss": float(miss_logit),
                "p_answer_bearing": p_b,
                "p_detectable_given_bearing": p_d,
                "p_joint_miss": p_m,
                "p_product_ablation": p_b * (1.0 - p_d),
                "detect_mask": int(mask),
                "y_bear": int(y_bear),
                "y_detect": int(y_detect) if mask else None,
                "y_miss": int(y_miss),
            }
        )
    return metrics, predictions, arr


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    use_state_feats = not args.no_state_feats
    n_state_feats = state_feat_dim_for_profile(args.input_profile) if use_state_feats else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    collate = make_collate(tokenizer, use_state_feats=use_state_feats)
    train_ds = MissRiskDataset(
        args.train,
        tokenizer,
        args.max_len,
        use_state_feats=use_state_feats,
        include_unit_content=args.include_unit_content,
        input_profile=args.input_profile,
        limit=args.max_train_samples,
    )
    dev_ds = MissRiskDataset(
        args.dev,
        tokenizer,
        args.max_len,
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
    model = JointMissRiskModel(args.encoder, n_state_feats, args.dropout).to(device)
    if args.warm_start_b:
        b_model = DetectabilityModel(args.encoder, n_state_feats, args.dropout)
        b_model.load_state_dict(torch.load(args.warm_start_b, map_location="cpu"))
        model.warm_start_from_detectability(b_model, copy_encoder=not args.no_copy_b_encoder)
        print("[init] warm-started detect components from Model B")

    learned_pos_weight = pos_weight_from_labels(
        [float(row.get("label_joint_miss", 0) or 0) for row in train_ds.rows]
    )
    miss_pos_weight_value = learned_pos_weight if args.miss_pos_weight is None else args.miss_pos_weight
    miss_pos_weight = torch.tensor([miss_pos_weight_value], device=device)
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
        loss_totals = {"loss": 0.0, "loss_miss": 0.0, "loss_bear": 0.0, "loss_detect": 0.0}
        for step, batch in enumerate(train_dl, start=1):
            bear_logits, detect_logits, miss_logits = forward_batch(model, batch, device, use_state_feats)
            loss, loss_parts = multitask_loss(
                bear_logits,
                detect_logits,
                miss_logits,
                batch,
                device,
                miss_pos_weight,
                args.lambda_b,
                args.lambda_d,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for key in loss_totals:
                loss_totals[key] += loss_parts[key]
            if step % 100 == 0:
                print(json.dumps({"epoch": epoch, "step": step, **{k: v / step for k, v in loss_totals.items()}}), flush=True)

        metrics, predictions, _ = run_eval(model, dev_dl, dev_ds, device, use_state_feats)
        epoch_report = {
            "epoch": epoch,
            "train_loss": {key: value / max(1, len(train_dl)) for key, value in loss_totals.items()},
            "dev": metrics,
        }
        history.append(epoch_report)
        print("[dev] " + json.dumps(epoch_report, sort_keys=True), flush=True)
        score = metrics["miss"]["auroc"]
        if not np.isnan(score) and score > best_auroc:
            best_auroc = float(score)
            torch.save(model.state_dict(), args.out / "model_C.pt")
            save_jsonl(args.out / "dev_predictions.best.jsonl", predictions)

    tokenizer.save_pretrained(args.out)
    save_json(args.out / "metrics.json", {"history": history, "best_dev_miss_auroc": best_auroc, "args": vars(args)})

    model.load_state_dict(torch.load(args.out / "model_C.pt", map_location=device))
    calib_path = args.calib or args.dev
    calib_ds = MissRiskDataset(
        calib_path,
        tokenizer,
        args.max_len,
        use_state_feats=use_state_feats,
        include_unit_content=args.include_unit_content,
        input_profile=args.input_profile,
    )
    calib_dl = DataLoader(calib_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
    calib_metrics, calib_predictions, calib_arr = run_eval(model, calib_dl, calib_ds, device, use_state_feats)
    temperature = fit_temperature(calib_arr["miss_logits"], calib_arr["y_miss"])
    calibrated = sigmoid_np(calib_arr["miss_logits"] / temperature)
    save_json(
        args.out / "temperature.json",
        {
            "temperature": temperature,
            "best_dev_miss_auroc": best_auroc,
            "calib_path": str(calib_path),
            "miss_pos_weight": miss_pos_weight_value,
            "auto_miss_pos_weight": learned_pos_weight,
            "miss_ece_equal_mass_before": calib_metrics["miss"]["ece_equal_mass"],
            "miss_ece_equal_mass_after": ece(calibrated, calib_arr["y_miss"], scheme="equal_mass"),
            "miss_brier_after": brier(calibrated, calib_arr["y_miss"]),
        },
    )
    save_jsonl(args.out / "calib_predictions.jsonl", calib_predictions)

    if args.test:
        test_ds = MissRiskDataset(
            args.test,
            tokenizer,
            args.max_len,
            use_state_feats=use_state_feats,
            include_unit_content=args.include_unit_content,
            input_profile=args.input_profile,
        )
        test_dl = DataLoader(test_ds, batch_size=args.bs, shuffle=False, collate_fn=collate)
        test_metrics, test_predictions, _ = run_eval(model, test_dl, test_ds, device, use_state_feats)
        save_json(args.out / "test_metrics.json", test_metrics)
        save_jsonl(args.out / "test_predictions.jsonl", test_predictions)


if __name__ == "__main__":
    main()
