from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


OBSERVATION_CHANNELS = [
    "metadata",
    "text",
    "ocr",
    "caption",
    "caption_or_ocr",
    "table_text",
    "table_structure",
    "source_image",
    "vlm_inspection",
    "layout_context",
    "neighbor_units",
    "sparse_frames",
    "dense_frames",
    "graph",
    "source_chunks",
]

QUALITY_KEYS = ["ocr", "caption", "visual", "table", "source_image", "vlm", "layout"]

INPUT_PROFILES = ("full", "no_state", "no_content", "state_only")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def compact_mapping(mapping: dict[str, Any] | None, keys: list[str]) -> str:
    if not isinstance(mapping, dict):
        return ""
    pieces = []
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            pieces.append(f"{key}={normalize_space(value)}")
    return " ".join(pieces)


def row_quality(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("state_quality") or row.get("quality") or {}
    return value if isinstance(value, dict) else {}


def state_features(row: dict[str, Any], *, input_profile: str = "full") -> list[float]:
    visible = set(as_list(row.get("visible_channels")))
    hidden = set(as_list(row.get("hidden_channels")))
    quality = row_quality(row)
    feats: list[float] = []
    if input_profile != "no_state":
        feats.extend(1.0 if channel in visible else 0.0 for channel in OBSERVATION_CHANNELS)
        feats.extend(1.0 if channel in hidden else 0.0 for channel in OBSERVATION_CHANNELS)
    feats.extend(float(quality.get(key, 0.0) or 0.0) for key in QUALITY_KEYS)
    return feats


STATE_FEAT_DIM = len(OBSERVATION_CHANNELS) * 2 + len(QUALITY_KEYS)
QUALITY_FEAT_DIM = len(QUALITY_KEYS)


def state_feat_dim_for_profile(input_profile: str) -> int:
    if input_profile == "no_state":
        return QUALITY_FEAT_DIM
    return STATE_FEAT_DIM


def build_input_text(
    row: dict[str, Any],
    *,
    include_unit_content: bool = False,
    input_profile: str = "full",
) -> str:
    """Build the cross-encoder input without leaking full oracle content by default.

    For partial observation states, `unit_content` may contain the full quote/table
    content. Feeding it would make detectability trivial, so the default input is
    restricted to the current visible state plus state/channel metadata.
    """

    if input_profile not in INPUT_PROFILES:
        raise ValueError(f"Unknown input_profile={input_profile!r}; expected one of {INPUT_PROFILES}")
    visible_channels = " ".join(as_list(row.get("visible_channels")))
    hidden_channels = " ".join(as_list(row.get("hidden_channels")))
    locator = row.get("locator") if isinstance(row.get("locator"), dict) else {}
    unit_meta = {
        "dataset": row.get("dataset"),
        "source_id": row.get("source_id"),
        "modality": row.get("modality"),
        "page_id": row.get("page_id") or locator.get("page_id"),
        "quote_id": row.get("quote_id") or locator.get("quote_id"),
    }
    if input_profile != "no_state":
        unit_meta["state_id"] = row.get("state_id")

    parts = [
        f"[UNIT_META] {compact_mapping(unit_meta, list(unit_meta.keys()))}",
        f"[STATE_QUALITY] {compact_mapping(row_quality(row), QUALITY_KEYS)}",
    ]
    if input_profile != "state_only":
        parts[:0] = [
            f"[QUESTION] {normalize_space(row.get('question'))}",
            f"[OBLIGATION] {normalize_space(row.get('obligation_text'))}",
        ]
    if input_profile != "no_state":
        parts.append(f"[VISIBLE_CHANNELS] {visible_channels}")
        parts.append(f"[HIDDEN_CHANNELS] {hidden_channels}")
    if input_profile not in {"no_content", "state_only"}:
        parts.append(f"[VISIBLE] {normalize_space(row.get('visible_content'))}")
    if include_unit_content:
        parts.append(f"[UNIT_CONTENT_ABLATION] {normalize_space(row.get('unit_content'))}")
    return "\n".join(parts)


class MissRiskDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_len: int = 384,
        *,
        oracle_only: bool = False,
        use_state_feats: bool = True,
        include_unit_content: bool = False,
        input_profile: str = "full",
        limit: int | None = None,
    ) -> None:
        if input_profile not in INPUT_PROFILES:
            raise ValueError(f"Unknown input_profile={input_profile!r}; expected one of {INPUT_PROFILES}")
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if oracle_only and int(row.get("label_answer_bearing", 0)) != 1:
                    continue
                self.rows.append(row)
                if limit is not None and len(self.rows) >= limit:
                    break
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_state_feats = use_state_feats
        self.include_unit_content = include_unit_content
        self.input_profile = input_profile

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        encoded = self.tokenizer(
            build_input_text(
                row,
                include_unit_content=self.include_unit_content,
                input_profile=self.input_profile,
            ),
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
        )
        y_bear = float(row.get("label_answer_bearing", 0) or 0)
        label_detectable = row.get("label_detectable")
        detect_mask = float(label_detectable is not None and y_bear == 1.0)
        item = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "y_bear": y_bear,
            "y_detect": float(label_detectable) if label_detectable is not None else 0.0,
            "detect_mask": detect_mask,
            "y_miss": float(row.get("label_joint_miss", 0) or 0),
            "row_index": float(index),
        }
        if self.use_state_feats:
            item["state_feats"] = state_features(row, input_profile=self.input_profile)
        return item


def make_collate(tokenizer: Any, *, use_state_feats: bool = True):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for row_idx, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row_idx, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[row_idx, :length] = torch.tensor(item["attention_mask"], dtype=torch.long)
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "y_bear": torch.tensor([item["y_bear"] for item in batch], dtype=torch.float),
            "y_detect": torch.tensor([item["y_detect"] for item in batch], dtype=torch.float),
            "detect_mask": torch.tensor([item["detect_mask"] for item in batch], dtype=torch.float),
            "y_miss": torch.tensor([item["y_miss"] for item in batch], dtype=torch.float),
            "row_index": torch.tensor([item["row_index"] for item in batch], dtype=torch.long),
        }
        if use_state_feats:
            output["state_feats"] = torch.tensor(
                [item["state_feats"] for item in batch],
                dtype=torch.float,
            )
        return output

    return collate


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def safe_auroc(labels: list[float] | np.ndarray, probs: list[float] | np.ndarray) -> float:
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probs))
    except Exception:
        order = np.argsort(probs)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(probs), dtype=float) + 1
        pos = labels == 1
        n_pos = float(pos.sum())
        n_neg = float((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(probs: list[float] | np.ndarray, labels: list[float] | np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(labels) == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def ece(
    probs: list[float] | np.ndarray,
    labels: list[float] | np.ndarray,
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(labels) == 0:
        return float("nan")
    if scheme == "equal_mass":
        edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    bucket_ids = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for bucket in range(n_bins):
        mask = bucket_ids == bucket
        if not mask.any():
            continue
        total += mask.mean() * abs(float(labels[mask].mean()) - float(probs[mask].mean()))
    return float(total)


def classification_metrics(
    labels: list[float] | np.ndarray,
    probs: list[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "accuracy": float((tp + tn) / len(labels)) if len(labels) else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auroc": safe_auroc(labels, probs),
        "brier": brier(probs, labels),
        "ece_equal_width": ece(probs, labels, scheme="equal_width"),
        "ece_equal_mass": ece(probs, labels, scheme="equal_mass"),
    }


def fit_temperature(logits: list[float] | np.ndarray, labels: list[float] | np.ndarray) -> float:
    logits_tensor = torch.tensor(np.asarray(logits), dtype=torch.float)
    labels_tensor = torch.tensor(np.asarray(labels), dtype=torch.float)
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=200)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits_tensor / temperature.clamp_min(1e-3), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp_min(1e-3).item())


def pos_weight_from_labels(labels: list[float] | np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    if pos <= 0:
        return 1.0
    return neg / pos


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": row.get("example_id"),
        "dataset": row.get("dataset"),
        "question_id": row.get("question_id"),
        "obligation_id": row.get("obligation_id"),
        "unit_id": row.get("unit_id"),
        "state_id": row.get("state_id"),
        "modality": row.get("modality"),
        "label_answer_bearing": row.get("label_answer_bearing"),
        "label_detectable": row.get("label_detectable"),
        "label_joint_miss": row.get("label_joint_miss"),
        "loss_mask_detect": row.get("loss_mask_detect"),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "label_answer_bearing": dict(Counter(str(row.get("label_answer_bearing")) for row in rows)),
        "label_detectable": dict(Counter(str(row.get("label_detectable")) for row in rows)),
        "label_joint_miss": dict(Counter(str(row.get("label_joint_miss")) for row in rows)),
        "state_id": dict(Counter(str(row.get("state_id")) for row in rows)),
        "modality": dict(Counter(str(row.get("modality")) for row in rows)),
        "dataset": dict(Counter(str(row.get("dataset")) for row in rows)),
    }
