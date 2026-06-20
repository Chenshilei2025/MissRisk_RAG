from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


JSON = dict[str, Any]


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path, limit: int | None = None) -> Iterator[JSON]:
    with open_text(path, "rt") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path, limit: int | None = None) -> list[JSON]:
    return list(iter_jsonl(path, limit=limit))


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def stable_id(*parts: Any, length: int = 16) -> str:
    text = "::".join(normalize_space(str(part)) for part in parts if part is not None)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def split_for_key(key: str, train_ratio: float = 0.8, dev_ratio: float = 0.1) -> str:
    bucket = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + dev_ratio:
        return "dev"
    return "test"


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def truncate(text: str, max_chars: int = 6000) -> str:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ... [truncated]"


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
        if value not in (None, "", [], {}):
            return value
    return None


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,}", text.lower()))


def lexical_overlap(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def make_obligation(question_id: str, question: str, modalities: list[str] | None = None) -> JSON:
    required = [str(m).lower() for m in (modalities or []) if str(m).strip()]
    return {
        "id": f"{question_id}:primary",
        "obligation_id": f"{question_id}:primary",
        "text": f"Find evidence that can answer, support, refute, or change the answer to: {question}",
        "required_modalities": sorted(set(required)),
        "source": "rule_question_obligation",
    }


def compact_answer(record: JSON) -> str:
    answers = record.get("answers")
    if isinstance(answers, list):
        values = []
        for item in answers:
            if isinstance(item, Mapping):
                values.append(str(item.get("answer", "")).strip())
            elif item is not None:
                values.append(str(item).strip())
        return "; ".join(v for v in values if v)
    return normalize_space(first_nonempty(record.get("answer_short"), record.get("answer"), ""))


def source_split_summary(rows: Iterable[JSON]) -> dict[str, int]:
    counts = {"train": 0, "dev": 0, "test": 0}
    for row in rows:
        split = str(row.get("split") or row.get("metadata", {}).get("split") or "train")
        counts[split] = counts.get(split, 0) + 1
    return counts


def find_json_inputs(root: Path, names: list[str] | None = None) -> list[Path]:
    if root.is_file():
        return [root]
    candidates: list[Path] = []
    for pattern in ("*.jsonl", "*.jsonl.gz", "*.json"):
        candidates.extend(sorted(root.glob(pattern)))
    if names:
        wanted = set(names)
        candidates = [path for path in candidates if path.name in wanted]
    return candidates
