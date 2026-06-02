from __future__ import annotations

from pathlib import Path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def not_implemented_yet(task: str) -> None:
    print(f"{task} is scaffolded. Fill in dataset-specific logic next.")
