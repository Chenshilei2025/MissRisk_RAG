from __future__ import annotations

from agentic_mm_rag.tools.base import ToolResult


class ObservationCache:
    """In-memory cache for deterministic or expensive observation tool results."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], ToolResult] = {}

    def get(self, unit_id: str, action_id: str) -> ToolResult | None:
        return self._items.get((unit_id, action_id))

    def set(self, result: ToolResult) -> None:
        self._items[(result.unit_id, result.action_id)] = result
