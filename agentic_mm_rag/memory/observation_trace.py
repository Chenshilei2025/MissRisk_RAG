from __future__ import annotations

from agentic_mm_rag.runtime.trace import ObservationTrace


class TraceStore:
    """Append-only in-memory trace store for early experiments."""

    def __init__(self) -> None:
        self._traces: list[ObservationTrace] = []

    def append(self, trace: ObservationTrace) -> None:
        self._traces.append(trace)

    def list(self) -> list[ObservationTrace]:
        return list(self._traces)
