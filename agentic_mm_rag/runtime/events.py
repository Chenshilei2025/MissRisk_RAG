from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from pydantic import BaseModel, Field


class RuntimeEvent(BaseModel):
    """Auditable event emitted by the MissRisk execution layer."""

    event_type: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventSink(Protocol):
    """Destination for runtime events."""

    def emit(self, event: RuntimeEvent) -> None:
        """Record one runtime event."""


class InMemoryEventSink:
    """Small event sink useful for tests, notebooks, and early experiments."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)
