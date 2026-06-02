from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentic_mm_rag.observation.states import ObservationChannel
from agentic_mm_rag.observation.units import ObservationUnit


class ObservationPayload(BaseModel):
    """Raw or derived content obtained by observing a source unit."""

    unit_id: str
    channel: ObservationChannel
    content: dict[str, Any] = Field(default_factory=dict)
    quality: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceStore(Protocol):
    """Adapter-facing interface for reading original multimodal source content."""

    def observe(
        self,
        unit: ObservationUnit,
        channel: ObservationChannel,
        *,
        options: dict[str, Any] | None = None,
    ) -> ObservationPayload:
        """Return content for one observation channel of a unit."""
