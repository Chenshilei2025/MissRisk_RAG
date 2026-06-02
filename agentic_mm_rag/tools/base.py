from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from agentic_mm_rag.observation.policy import ObservationAction
from agentic_mm_rag.observation.states import ObservationState
from agentic_mm_rag.observation.units import ObservationUnit


class ToolResult(BaseModel):
    """Result of applying an observation tool to a unit."""

    action_id: str
    unit_id: str
    next_state: ObservationState
    content: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationTool(Protocol):
    """Tool that can transform a unit's observation state."""

    tool_name: str

    def can_apply(self, unit: ObservationUnit, state: ObservationState) -> bool:
        """Return whether the tool can observe this unit in this state."""

    def apply(
        self,
        unit: ObservationUnit,
        state: ObservationState,
        action: ObservationAction,
    ) -> ToolResult:
        """Execute the observation action and return the next state."""
