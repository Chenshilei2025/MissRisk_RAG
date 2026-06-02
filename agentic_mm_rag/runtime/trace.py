from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_mm_rag.observation.policy import ObservationAction
from agentic_mm_rag.observation.risk import MissRiskEstimate


class ObservationTrace(BaseModel):
    """Machine-readable trace for case studies and debugging."""

    session_id: str
    selected_actions: list[ObservationAction] = Field(default_factory=list)
    risk_estimates: list[MissRiskEstimate] = Field(default_factory=list)
    notes: list[dict[str, Any]] = Field(default_factory=list)

    def add_risk(self, estimate: MissRiskEstimate) -> None:
        self.risk_estimates.append(estimate)

    def add_action(self, action: ObservationAction) -> None:
        self.selected_actions.append(action)
