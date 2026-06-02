from __future__ import annotations

from collections.abc import Iterable
from pydantic import BaseModel, Field


class ObservationAction(BaseModel):
    """Action that changes a unit's observation state."""

    action_id: str
    unit_id: str
    from_state_id: str
    to_state_id: str
    cost: float = Field(gt=0.0)
    current_risk: float = Field(ge=0.0, le=1.0)
    expected_next_risk: float = Field(ge=0.0, le=1.0)

    @property
    def expected_risk_reduction(self) -> float:
        return max(0.0, self.current_risk - self.expected_next_risk)

    @property
    def utility_per_cost(self) -> float:
        return self.expected_risk_reduction / self.cost


def greedy_risk_reduction(actions: Iterable[ObservationAction]) -> ObservationAction | None:
    """Select the action with the largest expected risk reduction per cost."""

    return max(actions, key=lambda action: action.utility_per_cost, default=None)
