from __future__ import annotations

from pydantic import BaseModel, Field


class Budget(BaseModel):
    """Simple observation budget for one MissRisk run."""

    max_actions: int = Field(default=3, ge=0)
    max_cost: float = Field(default=10.0, ge=0.0)
    actions_used: int = Field(default=0, ge=0)
    cost_used: float = Field(default=0.0, ge=0.0)

    def can_spend(self, cost: float) -> bool:
        return self.actions_used < self.max_actions and self.cost_used + cost <= self.max_cost

    def spend(self, cost: float) -> None:
        if not self.can_spend(cost):
            raise ValueError("observation budget exceeded")
        self.actions_used += 1
        self.cost_used += cost


class MissRiskSession(BaseModel):
    """Runtime state for one question."""

    session_id: str
    question_id: str
    question: str
    budget: Budget = Field(default_factory=Budget)
    metadata: dict[str, str] = Field(default_factory=dict)
