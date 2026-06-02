from __future__ import annotations

from agentic_mm_rag.observation.policy import ObservationAction, greedy_risk_reduction


class ObservationController:
    """Select observation actions using deterministic miss-risk reduction."""

    def select_action(self, actions: list[ObservationAction]) -> ObservationAction | None:
        return greedy_risk_reduction(actions)
