from __future__ import annotations

from agentic_mm_rag.observation.policy import ObservationAction, propose_observation_actions
from agentic_mm_rag.observation.states import ObservationState
from agentic_mm_rag.observation.units import ObservationUnit


class ObservationActionGenerator:
    """Rule-based first-pass generator for risk-reducing observation actions."""

    def propose(
        self,
        *,
        unit: ObservationUnit,
        state: ObservationState,
        current_risk: float,
    ) -> list[ObservationAction]:
        return propose_observation_actions(unit=unit, state=state, current_risk=current_risk)
