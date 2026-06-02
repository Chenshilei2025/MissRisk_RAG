from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_mm_rag.observation.obligations import SearchObligation
from agentic_mm_rag.observation.states import ObservationState
from agentic_mm_rag.observation.units import ObservationUnit


class RiskFeatures(BaseModel):
    """Serializable input bundle for miss-risk models."""

    question_id: str
    question: str
    obligation: SearchObligation
    unit: ObservationUnit
    state: ObservationState


class MissRiskEstimate(BaseModel):
    """Predictions from answer-bearing, detectability, and joint miss-risk heads."""

    question_id: str
    obligation_id: str
    unit_id: str
    state_id: str
    p_answer_bearing: float = Field(ge=0.0, le=1.0)
    p_detectable_given_bearing: float = Field(ge=0.0, le=1.0)
    p_joint_miss: float = Field(ge=0.0, le=1.0)


def joint_miss_label(label_answer_bearing: int, label_detectable: int) -> int:
    """Return 1 iff answer-bearing evidence exists but remains undetected."""

    if label_answer_bearing not in {0, 1}:
        raise ValueError("label_answer_bearing must be 0 or 1")
    if label_detectable not in {0, 1}:
        raise ValueError("label_detectable must be 0 or 1")
    return int(label_answer_bearing == 1 and label_detectable == 0)
