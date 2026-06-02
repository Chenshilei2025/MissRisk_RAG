from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AnswerDecision(StrEnum):
    ANSWER = "answer"
    UNDER_OBSERVED = "under_observed"
    ABSTAIN_TRUE_UNANSWERABLE = "abstain_true_unanswerable"


class AnswerGate:
    """Turn residual risk and support scores into a final action."""

    def __init__(
        self,
        *,
        answer_support_threshold: float = 0.6,
        miss_risk_threshold: float = 0.4,
        true_unanswerable_threshold: float = 0.2,
    ) -> None:
        self.answer_support_threshold = answer_support_threshold
        self.miss_risk_threshold = miss_risk_threshold
        self.true_unanswerable_threshold = true_unanswerable_threshold

    def decide(self, *, answer_support: float, residual_miss_risk: float) -> "GateResult":
        if answer_support >= self.answer_support_threshold and residual_miss_risk < self.miss_risk_threshold:
            return GateResult(decision=AnswerDecision.ANSWER, confidence=answer_support)
        if residual_miss_risk >= self.miss_risk_threshold:
            return GateResult(decision=AnswerDecision.UNDER_OBSERVED, confidence=residual_miss_risk)
        if residual_miss_risk <= self.true_unanswerable_threshold:
            return GateResult(
                decision=AnswerDecision.ABSTAIN_TRUE_UNANSWERABLE,
                confidence=1.0 - residual_miss_risk,
            )
        return GateResult(decision=AnswerDecision.UNDER_OBSERVED, confidence=residual_miss_risk)


class GateResult(BaseModel):
    decision: AnswerDecision
    confidence: float = Field(ge=0.0, le=1.0)
