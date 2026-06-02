from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from agentic_mm_rag.controller.answer_gate import AnswerGate, GateResult
from agentic_mm_rag.controller.observer import ObservationController
from agentic_mm_rag.observation.policy import ObservationAction
from agentic_mm_rag.observation.risk import MissRiskEstimate
from agentic_mm_rag.runtime.events import EventSink, InMemoryEventSink, RuntimeEvent
from agentic_mm_rag.runtime.session import MissRiskSession
from agentic_mm_rag.runtime.trace import ObservationTrace


RiskScorer = Callable[[str], MissRiskEstimate]
ActionProposer = Callable[[MissRiskEstimate], list[ObservationAction]]


class MissRiskRunResult(BaseModel):
    session: MissRiskSession
    trace: ObservationTrace
    gate_result: GateResult


class MissRiskRunner:
    """Small execution shell around MissRisk scoring and observation policy."""

    def __init__(
        self,
        *,
        risk_scorer: RiskScorer,
        action_proposer: ActionProposer,
        observer: ObservationController | None = None,
        answer_gate: AnswerGate | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.risk_scorer = risk_scorer
        self.action_proposer = action_proposer
        self.observer = observer or ObservationController()
        self.answer_gate = answer_gate or AnswerGate()
        self.event_sink = event_sink or InMemoryEventSink()

    def run(self, session: MissRiskSession, *, answer_support: float = 0.0) -> MissRiskRunResult:
        trace = ObservationTrace(session_id=session.session_id)
        self.event_sink.emit(RuntimeEvent(event_type="run_started", session_id=session.session_id))

        estimate = self.risk_scorer(session.question)
        trace.add_risk(estimate)
        self.event_sink.emit(
            RuntimeEvent(
                event_type="risk_estimated",
                session_id=session.session_id,
                payload=estimate.model_dump(),
            )
        )

        while session.budget.actions_used < session.budget.max_actions:
            actions = [
                action for action in self.action_proposer(estimate) if session.budget.can_spend(action.cost)
            ]
            action = self.observer.select_action(actions)
            if action is None:
                break

            session.budget.spend(action.cost)
            trace.add_action(action)
            self.event_sink.emit(
                RuntimeEvent(
                    event_type="observation_action_selected",
                    session_id=session.session_id,
                    payload=action.model_dump(),
                )
            )

            estimate = self.risk_scorer(session.question)
            trace.add_risk(estimate)
            if estimate.p_joint_miss < action.expected_next_risk:
                break

        gate_result = self.answer_gate.decide(
            answer_support=answer_support,
            residual_miss_risk=estimate.p_joint_miss,
        )
        self.event_sink.emit(
            RuntimeEvent(
                event_type="answer_gate_decision",
                session_id=session.session_id,
                payload=gate_result.model_dump(),
            )
        )
        return MissRiskRunResult(session=session, trace=trace, gate_result=gate_result)
