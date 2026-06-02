from agentic_mm_rag.controller.answer_gate import AnswerDecision, AnswerGate
from agentic_mm_rag.observation.policy import ObservationAction
from agentic_mm_rag.observation.risk import MissRiskEstimate
from agentic_mm_rag.runtime import InMemoryEventSink, MissRiskRunner, MissRiskSession


def make_estimate(risk: float) -> MissRiskEstimate:
    return MissRiskEstimate(
        question_id="q1",
        obligation_id="o1",
        unit_id="u1",
        state_id="caption_only",
        p_answer_bearing=0.9,
        p_detectable_given_bearing=0.2,
        p_joint_miss=risk,
    )


def test_answer_gate_marks_under_observed_when_risk_is_high() -> None:
    gate = AnswerGate(miss_risk_threshold=0.4)
    result = gate.decide(answer_support=0.8, residual_miss_risk=0.7)
    assert result.decision == AnswerDecision.UNDER_OBSERVED


def test_runner_emits_events_and_spends_budget() -> None:
    risks = iter([make_estimate(0.8), make_estimate(0.2)])
    sink = InMemoryEventSink()
    runner = MissRiskRunner(
        risk_scorer=lambda _question: next(risks),
        action_proposer=lambda estimate: [
            ObservationAction(
                action_id="inspect_source_image",
                unit_id=estimate.unit_id,
                from_state_id=estimate.state_id,
                to_state_id="source_image_vlm",
                cost=2.0,
                current_risk=estimate.p_joint_miss,
                expected_next_risk=0.3,
            )
        ],
        event_sink=sink,
    )

    result = runner.run(
        MissRiskSession(session_id="s1", question_id="q1", question="What value is shown?"),
        answer_support=0.8,
    )

    assert result.session.budget.actions_used == 1
    assert result.trace.selected_actions[0].action_id == "inspect_source_image"
    assert result.gate_result.decision == AnswerDecision.ANSWER
    assert [event.event_type for event in sink.events] == [
        "run_started",
        "risk_estimated",
        "observation_action_selected",
        "answer_gate_decision",
    ]
