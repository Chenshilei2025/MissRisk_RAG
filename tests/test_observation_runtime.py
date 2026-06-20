from __future__ import annotations

from agentic_mm_rag.controller.answer_gate import AnswerDecision
from agentic_mm_rag.observation import (
    MissRiskEstimate,
    ObservationState,
    ObservationUnit,
    joint_miss_label,
    propose_actions_from_estimate,
)
from agentic_mm_rag.observation.states import ObservationChannel
from agentic_mm_rag.observation.units import Modality, SourceType
from agentic_mm_rag.runtime import Budget, MissRiskRunner, MissRiskSession


def test_observation_unit_backfills_typed_views() -> None:
    unit = ObservationUnit(
        unit_id="u1",
        source_id="doc1",
        source_type=SourceType.DOCUMENT,
        modality=Modality.TABLE,
        raw_content={"table_text": "Year | Value\n2024 | 42", "image_path": "page.png"},
    )

    assert unit.views.text_view is None
    assert unit.views.table_view is not None
    assert unit.views.table_view.table_text == "Year | Value\n2024 | 42"
    assert unit.views.visual_view is not None


def test_joint_miss_label_and_action_proposal() -> None:
    unit = ObservationUnit(
        unit_id="img1",
        source_id="doc1",
        source_type=SourceType.DOCUMENT,
        modality=Modality.IMAGE,
        raw_content={"image_path": "chart.png"},
    )
    state = ObservationState(
        unit_id="img1",
        state_id="caption_only",
        observed_channels={ObservationChannel.CAPTION: True},
        hidden_channels=[ObservationChannel.VLM_INSPECTION],
    )
    estimate = MissRiskEstimate(
        question_id="q1",
        obligation_id="o1",
        unit_id="img1",
        state_id="caption_only",
        p_answer_bearing=0.8,
        p_detectable_given_bearing=0.2,
        p_joint_miss=0.7,
    )

    actions = propose_actions_from_estimate(unit=unit, state=state, estimate=estimate)
    assert joint_miss_label(1, 0) == 1
    assert any(action.action_id == "inspect_source_image" for action in actions)
    assert all(action.current_risk == 0.7 for action in actions)


def test_runner_returns_under_observed_when_risk_remains_high() -> None:
    estimate = MissRiskEstimate(
        question_id="q1",
        obligation_id="o1",
        unit_id="u1",
        state_id="s1",
        p_answer_bearing=0.9,
        p_detectable_given_bearing=0.1,
        p_joint_miss=0.8,
    )
    runner = MissRiskRunner(
        risk_scorer=lambda _: estimate,
        action_proposer=lambda _: [],
    )
    result = runner.run(
        MissRiskSession(
            session_id="s",
            question_id="q1",
            question="Where is the evidence?",
            budget=Budget(max_actions=0),
        ),
        answer_support=0.9,
    )

    assert result.gate_result.decision == AnswerDecision.UNDER_OBSERVED
    assert result.trace.risk_estimates[0].p_joint_miss == 0.8

