from agentic_mm_rag.observation import (
    ObservationAction,
    ObservationState,
    ObservationUnit,
    SearchObligation,
    greedy_risk_reduction,
    joint_miss_label,
)
from agentic_mm_rag.observation.states import ObservationChannel
from agentic_mm_rag.observation.units import Modality, SourceType


def test_joint_miss_label() -> None:
    assert joint_miss_label(1, 0) == 1
    assert joint_miss_label(1, 1) == 0
    assert joint_miss_label(0, 0) == 0


def test_observation_state_channel_lookup() -> None:
    state = ObservationState(
        unit_id="u1",
        state_id="ocr_only",
        observed_channels={ObservationChannel.OCR: True},
    )
    assert state.is_observed(ObservationChannel.OCR)
    assert not state.is_observed(ObservationChannel.SOURCE_IMAGE)


def test_basic_contract_models() -> None:
    unit = ObservationUnit(
        unit_id="u1",
        source_id="doc1",
        source_type=SourceType.DOCUMENT,
        modality=Modality.TEXT,
    )
    obligation = SearchObligation(id="o1", text="Check a value.")
    assert unit.unit_id == "u1"
    assert obligation.required_modalities == []


def test_greedy_risk_reduction() -> None:
    actions = [
        ObservationAction(
            action_id="a1",
            unit_id="u1",
            from_state_id="caption_only",
            to_state_id="vlm",
            cost=2.0,
            current_risk=0.8,
            expected_next_risk=0.4,
        ),
        ObservationAction(
            action_id="a2",
            unit_id="u2",
            from_state_id="text_only",
            to_state_id="full",
            cost=1.0,
            current_risk=0.6,
            expected_next_risk=0.2,
        ),
    ]
    assert greedy_risk_reduction(actions).action_id == "a2"
