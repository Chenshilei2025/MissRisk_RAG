from agentic_mm_rag.observation import (
    ObservationAction,
    ObservationState,
    ObservationUnit,
    SearchObligation,
    greedy_risk_reduction,
    joint_miss_label,
    missing_channels,
    propose_observation_actions,
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


def test_action_generator_maps_image_missing_channels() -> None:
    unit = ObservationUnit(
        unit_id="u_img",
        source_id="doc1",
        source_type=SourceType.DOCUMENT,
        modality=Modality.IMAGE,
        raw_content={"caption": "A chart.", "image_path": "chart.png"},
    )
    state = ObservationState(
        unit_id="u_img",
        state_id="image_caption_only",
        observed_channels={ObservationChannel.CAPTION: True},
    )

    actions = propose_observation_actions(unit=unit, state=state, current_risk=0.8)

    assert ObservationChannel.VLM_INSPECTION in missing_channels(unit, state)
    assert {action.action_id for action in actions} >= {"run_ocr", "inspect_source_image"}
    assert greedy_risk_reduction(actions).current_risk == 0.8


def test_action_generator_maps_table_and_video_channels() -> None:
    table_unit = ObservationUnit(
        unit_id="u_table",
        source_id="doc1",
        source_type=SourceType.TABLE,
        modality=Modality.TABLE,
        raw_content={"table_text": "A | B"},
    )
    table_state = ObservationState(
        unit_id="u_table",
        state_id="table_flattened",
        observed_channels={ObservationChannel.TEXT: True},
    )
    table_actions = propose_observation_actions(unit=table_unit, state=table_state, current_risk=0.7)

    video_unit = ObservationUnit(
        unit_id="u_video",
        source_id="vid1",
        source_type=SourceType.VIDEO,
        modality=Modality.VIDEO_SEGMENT,
    )
    video_state = ObservationState(
        unit_id="u_video",
        state_id="sparse_frames",
        observed_channels={ObservationChannel.SPARSE_FRAMES: True},
    )
    video_actions = propose_observation_actions(unit=video_unit, state=video_state, current_risk=0.9)

    assert [action.action_id for action in table_actions] == ["parse_table_structure"]
    assert [action.action_id for action in video_actions] == ["sample_dense_frames"]
