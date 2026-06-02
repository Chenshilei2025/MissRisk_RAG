from __future__ import annotations

from collections.abc import Iterable
from pydantic import BaseModel, Field

from agentic_mm_rag.observation.risk import MissRiskEstimate
from agentic_mm_rag.observation.states import ObservationChannel, ObservationState
from agentic_mm_rag.observation.units import Modality, ObservationUnit


class ObservationAction(BaseModel):
    """Action that changes a unit's observation state."""

    action_id: str
    unit_id: str
    from_state_id: str
    to_state_id: str
    cost: float = Field(gt=0.0)
    current_risk: float = Field(ge=0.0, le=1.0)
    expected_next_risk: float = Field(ge=0.0, le=1.0)

    @property
    def expected_risk_reduction(self) -> float:
        return max(0.0, self.current_risk - self.expected_next_risk)

    @property
    def utility_per_cost(self) -> float:
        return self.expected_risk_reduction / self.cost


class ObservationActionSpec(BaseModel):
    """Rule for proposing one observation-state transition."""

    action_id: str
    missing_channel: ObservationChannel
    to_state_id: str
    cost: float = Field(gt=0.0)
    expected_risk_multiplier: float = Field(ge=0.0, le=1.0)
    modalities: set[Modality] = Field(default_factory=set)

    def applies_to(self, unit: ObservationUnit, state: ObservationState) -> bool:
        if self.modalities and unit.modality not in self.modalities:
            return False
        return self.missing_channel in missing_channels(unit, state)


DEFAULT_ACTION_SPECS: tuple[ObservationActionSpec, ...] = (
    ObservationActionSpec(
        action_id="run_ocr",
        missing_channel=ObservationChannel.OCR,
        to_state_id="ocr_only",
        cost=1.0,
        expected_risk_multiplier=0.70,
        modalities={Modality.IMAGE, Modality.MULTIMODAL},
    ),
    ObservationActionSpec(
        action_id="inspect_source_image",
        missing_channel=ObservationChannel.VLM_INSPECTION,
        to_state_id="source_image_vlm",
        cost=2.0,
        expected_risk_multiplier=0.45,
        modalities={Modality.IMAGE, Modality.MULTIMODAL},
    ),
    ObservationActionSpec(
        action_id="parse_table_structure",
        missing_channel=ObservationChannel.TABLE_STRUCTURE,
        to_state_id="table_structure",
        cost=1.5,
        expected_risk_multiplier=0.55,
        modalities={Modality.TABLE, Modality.MULTIMODAL, Modality.IMAGE},
    ),
    ObservationActionSpec(
        action_id="sample_sparse_frames",
        missing_channel=ObservationChannel.SPARSE_FRAMES,
        to_state_id="sparse_frames",
        cost=2.0,
        expected_risk_multiplier=0.65,
        modalities={Modality.VIDEO_SEGMENT, Modality.FRAME_GROUP},
    ),
    ObservationActionSpec(
        action_id="sample_dense_frames",
        missing_channel=ObservationChannel.DENSE_FRAMES,
        to_state_id="dense_frames",
        cost=4.0,
        expected_risk_multiplier=0.40,
        modalities={Modality.VIDEO_SEGMENT, Modality.FRAME_GROUP},
    ),
    ObservationActionSpec(
        action_id="expand_graph_source_chunks",
        missing_channel=ObservationChannel.SOURCE_CHUNKS,
        to_state_id="graph_plus_source_chunks",
        cost=1.5,
        expected_risk_multiplier=0.60,
        modalities={Modality.GRAPH_RELATION},
    ),
)


def missing_channels(unit: ObservationUnit, state: ObservationState) -> set[ObservationChannel]:
    """Infer observable-but-missing channels for a unit under the current state."""

    channels = set(state.hidden_channels)
    channels.update(_channels_from_unit_content(unit))
    channels.update(_channels_from_modality(unit.modality))
    return {channel for channel in channels if not state.is_observed(channel)}


def propose_observation_actions(
    *,
    unit: ObservationUnit,
    state: ObservationState,
    current_risk: float,
    specs: Iterable[ObservationActionSpec] = DEFAULT_ACTION_SPECS,
) -> list[ObservationAction]:
    """Generate deterministic action candidates from missing observation channels."""

    actions: list[ObservationAction] = []
    seen: set[str] = set()
    for spec in specs:
        if not spec.applies_to(unit, state):
            continue
        action_key = f"{spec.action_id}:{unit.unit_id}"
        if action_key in seen:
            continue
        seen.add(action_key)
        actions.append(
            ObservationAction(
                action_id=spec.action_id,
                unit_id=unit.unit_id,
                from_state_id=state.state_id,
                to_state_id=spec.to_state_id,
                cost=spec.cost,
                current_risk=current_risk,
                expected_next_risk=max(0.0, min(1.0, current_risk * spec.expected_risk_multiplier)),
            )
        )
    return actions


def propose_actions_from_estimate(
    *,
    unit: ObservationUnit,
    state: ObservationState,
    estimate: MissRiskEstimate,
    specs: Iterable[ObservationActionSpec] = DEFAULT_ACTION_SPECS,
) -> list[ObservationAction]:
    """Generate actions using `p_joint_miss` as the current residual risk."""

    return propose_observation_actions(
        unit=unit,
        state=state,
        current_risk=estimate.p_joint_miss,
        specs=specs,
    )


def greedy_risk_reduction(actions: Iterable[ObservationAction]) -> ObservationAction | None:
    """Select the action with the largest expected risk reduction per cost."""

    return max(actions, key=lambda action: action.utility_per_cost, default=None)


def _channels_from_unit_content(unit: ObservationUnit) -> set[ObservationChannel]:
    channels: set[ObservationChannel] = set()
    for key in unit.raw_content:
        channel = {
            "text": ObservationChannel.TEXT,
            "ocr": ObservationChannel.OCR,
            "caption": ObservationChannel.CAPTION,
            "image_path": ObservationChannel.SOURCE_IMAGE,
            "table": ObservationChannel.TABLE_STRUCTURE,
            "table_text": ObservationChannel.TABLE_STRUCTURE,
            "rows": ObservationChannel.TABLE_STRUCTURE,
            "schema": ObservationChannel.TABLE_STRUCTURE,
            "frames": ObservationChannel.SPARSE_FRAMES,
            "dense_frames": ObservationChannel.DENSE_FRAMES,
            "graph": ObservationChannel.GRAPH,
            "source_chunks": ObservationChannel.SOURCE_CHUNKS,
        }.get(key)
        if channel is not None:
            channels.add(channel)
    return channels


def _channels_from_modality(modality: Modality) -> set[ObservationChannel]:
    if modality in {Modality.IMAGE, Modality.MULTIMODAL}:
        return {
            ObservationChannel.OCR,
            ObservationChannel.SOURCE_IMAGE,
            ObservationChannel.VLM_INSPECTION,
        }
    if modality == Modality.TABLE:
        return {ObservationChannel.TEXT, ObservationChannel.TABLE_STRUCTURE}
    if modality in {Modality.VIDEO_SEGMENT, Modality.FRAME_GROUP}:
        return {ObservationChannel.SPARSE_FRAMES, ObservationChannel.DENSE_FRAMES}
    if modality == Modality.GRAPH_RELATION:
        return {ObservationChannel.GRAPH, ObservationChannel.SOURCE_CHUNKS}
    return set()
