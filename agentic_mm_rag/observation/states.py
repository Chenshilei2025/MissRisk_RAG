from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ObservationChannel(StrEnum):
    TEXT = "text"
    OCR = "ocr"
    CAPTION = "caption"
    SOURCE_IMAGE = "source_image"
    VLM_INSPECTION = "vlm_inspection"
    TABLE_STRUCTURE = "table_structure"
    SPARSE_FRAMES = "sparse_frames"
    DENSE_FRAMES = "dense_frames"
    GRAPH = "graph"
    SOURCE_CHUNKS = "source_chunks"


class ObservationState(BaseModel):
    """What the current system has actually observed for one unit."""

    unit_id: str
    state_id: str
    observed_channels: dict[ObservationChannel, bool] = Field(default_factory=dict)
    quality: dict[str, float] = Field(default_factory=dict)
    visible_content: dict[str, Any] = Field(default_factory=dict)
    hidden_channels: list[ObservationChannel] = Field(default_factory=list)

    def is_observed(self, channel: ObservationChannel) -> bool:
        return self.observed_channels.get(channel, False)
