from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    DOCUMENT = "document"
    SLIDE = "slide"
    TABLE = "table"
    IMAGE = "image"
    VIDEO = "video"
    GRAPH = "graph"


class Modality(StrEnum):
    TEXT = "text"
    OCR = "ocr"
    CAPTION = "caption"
    IMAGE = "image"
    TABLE = "table"
    VIDEO_SEGMENT = "video_segment"
    FRAME_GROUP = "frame_group"
    GRAPH_RELATION = "graph_relation"
    MULTIMODAL = "multimodal"


class ObservationUnit(BaseModel):
    """Smallest source unit that can be observed by the system."""

    unit_id: str
    source_id: str
    source_type: SourceType
    modality: Modality
    locator: dict[str, Any] = Field(default_factory=dict)
    raw_content: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    label_answer_bearing: int | None = Field(default=None, ge=0, le=1)
