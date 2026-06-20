from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from agentic_mm_rag._compat import StrEnum


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


class TextView(BaseModel):
    """Textual view exposed to text encoders and audit tools."""

    content: str
    source_key: str | None = None


class VisualView(BaseModel):
    """Pixel-bearing view that can be loaded by a multimodal model."""

    image_path: str
    extracted_path: str | None = None
    asset_available: bool | None = None
    width: int | None = Field(default=None, ge=0)
    height: int | None = Field(default=None, ge=0)
    mime_type: str | None = None


class TableView(BaseModel):
    """Structured/table view, optionally backed by the original image crop."""

    table_text: str | None = None
    image_path: str | None = None
    source_key: str | None = None


class ObservationViews(BaseModel):
    """Parallel views for the same observation unit.

    `raw_content` is kept for backward compatibility. New multimodal pipelines
    should prefer these typed views so text, table, and visual inputs are not
    collapsed into a single string.
    """

    text_view: TextView | None = None
    visual_view: VisualView | None = None
    table_view: TableView | None = None


class ObservationUnit(BaseModel):
    """Smallest source unit that can be observed by the system."""

    unit_id: str
    source_id: str
    source_type: SourceType
    modality: Modality
    locator: dict[str, Any] = Field(default_factory=dict)
    raw_content: dict[str, Any] = Field(default_factory=dict)
    views: ObservationViews = Field(default_factory=ObservationViews)
    metadata: dict[str, Any] = Field(default_factory=dict)
    label_answer_bearing: int | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def populate_views_from_raw_content(self) -> "ObservationUnit":
        """Backfill typed views from legacy raw_content when possible."""

        if self.views.text_view is None:
            for key in ("text", "ocr", "caption"):
                value = self.raw_content.get(key)
                if isinstance(value, str) and value.strip():
                    self.views.text_view = TextView(content=value, source_key=key)
                    break

        image_path = self.raw_content.get("image_path") or self.raw_content.get("page_image_path")
        if self.views.visual_view is None and isinstance(image_path, str) and image_path.strip():
            self.views.visual_view = VisualView(image_path=image_path)

        table_text = self.raw_content.get("table_text") or self.raw_content.get("table")
        if self.views.table_view is None and (
            isinstance(table_text, str) or isinstance(image_path, str)
        ):
            self.views.table_view = TableView(
                table_text=table_text if isinstance(table_text, str) else None,
                image_path=image_path if isinstance(image_path, str) else None,
                source_key="table_text" if isinstance(table_text, str) else None,
            )
        return self
