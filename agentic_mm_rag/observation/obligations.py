from __future__ import annotations

from pydantic import BaseModel, Field


class SearchObligation(BaseModel):
    """Fact-facing requirement that should be checked to answer a question."""

    id: str
    text: str
    required_modalities: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
