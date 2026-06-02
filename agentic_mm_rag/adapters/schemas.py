from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_mm_rag.retrieval.candidates import RetrievalBatch


class ExternalRetrievedItem(BaseModel):
    """Repository-neutral wrapper around one external retrieved item."""

    item_id: str
    source_id: str
    rank: int = Field(ge=1)
    score: float | None = None
    retriever_name: str
    content: dict[str, Any] = Field(default_factory=dict)
    locator: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MissRiskAdapterOutput(BaseModel):
    """Converted MissRisk inputs plus optional adapter diagnostics."""

    retrieval_batch: RetrievalBatch
    adapter_name: str
    diagnostics: dict[str, Any] = Field(default_factory=dict)
