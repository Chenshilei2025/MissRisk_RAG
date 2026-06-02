from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic_mm_rag.observation.states import ObservationState
from agentic_mm_rag.observation.units import ObservationUnit


class RetrievalQuery(BaseModel):
    """Question/query payload received from an upstream RAG pipeline."""

    question_id: str
    question: str
    query_text: str | None = None
    query_variants: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedUnit(BaseModel):
    """Ranked candidate unit from a retriever, reranker, or agent tool."""

    unit: ObservationUnit
    initial_state: ObservationState
    rank: int = Field(ge=1)
    score: float | None = None
    retriever_name: str
    query_variant: str | None = None
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalBatch(BaseModel):
    """All candidates returned for one question by one or more retrieval paths."""

    query: RetrievalQuery
    candidates: list[RetrievedUnit] = Field(default_factory=list)

    def top_k(self, k: int) -> list[RetrievedUnit]:
        if k < 0:
            raise ValueError("k must be non-negative")
        return sorted(self.candidates, key=lambda candidate: candidate.rank)[:k]
