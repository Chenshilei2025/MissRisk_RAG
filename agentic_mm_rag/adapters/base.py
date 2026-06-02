from __future__ import annotations

from typing import Any, Protocol

from agentic_mm_rag.adapters.schemas import ExternalRetrievedItem, MissRiskAdapterOutput
from agentic_mm_rag.observation.states import ObservationState
from agentic_mm_rag.observation.units import ObservationUnit
from agentic_mm_rag.retrieval.candidates import RetrievedUnit, RetrievalBatch, RetrievalQuery


class UnitMapper(Protocol):
    """Convert one external retrieval item into MissRisk observation objects."""

    def to_observation_unit(self, item: ExternalRetrievedItem) -> ObservationUnit:
        """Map external item content and locator into an ObservationUnit."""

    def to_initial_state(self, item: ExternalRetrievedItem, unit: ObservationUnit) -> ObservationState:
        """Describe which channels were observed by the upstream RAG system."""


class ExternalRAGAdapter(Protocol):
    """Adapter contract for external multimodal RAG repositories."""

    adapter_name: str

    def normalize_retrieved_items(self, raw_items: list[Any]) -> list[ExternalRetrievedItem]:
        """Wrap repository-specific retrieval results in ExternalRetrievedItem objects."""

    def convert(
        self,
        query: RetrievalQuery,
        raw_items: list[Any],
        mapper: UnitMapper,
    ) -> MissRiskAdapterOutput:
        """Convert retrieved items into a MissRisk retrieval batch."""


def convert_items_with_mapper(
    *,
    adapter_name: str,
    query: RetrievalQuery,
    items: list[ExternalRetrievedItem],
    mapper: UnitMapper,
) -> MissRiskAdapterOutput:
    """Shared conversion helper for simple adapters."""

    candidates: list[RetrievedUnit] = []
    for item in items:
        unit = mapper.to_observation_unit(item)
        state = mapper.to_initial_state(item, unit)
        candidates.append(
            RetrievedUnit(
                unit=unit,
                initial_state=state,
                rank=item.rank,
                score=item.score,
                retriever_name=item.retriever_name,
                retrieval_metadata=item.metadata,
            )
        )
    return MissRiskAdapterOutput(
        retrieval_batch=RetrievalBatch(query=query, candidates=candidates),
        adapter_name=adapter_name,
        diagnostics={"converted_items": len(candidates)},
    )
