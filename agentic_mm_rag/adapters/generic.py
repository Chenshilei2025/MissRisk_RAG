from __future__ import annotations

from typing import Any

from agentic_mm_rag.adapters.base import UnitMapper, convert_items_with_mapper
from agentic_mm_rag.adapters.schemas import ExternalRetrievedItem, MissRiskAdapterOutput
from agentic_mm_rag.observation.states import ObservationChannel, ObservationState
from agentic_mm_rag.observation.units import Modality, ObservationUnit, SourceType
from agentic_mm_rag.retrieval.candidates import RetrievalQuery


class GenericDictMapper:
    """Default mapper for dict-like retrieved items.

    This mapper is intentionally conservative. Specific RAG repositories should
    subclass or replace it when they expose richer source locators or modalities.
    """

    def to_observation_unit(self, item: ExternalRetrievedItem) -> ObservationUnit:
        source_type = item.metadata.get("source_type", SourceType.DOCUMENT)
        modality = item.metadata.get("modality", Modality.TEXT)
        return ObservationUnit(
            unit_id=item.item_id,
            source_id=item.source_id,
            source_type=SourceType(source_type),
            modality=Modality(modality),
            locator=item.locator,
            raw_content=item.content,
            metadata=item.metadata,
        )

    def to_initial_state(self, item: ExternalRetrievedItem, unit: ObservationUnit) -> ObservationState:
        observed_channels = {}
        if "text" in item.content:
            observed_channels[ObservationChannel.TEXT] = True
        if "ocr" in item.content:
            observed_channels[ObservationChannel.OCR] = True
        if "caption" in item.content:
            observed_channels[ObservationChannel.CAPTION] = True

        return ObservationState(
            unit_id=unit.unit_id,
            state_id=item.metadata.get("state_id", "retrieved_content"),
            observed_channels=observed_channels,
            visible_content=item.content,
        )


class GenericDictAdapter:
    """Adapter for simple dict-based retrieval outputs."""

    adapter_name = "generic_dict"

    def normalize_retrieved_items(self, raw_items: list[Any]) -> list[ExternalRetrievedItem]:
        items: list[ExternalRetrievedItem] = []
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                raise TypeError("GenericDictAdapter expects each raw item to be a dict")
            items.append(
                ExternalRetrievedItem(
                    item_id=str(raw_item.get("item_id", raw_item.get("unit_id", f"item_{index}"))),
                    source_id=str(raw_item.get("source_id", raw_item.get("doc_id", "unknown_source"))),
                    rank=int(raw_item.get("rank", index)),
                    score=raw_item.get("score"),
                    retriever_name=str(raw_item.get("retriever_name", "unknown_retriever")),
                    content=dict(raw_item.get("content", {})),
                    locator=dict(raw_item.get("locator", {})),
                    metadata=dict(raw_item.get("metadata", {})),
                )
            )
        return items

    def convert(
        self,
        query: RetrievalQuery,
        raw_items: list[Any],
        mapper: UnitMapper | None = None,
    ) -> MissRiskAdapterOutput:
        items = self.normalize_retrieved_items(raw_items)
        return convert_items_with_mapper(
            adapter_name=self.adapter_name,
            query=query,
            items=items,
            mapper=mapper or GenericDictMapper(),
        )
