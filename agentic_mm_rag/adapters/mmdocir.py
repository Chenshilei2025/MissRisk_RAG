from __future__ import annotations

from typing import Any

from agentic_mm_rag.adapters.base import UnitMapper, convert_items_with_mapper
from agentic_mm_rag.adapters.schemas import ExternalRetrievedItem, MissRiskAdapterOutput
from agentic_mm_rag.observation.states import ObservationChannel, ObservationState
from agentic_mm_rag.observation.units import Modality, ObservationUnit, SourceType
from agentic_mm_rag.retrieval.candidates import RetrievalQuery


class MMDocIRMapper:
    """Map MMDocIR/ColPali-style retrieval rows into MissRisk observation objects."""

    def to_observation_unit(self, item: ExternalRetrievedItem) -> ObservationUnit:
        metadata = dict(item.metadata)
        source_type = SourceType(metadata.get("source_type", SourceType.DOCUMENT))
        modality = Modality(metadata.get("modality", infer_modality(item.content, metadata)))
        return ObservationUnit(
            unit_id=item.item_id,
            source_id=item.source_id,
            source_type=source_type,
            modality=modality,
            locator=item.locator,
            raw_content=item.content,
            metadata=metadata,
        )

    def to_initial_state(self, item: ExternalRetrievedItem, unit: ObservationUnit) -> ObservationState:
        observed_channels = observed_channels_for_content(item.content)
        return ObservationState(
            unit_id=unit.unit_id,
            state_id=item.metadata.get("state_id", "retrieved_by_mmdocir"),
            observed_channels=observed_channels,
            visible_content=item.content,
            hidden_channels=hidden_channels_for_unit(unit.modality, observed_channels),
        )


class MMDocIRAdapter:
    """Adapter for official MMDocIR and visual document retrieval top-k exports."""

    adapter_name = "mmdocir"

    def normalize_retrieved_items(self, raw_items: list[Any]) -> list[ExternalRetrievedItem]:
        items: list[ExternalRetrievedItem] = []
        for index, raw_item in enumerate(raw_items, start=1):
            if not isinstance(raw_item, dict):
                raise TypeError("MMDocIRAdapter expects each raw retrieval item to be a dict")
            metadata = build_metadata(raw_item)
            content = build_content(raw_item)
            locator = build_locator(raw_item)
            source_id = first_string(raw_item, ("source_id", "doc_name", "doc_id", "document_id")) or "unknown_source"
            item_id = first_string(raw_item, ("item_id", "unit_id", "quote_id", "passage_id", "chunk_id"))
            if item_id is None:
                item_id = make_item_id(source_id=source_id, locator=locator, rank=index)
            items.append(
                ExternalRetrievedItem(
                    item_id=item_id,
                    source_id=source_id,
                    rank=int(raw_item.get("rank", raw_item.get("position", index))),
                    score=first_float(raw_item, ("score", "similarity", "relevance_score", "rerank_score")),
                    retriever_name=(
                        first_string(raw_item, ("retriever_name", "retriever", "model_name", "method"))
                        or "mmdocir"
                    ),
                    content=content,
                    locator=locator,
                    metadata=metadata,
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
            mapper=mapper or MMDocIRMapper(),
        )


def build_content(raw_item: dict[str, Any]) -> dict[str, Any]:
    content = dict(raw_item.get("content", {}))
    for source_key, target_key in (
        ("text", "text"),
        ("ocr", "ocr"),
        ("caption", "caption"),
        ("img_description", "caption"),
        ("table_text", "table_text"),
        ("image_path", "image_path"),
        ("img_path", "image_path"),
        ("page_image_path", "page_image_path"),
    ):
        value = raw_item.get(source_key)
        if value is not None and target_key not in content:
            content[target_key] = value
    return content


def build_locator(raw_item: dict[str, Any]) -> dict[str, Any]:
    locator = dict(raw_item.get("locator", {}))
    for key in (
        "doc_name",
        "page_id",
        "page",
        "layout_id",
        "block_id",
        "quote_id",
        "image_id",
        "bbox",
    ):
        if raw_item.get(key) is not None and key not in locator:
            locator[key] = raw_item[key]
    return locator


def build_metadata(raw_item: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(raw_item.get("metadata", {}))
    metadata.setdefault("dataset", raw_item.get("dataset", "mmdocir"))
    metadata.setdefault("adapter_family", "mmdocir")
    if "source_type" not in metadata:
        metadata["source_type"] = SourceType.DOCUMENT.value
    if "modality" not in metadata:
        metadata["modality"] = infer_modality(build_content(raw_item), raw_item).value
    for key in ("query_id", "question_id", "q_id", "split", "retriever_name"):
        if raw_item.get(key) is not None and key not in metadata:
            metadata[key] = raw_item[key]
    return metadata


def infer_modality(content: dict[str, Any], metadata: dict[str, Any]) -> Modality:
    raw_modality = metadata.get("modality")
    if raw_modality:
        return Modality(raw_modality)
    raw_type = str(metadata.get("type", metadata.get("quote_type", ""))).lower()
    if raw_type == "table" or content.get("table_text"):
        return Modality.TABLE
    if content.get("image_path") or content.get("page_image_path") or raw_type in {"image", "figure"}:
        return Modality.IMAGE
    if content.get("ocr"):
        return Modality.OCR
    if content.get("caption"):
        return Modality.CAPTION
    if content.get("text"):
        return Modality.TEXT
    return Modality.MULTIMODAL


def observed_channels_for_content(content: dict[str, Any]) -> dict[ObservationChannel, bool]:
    observed: dict[ObservationChannel, bool] = {}
    if content.get("text"):
        observed[ObservationChannel.TEXT] = True
    if content.get("ocr"):
        observed[ObservationChannel.OCR] = True
    if content.get("caption"):
        observed[ObservationChannel.CAPTION] = True
    if content.get("table_text"):
        observed[ObservationChannel.TABLE_STRUCTURE] = True
    if content.get("image_path") or content.get("page_image_path"):
        observed[ObservationChannel.SOURCE_IMAGE] = True
    return observed


def hidden_channels_for_unit(
    modality: Modality,
    observed_channels: dict[ObservationChannel, bool],
) -> list[ObservationChannel]:
    relevant = {
        Modality.TEXT: [ObservationChannel.OCR, ObservationChannel.SOURCE_IMAGE],
        Modality.OCR: [ObservationChannel.TEXT, ObservationChannel.SOURCE_IMAGE],
        Modality.CAPTION: [ObservationChannel.SOURCE_IMAGE, ObservationChannel.VLM_INSPECTION],
        Modality.IMAGE: [ObservationChannel.CAPTION, ObservationChannel.VLM_INSPECTION],
        Modality.TABLE: [
            ObservationChannel.OCR,
            ObservationChannel.TABLE_STRUCTURE,
            ObservationChannel.SOURCE_IMAGE,
            ObservationChannel.VLM_INSPECTION,
        ],
        Modality.MULTIMODAL: [
            ObservationChannel.TEXT,
            ObservationChannel.OCR,
            ObservationChannel.CAPTION,
            ObservationChannel.SOURCE_IMAGE,
            ObservationChannel.VLM_INSPECTION,
        ],
    }.get(modality, [])
    return [channel for channel in relevant if not observed_channels.get(channel, False)]


def make_item_id(*, source_id: str, locator: dict[str, Any], rank: int) -> str:
    page = locator.get("page_id", locator.get("page", "unknown_page"))
    block = locator.get("layout_id", locator.get("block_id", f"rank_{rank}"))
    return f"{source_id}:page_{page}:block_{block}"


def first_string(raw_item: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = raw_item.get(key)
        if value is not None and value != "":
            return str(value)
    return None


def first_float(raw_item: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = raw_item.get(key)
        if value is None or value == "":
            continue
        return float(value)
    return None
