from __future__ import annotations

from agentic_mm_rag.adapters.generic import GenericDictAdapter
from agentic_mm_rag.adapters.mmdocir import MMDocIRAdapter
from agentic_mm_rag.observation.states import ObservationChannel
from agentic_mm_rag.observation.units import Modality, SourceType
from agentic_mm_rag.retrieval import RetrievalQuery


def test_generic_dict_adapter_maps_retrieved_items() -> None:
    adapter = GenericDictAdapter()
    output = adapter.convert(
        query=RetrievalQuery(question_id="q1", question="What value is shown?"),
        raw_items=[
            {
                "item_id": "doc1_page2_chart",
                "source_id": "doc1",
                "rank": 1,
                "score": 0.91,
                "retriever_name": "colpali",
                "content": {"caption": "A chart showing 42."},
                "locator": {"page_id": 2, "block_id": "chart"},
                "metadata": {"source_type": "document", "modality": "image"},
            }
        ],
    )

    batch = output.retrieval_batch
    assert output.adapter_name == "generic_dict"
    assert batch.query.question_id == "q1"
    assert batch.candidates[0].unit.modality == Modality.IMAGE
    assert batch.candidates[0].initial_state.is_observed(ObservationChannel.CAPTION)


def test_mmdocir_adapter_infers_visual_document_units() -> None:
    adapter = MMDocIRAdapter()
    output = adapter.convert(
        query=RetrievalQuery(question_id="q2", question="Which chart peaks?"),
        raw_items=[
            {
                "doc_name": "paper.pdf",
                "page_id": 5,
                "rank": 2,
                "score": 0.77,
                "img_description": "The collapsed tree line peaks at 2000 tokens.",
                "image_path": "images/paper_page5_chart.png",
            }
        ],
    )

    item = output.retrieval_batch.candidates[0]
    assert output.adapter_name == "mmdocir"
    assert item.unit.source_type == SourceType.DOCUMENT
    assert item.unit.modality == Modality.IMAGE
    assert item.unit.views.visual_view is not None
    assert item.initial_state.is_observed(ObservationChannel.SOURCE_IMAGE)
