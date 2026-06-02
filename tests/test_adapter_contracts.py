from agentic_mm_rag.adapters.generic import GenericDictAdapter
from agentic_mm_rag.observation.states import ObservationChannel
from agentic_mm_rag.observation.units import Modality, SourceType
from agentic_mm_rag.retrieval import RetrievalQuery


def test_generic_dict_adapter_converts_retrieval_items() -> None:
    adapter = GenericDictAdapter()
    query = RetrievalQuery(question_id="q1", question="What value is shown?")
    output = adapter.convert(
        query=query,
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

    candidate = output.retrieval_batch.candidates[0]
    assert output.adapter_name == "generic_dict"
    assert output.diagnostics["converted_items"] == 1
    assert candidate.unit.source_type == SourceType.DOCUMENT
    assert candidate.unit.modality == Modality.IMAGE
    assert candidate.initial_state.is_observed(ObservationChannel.CAPTION)


def test_retrieval_batch_top_k_sorts_by_rank() -> None:
    adapter = GenericDictAdapter()
    output = adapter.convert(
        query=RetrievalQuery(question_id="q1", question="Where is the answer?"),
        raw_items=[
            {"item_id": "u2", "source_id": "s", "rank": 2, "content": {"text": "b"}},
            {"item_id": "u1", "source_id": "s", "rank": 1, "content": {"text": "a"}},
        ],
    )

    assert [candidate.unit.unit_id for candidate in output.retrieval_batch.top_k(1)] == ["u1"]
