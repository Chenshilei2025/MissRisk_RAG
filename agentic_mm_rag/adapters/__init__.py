"""Adapter interfaces for connecting external multimodal RAG repositories."""

from agentic_mm_rag.adapters.base import ExternalRAGAdapter, UnitMapper
from agentic_mm_rag.adapters.mmdocir import MMDocIRAdapter, MMDocIRMapper
from agentic_mm_rag.adapters.schemas import ExternalRetrievedItem, MissRiskAdapterOutput

__all__ = [
    "ExternalRAGAdapter",
    "ExternalRetrievedItem",
    "MMDocIRAdapter",
    "MMDocIRMapper",
    "MissRiskAdapterOutput",
    "UnitMapper",
]
