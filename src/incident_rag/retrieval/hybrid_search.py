"""Hybrid retrieval combining dense semantic search and BM25 keyword search using RRF."""

from typing import List, Optional, Tuple
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.retrieval.bm25_search import BM25SearchEngine
from incident_rag.retrieval.dense_search import DenseSearchEngine
from incident_rag.retrieval.fusion import reciprocal_rank_fusion
from incident_rag.schemas import RetrievalResultItem, RetrievalStageResults

logger = get_logger(__name__)


class HybridSearchEngine:
    """Orchestrates multi-channel retrieval and reciprocal rank fusion."""

    def __init__(self, dense_engine: DenseSearchEngine, bm25_engine: BM25SearchEngine):
        self.dense_engine = dense_engine
        self.bm25_engine = bm25_engine
        self.settings = get_settings()

    def search(
        self,
        query: str,
        top_k: int = 10,
        machine_id_filter: Optional[str] = None,
        incident_type_filter: Optional[str] = None,
        rrf_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
    ) -> Tuple[List[RetrievalResultItem], RetrievalStageResults]:
        """Executes dense and BM25 searches and fuses the results using RRF."""
        k_val = rrf_k if rrf_k is not None else self.settings.RRF_K
        d_top = dense_top_k if dense_top_k is not None else self.settings.DENSE_TOP_K
        b_top = bm25_top_k if bm25_top_k is not None else self.settings.BM25_TOP_K

        # Retrieve independently from both channels
        dense_results = self.dense_engine.search(
            query=query,
            top_k=d_top,
            machine_id_filter=machine_id_filter,
            incident_type_filter=incident_type_filter,
        )

        bm25_results = self.bm25_engine.search(
            query=query,
            top_k=b_top,
            machine_id_filter=machine_id_filter,
            incident_type_filter=incident_type_filter,
        )

        # Fuse rankings using RRF
        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=k_val,
            top_k=top_k,
        )

        stage_results = RetrievalStageResults(
            dense_results=dense_results,
            bm25_results=bm25_results,
            fused_results=fused_results,
            reranked_results=[],
        )

        return fused_results, stage_results
