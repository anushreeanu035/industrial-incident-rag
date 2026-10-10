"""Unified RAG pipeline orchestrating query rewriting, hybrid search, reranking, and generation."""

import time
from typing import List, Optional
from incident_rag.config import get_settings
from incident_rag.embeddings.encoder import get_embedding_encoder
from incident_rag.generation import GroundedAnswerGenerator
from incident_rag.logging_config import get_logger
from incident_rag.query_rewriting import QueryRewriter
from incident_rag.reranking import get_cross_encoder_reranker
from incident_rag.retrieval.bm25_search import BM25SearchEngine
from incident_rag.retrieval.dense_search import DenseSearchEngine
from incident_rag.retrieval.fusion import reciprocal_rank_fusion
from incident_rag.retrieval.hybrid_search import HybridSearchEngine
from incident_rag.schemas import (
    InvestigationRequest,
    InvestigationResponse,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalResultItem,
    RetrievalStageResults,
    RetrievalStrategy,
)
from incident_rag.storage.document_store import DocumentStore
from incident_rag.storage.vector_store import VectorStore

logger = get_logger(__name__)


class IncidentRAGPipeline:
    """End-to-end RAG pipeline for the Industrial Incident Investigator."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_engine: Optional[BM25SearchEngine] = None,
        document_store: Optional[DocumentStore] = None,
        rewriter: Optional[QueryRewriter] = None,
        generator: Optional[GroundedAnswerGenerator] = None,
    ):
        self.settings = get_settings()
        self.encoder = get_embedding_encoder()
        self.vector_store = vector_store or VectorStore(vector_dim=self.encoder.embedding_dim)
        self.bm25_engine = bm25_engine or BM25SearchEngine()
        self.doc_store = document_store or DocumentStore()

        self.dense_engine = DenseSearchEngine(self.vector_store, self.encoder)
        self.hybrid_engine = HybridSearchEngine(self.dense_engine, self.bm25_engine)
        self.reranker = get_cross_encoder_reranker()
        self.rewriter = rewriter or QueryRewriter()
        self.generator = generator or GroundedAnswerGenerator()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Executes multi-stage retrieval based on selected strategy and captures stage results."""
        start_time = time.perf_counter()
        query = request.query.strip()
        top_k = request.top_k
        strategy = request.strategy
        machine_filter = request.machine_id_filter
        incident_filter = request.incident_type_filter

        stage_results = RetrievalStageResults()
        rewritten_queries: List[str] = []
        final_results: List[RetrievalResultItem] = []

        if strategy == RetrievalStrategy.DENSE:
            dense_res = self.dense_engine.search(
                query=query,
                top_k=top_k,
                machine_id_filter=machine_filter,
                incident_type_filter=incident_filter,
            )
            stage_results.dense_results = dense_res
            final_results = dense_res

        elif strategy == RetrievalStrategy.BM25:
            bm25_res = self.bm25_engine.search(
                query=query,
                top_k=top_k,
                machine_id_filter=machine_filter,
                incident_type_filter=incident_filter,
            )
            stage_results.bm25_results = bm25_res
            final_results = bm25_res

        elif strategy == RetrievalStrategy.HYBRID_RRF:
            fused_res, stages = self.hybrid_engine.search(
                query=query,
                top_k=top_k,
                machine_id_filter=machine_filter,
                incident_type_filter=incident_filter,
            )
            stage_results = stages
            final_results = fused_res

        elif strategy == RetrievalStrategy.HYBRID_RERANK:
            # Candidates for reranking
            candidate_k = max(top_k * 2, self.settings.RERANK_TOP_K)
            fused_res, stages = self.hybrid_engine.search(
                query=query,
                top_k=candidate_k,
                machine_id_filter=machine_filter,
                incident_type_filter=incident_filter,
            )
            reranked_res, _ = self.reranker.rerank(query, fused_res, top_k=top_k)
            stage_results = stages
            stage_results.reranked_results = reranked_res
            final_results = reranked_res

        else:  # FULL_ADVANCED
            enable_rewriting = (
                request.enable_rewriting
                if request.enable_rewriting is not None
                else self.settings.QUERY_REWRITING_ENABLED
            )

            queries_to_search = [query]
            if enable_rewriting:
                rewrite_resp = self.rewriter.rewrite(query)
                rewritten_queries = rewrite_resp.rewritten_queries
                queries_to_search.extend(rewritten_queries)

            all_dense: List[RetrievalResultItem] = []
            all_bm25: List[RetrievalResultItem] = []

            for q in queries_to_search:
                d_res = self.dense_engine.search(
                    query=q,
                    top_k=self.settings.DENSE_TOP_K,
                    machine_id_filter=machine_filter,
                    incident_type_filter=incident_filter,
                )
                b_res = self.bm25_engine.search(
                    query=q,
                    top_k=self.settings.BM25_TOP_K,
                    machine_id_filter=machine_filter,
                    incident_type_filter=incident_filter,
                )
                all_dense.extend(d_res)
                all_bm25.extend(b_res)

            # Deduplicate by chunk_id while retaining best ranks
            fused_res = reciprocal_rank_fusion(
                dense_results=all_dense,
                bm25_results=all_bm25,
                k=self.settings.RRF_K,
                top_k=max(top_k * 3, 15),
            )

            reranked_res, _ = self.reranker.rerank(query, fused_res, top_k=top_k)

            stage_results.dense_results = all_dense[:top_k]
            stage_results.bm25_results = all_bm25[:top_k]
            stage_results.fused_results = fused_res[:top_k]
            stage_results.reranked_results = reranked_res
            final_results = reranked_res

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalResponse(
            query=query,
            rewritten_queries=rewritten_queries,
            strategy=strategy.value if hasattr(strategy, "value") else str(strategy),
            results=final_results,
            stage_results=stage_results,
            execution_time_ms=execution_time_ms,
        )

    def investigate(self, request: InvestigationRequest) -> InvestigationResponse:
        """Executes full investigation pipeline from retrieval through structured generation."""
        retrieval_req = RetrievalRequest(
            query=request.query,
            strategy=request.strategy,
            top_k=request.top_k,
            machine_id_filter=request.machine_id_filter,
            incident_type_filter=request.incident_type_filter,
            enable_rewriting=request.enable_rewriting,
        )
        retrieval_resp = self.retrieve(retrieval_req)

        meta = {
            "strategy": retrieval_resp.strategy,
            "retrieval_time_ms": retrieval_resp.execution_time_ms,
            "num_passages_retrieved": len(retrieval_resp.results),
            "rewritten_queries": retrieval_resp.rewritten_queries,
            "machine_id_filter": request.machine_id_filter,
            "incident_type_filter": request.incident_type_filter,
        }

        response = self.generator.generate(
            query=request.query,
            retrieved_items=retrieval_resp.results,
            retrieval_metadata=meta,
        )
        return response
