"""FastAPI dependency injection providers and application singletons."""

from functools import lru_cache
from incident_rag.config import Settings, get_settings
from incident_rag.embeddings.encoder import get_embedding_encoder
from incident_rag.evaluation.benchmark_runner import BenchmarkRunner
from incident_rag.generation import GroundedAnswerGenerator
from incident_rag.ingestion.service import IngestionService
from incident_rag.pipeline import IncidentRAGPipeline
from incident_rag.query_rewriting import QueryRewriter
from incident_rag.reranking import get_cross_encoder_reranker
from incident_rag.retrieval.bm25_search import BM25SearchEngine
from incident_rag.storage.document_store import DocumentStore
from incident_rag.storage.experiment_store import ExperimentStore
from incident_rag.storage.vector_store import VectorStore

_VECTOR_STORE = None
_BM25_ENGINE = None
_DOCUMENT_STORE = None
_INGESTION_SERVICE = None
_PIPELINE = None
_BENCHMARK_RUNNER = None
_EXPERIMENT_STORE = None


def get_document_store() -> DocumentStore:
    global _DOCUMENT_STORE
    if _DOCUMENT_STORE is None:
        _DOCUMENT_STORE = DocumentStore()
    return _DOCUMENT_STORE


def get_vector_store() -> VectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        encoder = get_embedding_encoder()
        _VECTOR_STORE = VectorStore(vector_dim=encoder.embedding_dim)
    return _VECTOR_STORE


def get_bm25_engine() -> BM25SearchEngine:
    global _BM25_ENGINE
    if _BM25_ENGINE is None:
        _BM25_ENGINE = BM25SearchEngine()
    return _BM25_ENGINE


def get_ingestion_service() -> IngestionService:
    global _INGESTION_SERVICE
    if _INGESTION_SERVICE is None:
        _INGESTION_SERVICE = IngestionService(
            document_store=get_document_store(),
            vector_store=get_vector_store(),
            encoder=get_embedding_encoder(),
            bm25_engine=get_bm25_engine(),
        )
    return _INGESTION_SERVICE


def get_rag_pipeline() -> IncidentRAGPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = IncidentRAGPipeline(
            vector_store=get_vector_store(),
            bm25_engine=get_bm25_engine(),
            document_store=get_document_store(),
            rewriter=QueryRewriter(),
            generator=GroundedAnswerGenerator(),
        )
    return _PIPELINE


def get_experiment_store() -> ExperimentStore:
    global _EXPERIMENT_STORE
    if _EXPERIMENT_STORE is None:
        _EXPERIMENT_STORE = ExperimentStore()
    return _EXPERIMENT_STORE


def get_benchmark_runner() -> BenchmarkRunner:
    global _BENCHMARK_RUNNER
    if _BENCHMARK_RUNNER is None:
        _BENCHMARK_RUNNER = BenchmarkRunner(pipeline=get_rag_pipeline())
    return _BENCHMARK_RUNNER
