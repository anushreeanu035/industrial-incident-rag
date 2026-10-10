"""Pydantic data models and schemas for Industrial Incident Investigator."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    """Supported retrieval strategies in the RAG pipeline."""
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_RERANK = "hybrid_rerank"
    FULL_ADVANCED = "full_advanced"


# ---------------------------------------------------------------------------
# Document and Chunk Schemas
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """Metadata for a single chunk."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    chunk_index: int = Field(..., description="Zero-based index of the chunk in document")
    page_number: Optional[int] = Field(None, description="Page number if extracted from PDF")
    start_char: int = Field(default=0, description="Character start offset in document text")
    end_char: int = Field(default=0, description="Character end offset in document text")
    machine_id: Optional[str] = Field(None, description="Industrial equipment ID (e.g., HP-17)")
    incident_type: Optional[str] = Field(None, description="Category of incident")
    incident_date: Optional[str] = Field(None, description="Date of the incident")
    source_filename: str = Field(..., description="Original filename")


class DocumentChunk(BaseModel):
    """Chunk of text indexed in vector and keyword engines."""
    chunk_id: str
    document_id: str
    text: str
    metadata: ChunkMetadata


class DocumentMetadata(BaseModel):
    """Metadata tracking ingested documents."""
    document_id: str = Field(..., description="Unique document ID (hash-based)")
    filename: str
    file_type: str = Field(..., description="File extension: pdf, txt, or md")
    file_size_bytes: int
    upload_timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    machine_id: Optional[str] = None
    incident_type: Optional[str] = None
    incident_date: Optional[str] = None
    page_count: int = 1
    chunk_count: int = 0
    checksum_sha256: str
    status: str = Field(default="pending", description="indexed | failed | pending")
    error_message: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Response returned upon document ingestion."""
    document_id: str
    filename: str
    chunks_created: int
    status: str
    message: str


# ---------------------------------------------------------------------------
# Retrieval and Ranking Schemas
# ---------------------------------------------------------------------------

class RetrievalResultItem(BaseModel):
    """Individual retrieved chunk with multi-stage scoring metadata."""
    chunk_id: str
    document_id: str
    text: str
    metadata: ChunkMetadata
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rrf_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    final_rank: int = 1


class RetrievalStageResults(BaseModel):
    """Full inspection results across all pipeline stages (for Retrieval Explorer)."""
    dense_results: List[RetrievalResultItem] = []
    bm25_results: List[RetrievalResultItem] = []
    fused_results: List[RetrievalResultItem] = []
    reranked_results: List[RetrievalResultItem] = []


class QueryRewriteRequest(BaseModel):
    query: str


class QueryRewriteResponse(BaseModel):
    original_query: str
    rewritten_queries: List[str]


class RetrievalRequest(BaseModel):
    """Request payload for retrieval endpoints."""
    query: str
    strategy: RetrievalStrategy = RetrievalStrategy.FULL_ADVANCED
    top_k: int = Field(default=5, ge=1, le=50)
    machine_id_filter: Optional[str] = None
    incident_type_filter: Optional[str] = None
    document_id_filter: Optional[str] = None
    enable_rewriting: Optional[bool] = None


class RetrievalResponse(BaseModel):
    """Response returned by the retrieval endpoint."""
    query: str
    rewritten_queries: List[str] = []
    strategy: str
    results: List[RetrievalResultItem]
    stage_results: Optional[RetrievalStageResults] = None
    execution_time_ms: float


# ---------------------------------------------------------------------------
# Investigation and Generation Schemas
# ---------------------------------------------------------------------------

class EvidenceCitation(BaseModel):
    """Traceable citation to an extracted evidence passage."""
    citation_id: str = Field(..., description="Citation marker (e.g. [CIT-1])")
    document_id: str
    chunk_id: str
    source_name: str
    page_number: Optional[int] = None
    excerpt: str
    relevance_reason: str


class InvestigationHypothesis(BaseModel):
    """Root cause hypothesis with confidence indicator and supporting citations."""
    hypothesis: str
    confidence_level: str = Field(
        ...,
        description="Confidence indicator: 'High (Documented Fact)', 'Medium (Strong Evidence)', 'Low (Tentative Hypothesis)', or 'Insufficient Evidence'"
    )
    supporting_evidence: List[str]
    citation_ids: List[str]


class InvestigationResponse(BaseModel):
    """Structured response from the incident investigation pipeline."""
    executive_summary: str
    documented_observations: List[str]
    potential_root_causes: List[InvestigationHypothesis]
    corrective_actions: List[str]
    missing_evidence_and_uncertainties: List[str]
    citations: List[EvidenceCitation]
    is_sufficient_evidence: bool = True
    retrieval_metadata: Dict[str, Any] = {}


class InvestigationRequest(BaseModel):
    """Request payload for the investigation workspace."""
    query: str
    machine_id_filter: Optional[str] = None
    incident_type_filter: Optional[str] = None
    document_id_filter: Optional[str] = None
    strategy: RetrievalStrategy = RetrievalStrategy.FULL_ADVANCED
    top_k: int = Field(default=5, ge=1, le=20)
    enable_rewriting: Optional[bool] = None


# ---------------------------------------------------------------------------
# Evaluation and Benchmark Schemas
# ---------------------------------------------------------------------------

class RetrievalMetricScores(BaseModel):
    """Deterministic retrieval ranking metrics."""
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0


class RagasMetricScores(BaseModel):
    """Generation and context metrics computed via Ragas or ground truth."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None


class StrategyBenchmarkResult(BaseModel):
    """Benchmark outcome for a specific pipeline strategy."""
    strategy: str
    retrieval_metrics: RetrievalMetricScores
    ragas_metrics: Optional[RagasMetricScores] = None
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0


class BenchmarkReport(BaseModel):
    """Reproducible evaluation report stored to JSON and CSV."""
    run_id: str
    timestamp: str
    dataset_version: str
    dataset_hash: str
    embedding_model: str
    reranker_model: str
    llm_model: str
    configuration: Dict[str, Any]
    results: List[StrategyBenchmarkResult]


class EvaluationRunRequest(BaseModel):
    """Request to initiate a benchmark evaluation run."""
    dataset_name: str = "default"
    strategies: List[RetrievalStrategy] = [
        RetrievalStrategy.DENSE,
        RetrievalStrategy.BM25,
        RetrievalStrategy.HYBRID_RRF,
        RetrievalStrategy.HYBRID_RERANK,
        RetrievalStrategy.FULL_ADVANCED
    ]
    top_k: int = 5
    run_ragas: bool = False
