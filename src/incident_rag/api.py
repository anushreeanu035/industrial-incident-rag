"""FastAPI application for the Industrial Incident Investigator."""

from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from incident_rag import __version__
from incident_rag.config import Settings, get_settings
from incident_rag.dependencies import (
    get_benchmark_runner,
    get_bm25_engine,
    get_document_store,
    get_experiment_store,
    get_ingestion_service,
    get_rag_pipeline,
    get_vector_store,
)
from incident_rag.evaluation.benchmark_runner import BenchmarkRunner
from incident_rag.evaluation.reports import save_benchmark_report
from incident_rag.ingestion.service import IngestionService
from incident_rag.logging_config import configure_logging, get_logger
from incident_rag.pipeline import IncidentRAGPipeline
from incident_rag.schemas import (
    BenchmarkReport,
    DocumentMetadata,
    DocumentUploadResponse,
    EvaluationRunRequest,
    InvestigationRequest,
    InvestigationResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from incident_rag.storage.document_store import DocumentStore
from incident_rag.storage.experiment_store import ExperimentStore
from incident_rag.storage.vector_store import VectorStore

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes stores and seeds raw documents if knowledge base is empty."""
    settings = get_settings()
    logger.info("Initializing Industrial Incident Investigator API...")

    # Pre-warm stores and pipeline
    doc_store = get_document_store()
    ingestion_service = get_ingestion_service()

    # Automatically index sample incident reports if database is empty
    if len(doc_store.list_documents()) == 0:
        raw_dir = settings.resolve_path("data/raw")
        if raw_dir.exists():
            logger.info(f"Seeding knowledge base from {raw_dir}...")
            ingestion_service.ingest_directory(raw_dir)
            logger.info(f"Seeding completed. Total indexed documents: {len(doc_store.list_documents())}")

    yield

    logger.info("Shutting down Industrial Incident Investigator API.")


app = FastAPI(
    title="Industrial Incident Investigator API",
    description="Enterprise AI investigation platform for industrial equipment failures, root cause analysis, and RAG evaluation.",
    version=__version__,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_and_logging(request: Request, call_next):
    """Injects correlation ID and logs request latency."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
    return response


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check(
    settings: Settings = Depends(get_settings),
    doc_store: DocumentStore = Depends(get_document_store),
    vec_store: VectorStore = Depends(get_vector_store),
):
    """System health check and loaded resources report."""
    return {
        "status": "healthy",
        "version": __version__,
        "environment": settings.ENVIRONMENT,
        "indexed_documents": len(doc_store.list_documents()),
        "indexed_chunks": vec_store.count_chunks(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "groq_configured": bool(settings.GROQ_API_KEY.strip()),
    }


# ---------------------------------------------------------------------------
# Document Ingestion & Knowledge Base Management
# ---------------------------------------------------------------------------

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Uploads and indexes an incident report (PDF, Markdown, or TXT). Max size: 25MB."""
    allowed_extensions = {".pdf", ".md", ".txt", ".markdown"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats: .pdf, .md, .txt",
        )

    # Max 25 MB limit
    MAX_FILE_SIZE = 25 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum permitted limit of 25MB.",
        )

    try:
        result = ingestion_service.ingest_bytes(content, file.filename, force_reindex=True)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing upload '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process document upload.")


@app.get("/api/v1/documents", response_model=List[DocumentMetadata], tags=["Documents"])
async def list_documents(doc_store: DocumentStore = Depends(get_document_store)):
    """Lists all ingested documents in the knowledge base."""
    return doc_store.list_documents()


@app.get("/api/v1/documents/{document_id}", response_model=DocumentMetadata, tags=["Documents"])
async def get_document(document_id: str, doc_store: DocumentStore = Depends(get_document_store)):
    """Retrieves document metadata by document ID."""
    doc = doc_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return doc


@app.delete("/api/v1/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Deletes a document and its chunks from all vector and keyword indexes."""
    success = ingestion_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return {"message": f"Document '{document_id}' successfully removed from knowledge base."}


# ---------------------------------------------------------------------------
# Investigation & Retrieval Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/investigate", response_model=InvestigationResponse, tags=["Investigation"])
async def run_investigation(
    request: InvestigationRequest,
    pipeline: IncidentRAGPipeline = Depends(get_rag_pipeline),
):
    """Executes full grounded incident investigation and root cause synthesis."""
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Investigation query cannot be empty.")
    try:
        response = pipeline.investigate(request)
        return response
    except Exception as e:
        logger.error(f"Error during investigation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Investigation pipeline failed.")


@app.post("/api/v1/retrieve", response_model=RetrievalResponse, tags=["Retrieval"])
async def run_retrieval(
    request: RetrievalRequest,
    pipeline: IncidentRAGPipeline = Depends(get_rag_pipeline),
):
    """Executes multi-stage retrieval and exposes candidate ranks across pipeline stages."""
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Retrieval query cannot be empty.")
    try:
        response = pipeline.retrieve(request)
        return response
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Retrieval operation failed.")


# ---------------------------------------------------------------------------
# RAG Evaluation & Benchmarking Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/evaluate", response_model=BenchmarkReport, tags=["Evaluation"])
async def execute_evaluation(
    request: EvaluationRunRequest,
    runner: BenchmarkRunner = Depends(get_benchmark_runner),
    exp_store: ExperimentStore = Depends(get_experiment_store),
):
    """Executes reproducible benchmark evaluation over ground truth dataset and records report."""
    try:
        report = runner.run_benchmark(
            strategies=request.strategies,
            top_k=request.top_k,
            run_ragas=request.run_ragas,
        )
        exp_store.record_run(report)
        return report
    except Exception as e:
        logger.error(f"Evaluation benchmark failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Benchmark execution failed: {str(e)}")


@app.get("/api/v1/evaluations", response_model=List[BenchmarkReport], tags=["Evaluation"])
async def list_evaluations(exp_store: ExperimentStore = Depends(get_experiment_store)):
    """Lists historical evaluation reports."""
    return exp_store.list_runs()


@app.get("/api/v1/evaluations/{run_id}", response_model=BenchmarkReport, tags=["Evaluation"])
async def get_evaluation(run_id: str, exp_store: ExperimentStore = Depends(get_experiment_store)):
    """Retrieves a specific evaluation report by run ID."""
    report = exp_store.get_run(run_id)
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation report not found.")
    return report


# ---------------------------------------------------------------------------
# System Settings
# ---------------------------------------------------------------------------

@app.get("/api/v1/settings", tags=["System"])
async def get_system_settings(settings: Settings = Depends(get_settings)):
    """Returns active runtime configuration and model configurations."""
    return {
        "embedding_model": settings.EMBEDDING_MODEL,
        "reranker_model": settings.RERANKER_MODEL,
        "groq_model": settings.GROQ_MODEL,
        "dense_top_k": settings.DENSE_TOP_K,
        "bm25_top_k": settings.BM25_TOP_K,
        "rrf_k": settings.RRF_K,
        "rerank_top_k": settings.RERANK_TOP_K,
        "final_context_k": settings.FINAL_CONTEXT_K,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP,
        "query_rewriting_enabled": settings.QUERY_REWRITING_ENABLED,
        "num_query_rewrites": settings.NUM_QUERY_REWRITES,
        "environment": settings.ENVIRONMENT,
    }
