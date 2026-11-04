"""FastAPI application for the Industrial Incident Investigator."""

import sys
from pathlib import Path

# Ensure package root ('src') and project root are in sys.path when executed directly or via IDE runner
_src_dir = str(Path(__file__).resolve().parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_root_dir = str(Path(__file__).resolve().parent.parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from contextlib import asynccontextmanager
import asyncio
import os
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
from incident_rag.embeddings.encoder import get_embedding_encoder
from incident_rag.evaluation.benchmark_runner import BenchmarkRunner
from incident_rag.evaluation.reports import save_benchmark_report
from incident_rag.ingestion.service import IngestionService
from incident_rag.logging_config import configure_logging, get_logger
from incident_rag.mem_utils import (
    configure_runtime_memory_limits,
    get_process_memory_mb,
    get_upload_limiter,
    trim_memory,
)
from incident_rag.pipeline import IncidentRAGPipeline
from incident_rag.reranking import get_cross_encoder_reranker
from incident_rag.schemas import (
    BenchmarkReport,
    DocumentAnalysisResponse,
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
    """Initializes storage and runtime configurations with immediate, non-blocking port binding."""
    configure_runtime_memory_limits()
    start_rss = get_process_memory_mb()
    logger.info(f"Initializing Industrial Incident Investigator API... (Initial RSS: {start_rss:.1f} MB)")

    # Pre-warm metadata and persistent disk stores (BM25, document store, vector client)
    # These lightweight disk/metadata operations execute in milliseconds without loading heavy ML models
    doc_store = get_document_store()
    get_vector_store()
    get_bm25_engine()

    ready_rss = get_process_memory_mb()
    logger.info(f"FastAPI storage ready to bind port. (RSS: {ready_rss:.1f} MB)")

    settings = get_settings()
    if settings.AUTO_SEED_DATA:
        # Non-blocking background seeding if explicitly enabled in configuration.
        # Uses asyncio.to_thread to avoid blocking the event loop on heavy synchronous work.
        async def _seed_background_kb():
            await asyncio.sleep(2.0)
            try:
                if len(doc_store.list_documents()) == 0:
                    raw_dir = settings.resolve_path("data/raw")
                    if raw_dir.exists():
                        logger.info(f"Starting background knowledge base seeding from {raw_dir}...")
                        ingestion_service = get_ingestion_service()
                        await asyncio.to_thread(ingestion_service.ingest_directory, raw_dir)
                        trim_memory()
                        logger.info(
                            f"Background seeding finished. Total indexed documents: {len(doc_store.list_documents())} "
                            f"(RSS: {get_process_memory_mb():.1f} MB)"
                        )
            except Exception as e:
                logger.warning(f"Background seeding encountered non-fatal error: {e}. API remains fully responsive.")
                trim_memory()

        asyncio.create_task(_seed_background_kb())
    else:
        logger.info("Automatic background knowledge base seeding is disabled (AUTO_SEED_DATA=False).")

    yield

    logger.info("Shutting down Industrial Incident Investigator API.")
    trim_memory()


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
    """System health check and loaded resources report.
    
    Guarantees lightweight execution: never initializes or forces loading of ML models.
    """
    encoder = get_embedding_encoder()
    reranker = get_cross_encoder_reranker()
    current_rss = get_process_memory_mb()
    return {
        "status": "healthy",
        "version": __version__,
        "environment": settings.ENVIRONMENT,
        "indexed_documents": len(doc_store.list_documents()),
        "indexed_chunks": vec_store.count_chunks(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_model_loaded": encoder.is_loaded,
        "embedding_batch_size": getattr(settings, "EMBEDDING_BATCH_SIZE", 2),
        "reranker_model": settings.RERANKER_MODEL,
        "reranker_enabled": reranker.is_enabled,
        "reranker_model_loaded": reranker.is_loaded,
        "auto_seed_data": settings.AUTO_SEED_DATA,
        "groq_configured": bool(settings.GROQ_API_KEY.strip()),
        "process_memory_mb": current_rss,
    }


# ---------------------------------------------------------------------------
# Document Ingestion & Knowledge Base Management
# ---------------------------------------------------------------------------

@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    """Uploads and indexes an incident report (PDF, Markdown, or TXT). Max size: 25MB.
    
    Protected by an atomic upload concurrency limiter to eliminate check-then-acquire
    race conditions and enforce strict single-upload isolation on Render Free (512MB RAM).
    """
    settings = get_settings()
    limiter = get_upload_limiter(max_concurrent=settings.MAX_CONCURRENT_UPLOADS)
    # Atomic non-blocking acquisition: eliminates check-then-acquire race conditions
    if not limiter.try_acquire():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Upload concurrency limit reached. The server is currently processing another document upload "
                "to protect memory limits on the 512 MB tier. Please retry in a few moments."
            ),
            headers={"Retry-After": "5"},
        )

    try:
        allowed_extensions = {".pdf", ".md", ".txt", ".markdown"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format '{ext}'. Supported formats: .pdf, .md, .txt",
            )

        max_mb = getattr(settings, "MAX_UPLOAD_SIZE_MB", 10)
        max_file_size = max_mb * 1024 * 1024
        READ_CHUNK_SIZE = 64 * 1024  # 64 KB streaming buffer

        chunks: List[bytes] = []
        total_bytes = 0

        while True:
            chunk = await file.read(READ_CHUNK_SIZE)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_file_size:
                # Stop reading immediately and free chunk buffers to prevent memory bloat
                del chunks
                raise HTTPException(
                    status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                    detail=f"File size exceeds maximum permitted limit of {max_mb}MB.",
                )
            chunks.append(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty.",
            )

        content = b"".join(chunks)
        del chunks

        # Offload synchronous CPU/embedding ingestion to thread to keep the asyncio event loop responsive
        result = await asyncio.to_thread(
            ingestion_service.ingest_bytes, content, file.filename, force_reindex=True
        )
        return result
    except HTTPException:
        # Re-raise HTTP exceptions (400, 413, etc.) directly; finally block guarantees limiter release
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except RuntimeError as re:
        logger.error(f"Runtime error processing upload '{file.filename}': {re}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Document processing failed: {str(re)}")
    except Exception as e:
        logger.error(f"Error processing upload '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process document upload.")
    finally:
        try:
            await file.close()
        except Exception:
            pass
        limiter.release()
        trim_memory()


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


@app.post(
    "/api/v1/documents/{document_id}/analyze",
    response_model=DocumentAnalysisResponse,
    tags=["Documents", "Investigation"],
)
async def analyze_document(
    document_id: str,
    pipeline: IncidentRAGPipeline = Depends(get_rag_pipeline),
):
    """Executes structured document-driven forensic investigation for a selected incident report."""
    try:
        response = pipeline.analyze_document(document_id)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during document analysis of '{document_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document analysis failed: {str(e)}",
        )



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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Investigation pipeline failed: {str(e)}")
    finally:
        trim_memory()


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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Retrieval operation failed: {str(e)}")
    finally:
        trim_memory()


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
    finally:
        trim_memory()


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
        "enable_reranker": settings.ENABLE_RERANKER,
        "auto_seed_data": settings.AUTO_SEED_DATA,
        "embedding_batch_size": getattr(settings, "EMBEDDING_BATCH_SIZE", 2),
        "max_concurrent_uploads": getattr(settings, "MAX_CONCURRENT_UPLOADS", 1),
        "max_upload_size_mb": getattr(settings, "MAX_UPLOAD_SIZE_MB", 10),
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


if __name__ == "__main__":
    import os
    import uvicorn
    from incident_rag.mem_utils import configure_runtime_memory_limits

    configure_runtime_memory_limits()
    host = os.environ.get("FASTAPI_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("FASTAPI_PORT", 8000)))
    uvicorn.run("incident_rag.api:app", host=host, port=port, workers=1, reload=False)
