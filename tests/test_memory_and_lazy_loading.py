"""Tests for memory optimization, lazy loading, and singleton caching."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest
from incident_rag.api import app
from incident_rag.embeddings.encoder import EmbeddingEncoder, get_embedding_encoder
from incident_rag.mem_utils import get_process_memory_mb, trim_memory
from incident_rag.reranking import CrossEncoderReranker, get_cross_encoder_reranker
from incident_rag.schemas import ChunkMetadata, RetrievalResultItem


@pytest.fixture(scope="module")
def api_test_client():
    with TestClient(app) as client:
        yield client


def test_process_memory_telemetry():
    """Verifies that memory tracking returns a positive float."""
    rss = get_process_memory_mb()
    assert isinstance(rss, float)
    assert rss >= 0.0

    trimmed_rss = trim_memory()
    assert isinstance(trimmed_rss, float)
    assert trimmed_rss >= 0.0


def test_embedding_encoder_lazy_loading():
    """Verifies that EmbeddingEncoder does NOT load the heavy model weights on instantiation."""
    encoder = EmbeddingEncoder(model_name="all-MiniLM-L6-v2")
    # Model should not be loaded on init
    assert encoder.is_loaded is False
    assert encoder._model is None

    # Embedding dimension must be known without model loading
    assert encoder.embedding_dim == 384
    assert encoder.is_loaded is False


def test_cross_encoder_lazy_loading():
    """Verifies that CrossEncoderReranker does NOT load the model on instantiation."""
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    assert reranker.is_loaded is False
    assert reranker._model is None


def test_singleton_caching_embedding_encoder():
    """Verifies that get_embedding_encoder returns the exact same singleton instance."""
    inst1 = get_embedding_encoder()
    inst2 = get_embedding_encoder()
    assert inst1 is inst2


def test_singleton_caching_cross_encoder():
    """Verifies that get_cross_encoder_reranker returns the exact same singleton instance."""
    inst1 = get_cross_encoder_reranker()
    inst2 = get_cross_encoder_reranker()
    assert inst1 is inst2


def test_reranker_graceful_fallback_on_load_failure():
    """Verifies that if the cross-encoder fails to load (e.g. OOM), it gracefully preserves candidate rank."""
    reranker = CrossEncoderReranker(model_name="nonexistent-model-to-simulate-failure")
    candidates = [
        RetrievalResultItem(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Overheating motor bearing failure",
            fused_score=0.85,
            fused_rank=1,
            final_rank=1,
            metadata=ChunkMetadata(
                chunk_id="chunk-1",
                document_id="doc-1",
                source_filename="doc1.md",
                chunk_index=0,
                machine_id="MTR-01",
                incident_type="Overheating",
            ),
        ),
        RetrievalResultItem(
            chunk_id="chunk-2",
            document_id="doc-2",
            text="Normal turbine operations log",
            fused_score=0.45,
            fused_rank=2,
            final_rank=2,
            metadata=ChunkMetadata(
                chunk_id="chunk-2",
                document_id="doc-2",
                source_filename="doc2.md",
                chunk_index=0,
                machine_id="TRB-02",
                incident_type="Vibration",
            ),
        ),
    ]

    with patch.object(reranker, "_get_model", return_value=None):
        results, latency = reranker.rerank("motor bearing overheating", candidates, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "chunk-1"
        assert results[1].chunk_id == "chunk-2"
        assert latency >= 0.0


def test_health_endpoint_does_not_force_model_loading(api_test_client):
    """Verifies that calling /health endpoint does not force loading heavy ML models."""
    response = api_test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "process_memory_mb" in data
    assert "embedding_model_loaded" in data
    assert "reranker_model_loaded" in data
    assert "reranker_enabled" in data
    assert "auto_seed_data" in data
    assert "embedding_batch_size" in data
    assert data["auto_seed_data"] is False
    assert data["reranker_enabled"] is False
    assert data["embedding_batch_size"] == 2


def test_cross_encoder_disabled_by_default_and_rrf_fallback():
    """Verifies that CrossEncoder does not load when disabled and uses RRF fallback."""
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", enabled=False)
    assert reranker.is_enabled is False
    assert reranker.is_loaded is False
    assert reranker._get_model() is None

    candidates = [
        RetrievalResultItem(
            chunk_id="c1",
            document_id="doc1",
            text="Overheating pump bearing",
            dense_score=0.9,
            bm25_score=4.0,
            rrf_score=0.03,
            final_rank=1,
            metadata=ChunkMetadata(
                chunk_id="c1",
                document_id="doc1",
                chunk_index=0,
                source_filename="doc1.md",
            ),
        ),
        RetrievalResultItem(
            chunk_id="c2",
            document_id="doc1",
            text="Cooling system valve inspection",
            dense_score=0.7,
            bm25_score=2.0,
            rrf_score=0.015,
            final_rank=2,
            metadata=ChunkMetadata(
                chunk_id="c2",
                document_id="doc1",
                chunk_index=1,
                source_filename="doc1.md",
            ),
        ),
    ]

    reranked, latency = reranker.rerank("bearing issue", candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"
    assert reranked[0].final_rank == 1
    assert reranked[1].chunk_id == "c2"
    assert reranked[1].final_rank == 2
    # Ensure model was never loaded into memory
    assert reranker.is_loaded is False


def test_embedding_batch_size_and_singleton_model_cache():
    """Verifies embedding model singleton caching and configurable batch encoding."""
    enc1 = EmbeddingEncoder(model_name="test-singleton-model")
    enc2 = EmbeddingEncoder(model_name="test-singleton-model")

    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: [[0.1] * 384 for _ in texts]
    enc1.model = mock_model

    # enc2 should share the same underlying model through the singleton cache
    assert enc2.model is mock_model
    assert enc1.is_loaded is True
    assert enc2.is_loaded is True

    # Test batch encoding with batch size 2
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    embeddings = enc1.encode_documents(chunks, batch_size=2)
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 384
    # With 3 chunks and batch_size 2, encode should have been called twice (2 + 1)
    assert mock_model.encode.call_count == 2


def test_settings_endpoint_reports_low_memory_configs(api_test_client):
    """Verifies that /api/v1/settings reports the low-memory environment configuration."""
    response = api_test_client.get("/api/v1/settings")
    assert response.status_code == 200
    data = response.json()
    assert data["auto_seed_data"] is False
    assert data["enable_reranker"] is False
    assert data["embedding_batch_size"] == 2
    assert data["max_concurrent_uploads"] == 1


def test_simultaneous_concurrent_uploads_atomic_race_condition(api_test_client):
    """Verifies that when two upload requests are launched simultaneously, exactly one succeeds and the other receives HTTP 429."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from incident_rag.ingestion.service import IngestionService
    from incident_rag.mem_utils import get_upload_limiter

    limiter = get_upload_limiter()
    assert limiter.is_locked is False

    original_ingest = IngestionService.ingest_bytes

    def slow_ingest(self, content_bytes, filename, force_reindex=False):
        # Hold execution slot for a fraction of a second to force concurrency overlap
        time.sleep(0.35)
        return original_ingest(self, content_bytes, filename, force_reindex)

    with patch.object(IngestionService, "ingest_bytes", slow_ingest):
        files1 = {"file": ("simultaneous_1.md", b"# Incident A\nMotor shaft cracked.", "text/markdown")}
        files2 = {"file": ("simultaneous_2.md", b"# Incident B\nCooling fan tripped.", "text/markdown")}

        def send_upload(file_data):
            return api_test_client.post("/api/v1/documents/upload", files=file_data)

        with ThreadPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(send_upload, files1)
            time.sleep(0.04)  # Small delay to ensure fut1 has acquired the atomic limiter
            fut2 = executor.submit(send_upload, files2)

            resp1 = fut1.result()
            resp2 = fut2.result()

        status_codes = sorted([resp1.status_code, resp2.status_code])
        # Exactly one must succeed (200) and the concurrent one must be rejected (429)
        assert status_codes == [200, 429]

        resp_429 = resp1 if resp1.status_code == 429 else resp2
        assert "Upload concurrency limit reached" in resp_429.json()["detail"]
        assert resp_429.headers.get("Retry-After") == "5"

    # Ensure limiter is fully released after simultaneous execution finishes
    assert limiter.is_locked is False

    # A subsequent upload must now succeed normally
    files3 = {"file": ("simultaneous_3.md", b"# Incident C\nSubsequent upload succeeds.", "text/markdown")}
    resp3 = api_test_client.post("/api/v1/documents/upload", files=files3)
    assert resp3.status_code == 200
    assert limiter.is_locked is False


def test_upload_limiter_released_on_validation_failure(api_test_client):
    """Verifies that the upload limiter is released immediately if format, size, or content validation fails."""
    from incident_rag.mem_utils import get_upload_limiter
    limiter = get_upload_limiter()
    assert limiter.is_locked is False

    # 1. Invalid file extension (.bin)
    resp_invalid_ext = api_test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("bad_file.bin", b"\x00\x01\x02", "application/octet-stream")},
    )
    assert resp_invalid_ext.status_code == 400
    assert limiter.is_locked is False

    # 2. Empty file (0 bytes)
    resp_empty = api_test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp_empty.status_code == 400
    assert limiter.is_locked is False

    # 3. Oversized file (> 25MB)
    resp_oversized = api_test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("huge.txt", b"A" * (26 * 1024 * 1024), "text/plain")},
    )
    assert resp_oversized.status_code == 413
    assert limiter.is_locked is False


def test_upload_lock_releases_after_exception(api_test_client):
    """Verifies that the upload limiter is cleanly released even when document parsing or ingestion raises an exception."""
    from incident_rag.mem_utils import get_upload_limiter
    limiter = get_upload_limiter()
    assert limiter.is_locked is False

    # Simulate an upload that causes an unhandled runtime failure
    with patch("incident_rag.ingestion.service.IngestionService.ingest_bytes", side_effect=RuntimeError("Simulated ingestion disk failure")):
        files = {"file": ("faulty_report.md", b"# Faulty content", "text/markdown")}
        resp = api_test_client.post("/api/v1/documents/upload", files=files)
        assert resp.status_code == 500

    # Ensure the upload lock was released in the finally block
    assert limiter.is_locked is False


def test_health_check_never_blocked_during_upload(api_test_client):
    """Verifies that /health is never blocked while an upload is holding the upload concurrency limiter."""
    from incident_rag.mem_utils import get_upload_limiter
    limiter = get_upload_limiter()

    # Acquire the limiter to simulate active upload
    acquired = limiter.try_acquire()
    assert acquired is True
    assert limiter.is_locked is True

    try:
        # /health must still respond with 200 OK immediately
        health_resp = api_test_client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"
    finally:
        limiter.release()
        assert limiter.is_locked is False


def test_process_wide_embedding_lock_serialization():
    """Verifies that get_embedding_execution_lock serializes batch encoding across instances."""
    from incident_rag.mem_utils import get_embedding_execution_lock
    lock = get_embedding_execution_lock()
    assert lock.locked() is False

    enc = EmbeddingEncoder(model_name="all-MiniLM-L6-v2")
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda texts, **kwargs: [[0.2] * 384 for _ in texts]
    enc.model = mock_model

    # Encode with lock
    results = enc.encode_documents(["Chunk A", "Chunk B"], batch_size=2)
    assert len(results) == 2
    # Ensure lock is released after execution
    assert lock.locked() is False

    # Test lock release after exception
    mock_model.encode.side_effect = RuntimeError("Encoding failure")
    with pytest.raises(RuntimeError):
        enc.encode_documents(["Chunk C"], batch_size=2)
    assert lock.locked() is False


def test_encode_query_uses_shared_embedding_lock():
    """Verifies that encode_query (used in QA and document analysis) executes under the shared embedding lock."""
    from incident_rag.mem_utils import get_embedding_execution_lock
    import numpy as np

    lock = get_embedding_execution_lock()
    assert lock.locked() is False

    enc = EmbeddingEncoder(model_name="all-MiniLM-L6-v2")
    lock_was_held = False

    def mock_encode_query(text, **kwargs):
        nonlocal lock_was_held
        lock_was_held = lock.locked()
        return np.array([0.5] * 384)

    mock_model = MagicMock()
    mock_model.encode.side_effect = mock_encode_query
    enc.model = mock_model

    result = enc.encode_query("What caused the compressor seal failure?")
    assert len(result) == 384
    assert lock_was_held is True
    # Lock must be released after completion
    assert lock.locked() is False

    # Test lock release on exception
    mock_model.encode.side_effect = RuntimeError("Query inference failed")
    with pytest.raises(RuntimeError):
        enc.encode_query("Failing query")
    assert lock.locked() is False


def test_chunked_upload_within_limit_succeeds(api_test_client):
    """Verifies that an upload within the configured limit succeeds via chunked streaming."""
    from incident_rag.mem_utils import get_upload_limiter
    limiter = get_upload_limiter()
    assert limiter.is_locked is False

    valid_content = b"# Incident Test\nPump motor bearing overheating observed at 14:00."
    resp = api_test_client.post(
        "/api/v1/documents/upload",
        files={"file": ("valid_chunked.md", valid_content, "text/markdown")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] in ["indexed", "duplicate"]
    assert limiter.is_locked is False


def test_chunked_upload_exceeding_10mb_returns_413(api_test_client):
    """Verifies that an upload exceeding a configured 10 MB limit returns HTTP 413 and releases limiter."""
    from incident_rag.config import get_settings
    from incident_rag.mem_utils import get_upload_limiter

    settings = get_settings()
    limiter = get_upload_limiter()
    assert limiter.is_locked is False

    # Simulate 10 MB limit as configured for Render
    with patch.object(settings, "MAX_UPLOAD_SIZE_MB", 10):
        oversized_data = b"X" * (10 * 1024 * 1024 + 10240)  # 10MB + 10KB
        resp = api_test_client.post(
            "/api/v1/documents/upload",
            files={"file": ("too_large.txt", oversized_data, "text/plain")},
        )
        assert resp.status_code == 413
        assert "exceeds maximum permitted limit of 10MB" in resp.json()["detail"]
        assert limiter.is_locked is False


def test_oversized_file_stops_reading_early():
    """Verifies that chunked reading aborts immediately when max_file_size is breached without reading the rest."""
    import asyncio
    from unittest.mock import AsyncMock
    from fastapi import UploadFile
    from incident_rag.api import upload_document
    from incident_rag.config import get_settings
    from incident_rag.mem_utils import get_upload_limiter

    settings = get_settings()
    limiter = get_upload_limiter()

    # 64 KB chunks: return 3 chunks (192 KB total), with max limit configured to 100 KB
    chunk_data = b"A" * (64 * 1024)
    mock_upload = AsyncMock(spec=UploadFile)
    mock_upload.filename = "test_early_stop.txt"
    # Stream produces 5 chunks, but after chunk 2 (128 KB > 100 KB), reading MUST stop
    mock_upload.read.side_effect = [chunk_data, chunk_data, chunk_data, chunk_data, b""]
    mock_upload.close = AsyncMock()

    mock_ingestion = MagicMock()

    # Configure MAX_UPLOAD_SIZE_MB to a tiny fractional limit (0.0001 MB ~ 100 bytes) for testing
    with patch.object(settings, "MAX_UPLOAD_SIZE_MB", 1):
        # With max 1MB = 1048576 bytes, send chunks until over 1MB
        big_chunk = b"B" * (600 * 1024)
        mock_upload.read.side_effect = [big_chunk, big_chunk, big_chunk, b""]

        with pytest.raises(Exception) as exc_info:
            asyncio.run(upload_document(file=mock_upload, ingestion_service=mock_ingestion))

        from fastapi import HTTPException
        assert isinstance(exc_info.value, HTTPException)
        assert exc_info.value.status_code == 413
        # Ingestion service must NEVER be called for oversized file
        assert mock_ingestion.ingest_bytes.called is False
        # Reading should have aborted on 2nd chunk (600KB + 600KB = 1.2MB > 1MB), so 3rd chunk was never read
        assert mock_upload.read.call_count == 2
        # File must be closed safely
        assert mock_upload.close.called is True
        # Limiter must be released
        assert limiter.is_locked is False


def test_pdf_resource_cleanup_on_error():
    """Verifies that PyMuPDF Document is safely closed even when text extraction throws an error."""
    from incident_rag.ingestion.loader import load_bytes

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = RuntimeError("Corrupt font descriptor")

    with patch("pymupdf.open", return_value=mock_doc):
        with pytest.raises(ValueError, match="Corrupt or unreadable PDF document"):
            load_bytes(b"%PDF-1.4 simulated bytes", "corrupted.pdf")

        # PyMuPDF document must be closed in finally block despite the extraction error
        assert mock_doc.close.called is True
