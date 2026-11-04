"""Ingestion service orchestrating loading, parsing, chunking, embedding, and storage."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from incident_rag.config import get_settings
from incident_rag.embeddings.encoder import EmbeddingEncoder, get_embedding_encoder
from incident_rag.ingestion.chunker import chunk_document
from incident_rag.ingestion.loader import LoadedDocument, load_bytes, load_file
from incident_rag.ingestion.metadata import extract_metadata_from_text
from incident_rag.logging_config import get_logger
from incident_rag.mem_utils import (
    get_ingestion_execution_lock,
    get_process_memory_mb,
    trim_memory,
)
from incident_rag.schemas import DocumentChunk, DocumentMetadata, DocumentUploadResponse
from incident_rag.storage.document_store import DocumentStore
from incident_rag.storage.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionService:
    """End-to-end ingestion pipeline for industrial incident documents."""

    def __init__(
        self,
        document_store: Optional[DocumentStore] = None,
        vector_store: Optional[VectorStore] = None,
        encoder: Optional[EmbeddingEncoder] = None,
        bm25_engine=None,
    ):
        self.settings = get_settings()
        self.doc_store = document_store or DocumentStore()
        self.encoder = encoder or get_embedding_encoder()
        self.vector_store = vector_store or VectorStore(vector_dim=self.encoder.embedding_dim)
        self.bm25_engine = bm25_engine

    def ingest_bytes(
        self,
        content_bytes: bytes,
        filename: str,
        force_reindex: bool = False,
    ) -> DocumentUploadResponse:
        """Processes raw file bytes, chunks, generates embeddings, and indexes them.

        Uses process-wide get_ingestion_execution_lock() so simultaneous uploads cannot
        trigger overlapping ingestion runs.
        """
        with get_ingestion_execution_lock():
            try:
                loaded_doc = load_bytes(content_bytes, filename)
            except Exception as e:
                logger.error(f"Failed loading document '{filename}': {e}")
                raise ValueError(f"Failed to load document '{filename}': {str(e)}")

            # Check for existing document version
            existing = self.doc_store.find_by_checksum(loaded_doc.checksum_sha256)
            if existing and not force_reindex:
                logger.info(f"Document '{filename}' already indexed (ID: {existing.document_id}). Skipping.")
                return DocumentUploadResponse(
                    document_id=existing.document_id,
                    filename=filename,
                    chunks_created=existing.chunk_count,
                    status=existing.status,
                    message="Document version already indexed. Skipping duplicate.",
                )

            # Extract incident-specific metadata
            extracted_meta = extract_metadata_from_text(loaded_doc.full_text, filename)

            # Split into structured chunks
            chunks = chunk_document(
                doc=loaded_doc,
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP,
                extracted_meta=extracted_meta,
            )

            if not chunks:
                error_msg = f"No chunks created from document '{filename}'"
                logger.warning(error_msg)
                doc_meta = DocumentMetadata(
                    document_id=loaded_doc.document_id,
                    filename=filename,
                    file_type=loaded_doc.file_type,
                    file_size_bytes=loaded_doc.file_size_bytes,
                    checksum_sha256=loaded_doc.checksum_sha256,
                    machine_id=extracted_meta.get("machine_id"),
                    incident_type=extracted_meta.get("incident_type"),
                    incident_date=extracted_meta.get("incident_date"),
                    page_count=len(loaded_doc.pages),
                    chunk_count=0,
                    status="failed",
                    error_message=error_msg,
                )
                self.doc_store.add_document(doc_meta)
                return DocumentUploadResponse(
                    document_id=loaded_doc.document_id,
                    filename=filename,
                    chunks_created=0,
                    status="failed",
                    message=error_msg,
                )

            # If re-indexing, remove prior chunks first
            if existing and force_reindex:
                self.delete_document(existing.document_id)

            # Generate dense embeddings using configured low-memory batch size
            texts = [chunk.text for chunk in chunks]
            batch_size = getattr(self.settings, "EMBEDDING_BATCH_SIZE", 2)
            rss_before_enc = get_process_memory_mb()
            logger.info(
                f"Ingesting '{filename}' ({len(chunks)} chunks, embedding_batch_size: {batch_size}, "
                f"RSS before: {rss_before_enc:.1f} MB)"
            )
            try:
                embeddings = self.encoder.encode_documents(texts, batch_size=batch_size)
            except (MemoryError, Exception) as enc_err:
                logger.error(f"Failed to generate embeddings during ingestion of '{filename}': {enc_err}", exc_info=True)
                trim_memory()
                raise RuntimeError(f"Embedding generation failed for document '{filename}': {str(enc_err)}") from enc_err

            # Index in Qdrant vector store
            self.vector_store.upsert_chunks(chunks, embeddings)

            # Update BM25 keyword index if configured
            if self.bm25_engine is not None:
                self.bm25_engine.add_chunks(chunks)

            # Persist document metadata
            doc_meta = DocumentMetadata(
                document_id=loaded_doc.document_id,
                filename=filename,
                file_type=loaded_doc.file_type,
                file_size_bytes=loaded_doc.file_size_bytes,
                checksum_sha256=loaded_doc.checksum_sha256,
                machine_id=extracted_meta.get("machine_id"),
                incident_type=extracted_meta.get("incident_type"),
                incident_date=extracted_meta.get("incident_date"),
                incident_id=extracted_meta.get("incident_id"),
                facility=extracted_meta.get("facility"),
                severity_level=extracted_meta.get("severity_level"),
                lead_investigator=extracted_meta.get("lead_investigator"),
                page_count=len(loaded_doc.pages),
                chunk_count=len(chunks),
                status="indexed",
                error_message=None,
            )
            self.doc_store.add_document(doc_meta)

            trim_memory()

            logger.info(
                f"Successfully ingested '{filename}' (ID: {loaded_doc.document_id}, "
                f"chunks: {len(chunks)}, machine_id: {extracted_meta.get('machine_id')}, "
                f"RSS: {get_process_memory_mb():.1f} MB)"
            )

            return DocumentUploadResponse(
                document_id=loaded_doc.document_id,
                filename=filename,
                chunks_created=len(chunks),
                status="indexed",
                message=f"Successfully indexed {len(chunks)} chunks.",
            )

    def ingest_file(self, file_path: Path, force_reindex: bool = False) -> DocumentUploadResponse:
        """Ingests a file from the filesystem."""
        content_bytes = file_path.read_bytes()
        return self.ingest_bytes(content_bytes, file_path.name, force_reindex=force_reindex)

    def ingest_directory(self, dir_path: Path, force_reindex: bool = False) -> List[DocumentUploadResponse]:
        """Scans a directory for supported document files and ingests them sequentially with memory cleanup."""
        results = []
        supported_exts = {".pdf", ".md", ".txt", ".markdown"}
        for f in sorted(dir_path.iterdir()):
            if f.is_file() and f.suffix.lower() in supported_exts:
                resp = self.ingest_file(f, force_reindex=force_reindex)
                results.append(resp)
                trim_memory()
        return results

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document from vector store, BM25, and document store."""
        self.vector_store.delete_document(document_id)
        if self.bm25_engine is not None:
            self.bm25_engine.remove_document_chunks(document_id)
        return self.doc_store.delete_document(document_id)
