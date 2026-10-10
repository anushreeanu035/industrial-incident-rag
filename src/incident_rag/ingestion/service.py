"""Ingestion service orchestrating loading, parsing, chunking, embedding, and storage."""

from pathlib import Path
from typing import List, Optional, Tuple, Union
from incident_rag.config import get_settings
from incident_rag.embeddings.encoder import EmbeddingEncoder, get_embedding_encoder
from incident_rag.ingestion.chunker import chunk_document
from incident_rag.ingestion.loader import LoadedDocument, load_bytes, load_file
from incident_rag.ingestion.metadata import extract_metadata_from_text
from incident_rag.logging_config import get_logger
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
        """Processes raw file bytes, chunks, generates embeddings, and indexes them."""
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

        # Generate dense embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.encoder.encode_documents(texts)

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
            page_count=len(loaded_doc.pages),
            chunk_count=len(chunks),
            status="indexed",
            error_message=None,
        )
        self.doc_store.add_document(doc_meta)

        logger.info(
            f"Successfully ingested '{filename}' (ID: {loaded_doc.document_id}, "
            f"chunks: {len(chunks)}, machine_id: {extracted_meta.get('machine_id')})"
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
        """Scans a directory for supported document files and ingests them."""
        results = []
        supported_exts = {".pdf", ".md", ".txt", ".markdown"}
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in supported_exts:
                resp = self.ingest_file(f, force_reindex=force_reindex)
                results.append(resp)
        return results

    def delete_document(self, document_id: str) -> bool:
        """Deletes a document from vector store, BM25, and document store."""
        self.vector_store.delete_document(document_id)
        if self.bm25_engine is not None:
            self.bm25_engine.remove_document_chunks(document_id)
        return self.doc_store.delete_document(document_id)
