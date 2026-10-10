"""Qdrant vector store implementation for dense retrieval with local persistence."""

from pathlib import Path
from typing import List, Optional, Union
import uuid
import qdrant_client
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import ChunkMetadata, DocumentChunk, RetrievalResultItem

logger = get_logger(__name__)


class VectorStore:
    """Interface to local Qdrant vector database for chunk embeddings and metadata."""

    def __init__(
        self,
        collection_name: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        vector_dim: int = 384,
        in_memory: bool = False,
    ):
        settings = get_settings()
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_dim = vector_dim

        if in_memory:
            logger.info("Initializing in-memory Qdrant client")
            self.client = qdrant_client.QdrantClient(location=":memory:")
        else:
            path = db_path or settings.resolve_path(settings.QDRANT_PATH)
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Initializing Qdrant client with local path: {path}")
            self.client = qdrant_client.QdrantClient(path=str(path))

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(f"Creating Qdrant collection: {self.collection_name} (dim={self.vector_dim})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Stores chunk vectors and metadata into Qdrant."""
        if not chunks or not embeddings:
            return

        points: List[PointStruct] = []
        for chunk, emb in zip(chunks, embeddings):
            # Compute deterministic UUID for Qdrant local mode
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "chunk_index": chunk.metadata.chunk_index,
                "page_number": chunk.metadata.page_number,
                "start_char": chunk.metadata.start_char,
                "end_char": chunk.metadata.end_char,
                "machine_id": chunk.metadata.machine_id,
                "incident_type": chunk.metadata.incident_type,
                "incident_date": chunk.metadata.incident_date,
                "source_filename": chunk.metadata.source_filename,
            }
            points.append(PointStruct(id=point_id, vector=emb, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} chunks into Qdrant collection '{self.collection_name}'")

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        machine_id_filter: Optional[str] = None,
        incident_type_filter: Optional[str] = None,
    ) -> List[RetrievalResultItem]:
        """Performs dense vector similarity search with optional metadata filters."""
        filter_conditions = []
        if machine_id_filter:
            filter_conditions.append(
                FieldCondition(key="machine_id", match=MatchValue(value=machine_id_filter))
            )
        if incident_type_filter:
            filter_conditions.append(
                FieldCondition(key="incident_type", match=MatchValue(value=incident_type_filter))
            )

        query_filter = Filter(must=filter_conditions) if filter_conditions else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

        results: List[RetrievalResultItem] = []
        for rank_idx, point in enumerate(response.points, start=1):
            p = point.payload
            meta = ChunkMetadata(
                chunk_id=p["chunk_id"],
                document_id=p["document_id"],
                chunk_index=p["chunk_index"],
                page_number=p.get("page_number"),
                start_char=p.get("start_char", 0),
                end_char=p.get("end_char", 0),
                machine_id=p.get("machine_id"),
                incident_type=p.get("incident_type"),
                incident_date=p.get("incident_date"),
                source_filename=p.get("source_filename", ""),
            )
            item = RetrievalResultItem(
                chunk_id=p["chunk_id"],
                document_id=p["document_id"],
                text=p["text"],
                metadata=meta,
                dense_score=float(point.score),
                dense_rank=rank_idx,
                final_rank=rank_idx,
            )
            results.append(item)

        return results

    def delete_document(self, document_id: str) -> None:
        """Deletes all chunks belonging to a document from Qdrant."""
        doc_filter = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        self.client.delete(collection_name=self.collection_name, points_selector=doc_filter)
        logger.info(f"Deleted chunks for document {document_id} from Qdrant")

    def count_chunks(self) -> int:
        """Returns the total number of indexed vectors."""
        res = self.client.count(collection_name=self.collection_name)
        return res.count

    def clear_collection(self) -> None:
        """Clears all vectors in the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()
