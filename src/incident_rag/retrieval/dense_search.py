"""Dense semantic search engine utilizing SentenceTransformers and Qdrant."""

from typing import List, Optional
from incident_rag.embeddings.encoder import EmbeddingEncoder, get_embedding_encoder
from incident_rag.logging_config import get_logger
from incident_rag.schemas import RetrievalResultItem
from incident_rag.storage.vector_store import VectorStore

logger = get_logger(__name__)


class DenseSearchEngine:
    """Executes dense vector similarity search over indexed chunks."""

    def __init__(self, vector_store: VectorStore, encoder: Optional[EmbeddingEncoder] = None):
        self.vector_store = vector_store
        self.encoder = encoder or get_embedding_encoder()

    def search(
        self,
        query: str,
        top_k: int = 10,
        machine_id_filter: Optional[str] = None,
        incident_type_filter: Optional[str] = None,
    ) -> List[RetrievalResultItem]:
        """Encodes query and queries vector store."""
        if not query.strip():
            return []

        query_vector = self.encoder.encode_query(query)
        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            machine_id_filter=machine_id_filter,
            incident_type_filter=incident_type_filter,
        )
        return results
