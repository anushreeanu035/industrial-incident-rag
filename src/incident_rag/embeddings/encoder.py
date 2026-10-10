"""Sentence Transformers dense embedding encoder."""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger

logger = get_logger(__name__)

_ENCODER_INSTANCE: Optional["EmbeddingEncoder"] = None


class EmbeddingEncoder:
    """Encodes texts into dense vector representations using Sentence Transformers."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def encode_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encodes a list of document chunks into normalized vectors."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """Encodes a single search query into a normalized vector."""
        embedding = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()


def get_embedding_encoder(model_name: Optional[str] = None) -> EmbeddingEncoder:
    """Returns singleton instance of EmbeddingEncoder."""
    global _ENCODER_INSTANCE
    if _ENCODER_INSTANCE is None or (model_name and _ENCODER_INSTANCE.model_name != model_name):
        _ENCODER_INSTANCE = EmbeddingEncoder(model_name=model_name)
    return _ENCODER_INSTANCE
