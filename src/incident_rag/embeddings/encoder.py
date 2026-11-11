"""Sentence Transformers dense embedding encoder with lazy loading, thread safety, and memory management."""

import threading
import time
from typing import Any, Dict, List, Optional, Union
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.mem_utils import (
    configure_runtime_memory_limits,
    get_embedding_execution_lock,
    get_process_memory_mb,
    trim_memory,
)

logger = get_logger(__name__)

_ENCODER_INSTANCE: Optional["EmbeddingEncoder"] = None
_ENCODER_LOCK = threading.Lock()

# Process-level model weight cache to prevent loading duplicate model instances across threads or instances
_SHARED_MODELS: Dict[str, Any] = {}
_SHARED_MODELS_LOCK = threading.Lock()

# Mapping of standard model names to their known embedding dimensions
# This allows vector stores and schema validations to know the dimension WITHOUT loading the model weights
KNOWN_EMBEDDING_DIMS = {
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class EmbeddingEncoder:
    """Encodes texts into dense vector representations with on-demand lazy model loading."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._lock = threading.Lock()
        self._embedding_dim = KNOWN_EMBEDDING_DIMS.get(self.model_name, getattr(settings, "EMBEDDING_DIM", 384))

    @property
    def embedding_dim(self) -> int:
        """Returns the embedding dimension without forcing model initialization."""
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        """Returns whether this encoder currently has the model loaded in memory without forcing initialization."""
        return self._model is not None

    def _get_model(self):
        """Thread-safe lazy initialization of SentenceTransformer model with singleton caching."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    # Check module-level shared cache first to avoid duplicate model instances
                    with _SHARED_MODELS_LOCK:
                        if self.model_name in _SHARED_MODELS:
                            self._model = _SHARED_MODELS[self.model_name]
                            return self._model

                    configure_runtime_memory_limits()
                    rss_before = get_process_memory_mb()
                    logger.info(
                        f"Initializing SentenceTransformer embedding model: {self.model_name} "
                        f"(Current RSS: {rss_before:.1f} MB)"
                    )
                    start_t = time.perf_counter()
                    try:
                        import torch
                        from sentence_transformers import SentenceTransformer

                        with torch.inference_mode():
                            # Attempt offline local files first to avoid remote HTTPS hang and HuggingFace rate-limits
                            try:
                                model = SentenceTransformer(self.model_name, device="cpu", local_files_only=True)
                            except Exception:
                                logger.info(f"Local model weights not found for '{self.model_name}', falling back to hub...")
                                model = SentenceTransformer(self.model_name, device="cpu")
                            model.eval()

                        if hasattr(model, "get_embedding_dimension"):
                            self._embedding_dim = model.get_embedding_dimension()
                        elif hasattr(model, "get_sentence_embedding_dimension"):
                            self._embedding_dim = model.get_sentence_embedding_dimension()

                        self._model = model
                        with _SHARED_MODELS_LOCK:
                            _SHARED_MODELS[self.model_name] = model

                        rss_after = trim_memory()
                        dur = (time.perf_counter() - start_t) * 1000.0
                        logger.info(
                            f"SentenceTransformer loaded in {dur:.0f}ms. "
                            f"Dimension: {self._embedding_dim} (RSS: {rss_after:.1f} MB, diff: +{rss_after - rss_before:.1f} MB)"
                        )
                    except (MemoryError, Exception) as e:
                        logger.error(f"Failed to load SentenceTransformer model '{self.model_name}': {e}", exc_info=True)
                        trim_memory()
                        raise RuntimeError(
                            f"Failed to load embedding model '{self.model_name}'. "
                            f"Ensure memory limit is sufficient. Error: {str(e)}"
                        ) from e
        return self._model

    @property
    def model(self):
        """Underlying model instance (lazily loaded)."""
        return self._get_model()

    @model.setter
    def model(self, value):
        """Allows injecting a mock model for testing or overriding."""
        self._model = value
        with _SHARED_MODELS_LOCK:
            if value is not None:
                _SHARED_MODELS[self.model_name] = value
            else:
                _SHARED_MODELS.pop(self.model_name, None)

    def encode_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Encodes document chunks in small batches to strictly constrain peak memory.

        Uses process-wide get_embedding_execution_lock() so multiple instances and threads
        cannot trigger overlapping batch tensor allocations.
        """
        if not texts:
            return []

        settings = get_settings()
        configured_batch = batch_size if batch_size is not None else getattr(settings, "EMBEDDING_BATCH_SIZE", 32)
        effective_batch_size = max(1, min(configured_batch, 128))

        with get_embedding_execution_lock():
            model = self._get_model()
            import torch

            rss_before = get_process_memory_mb()
            logger.debug(
                f"Encoding {len(texts)} chunks (batch_size: {effective_batch_size}, RSS before: {rss_before:.1f} MB)"
            )

            all_embeddings: List[List[float]] = []

            try:
                with torch.inference_mode():
                    for i in range(0, len(texts), effective_batch_size):
                        batch_texts = texts[i : i + effective_batch_size]
                        batch_emb = model.encode(
                            batch_texts,
                            batch_size=len(batch_texts),
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                        )
                        emb_list = batch_emb.tolist() if hasattr(batch_emb, "tolist") else batch_emb
                        all_embeddings.extend(emb_list)
                        del batch_texts, batch_emb

                rss_after = trim_memory()
                logger.debug(
                    f"Completed encoding {len(texts)} chunks. (RSS after trim: {rss_after:.1f} MB)"
                )
                return all_embeddings
            except (MemoryError, Exception) as e:
                logger.error(f"Error during document batch encoding: {e}", exc_info=True)
                trim_memory()
                raise RuntimeError(f"Embedding generation failed: {str(e)}") from e

    def encode_query(self, query: str) -> List[float]:
        """Encodes a single search query into a normalized vector.

        Uses process-wide get_embedding_execution_lock() so query encoding cannot
        overlap with document batch encoding or concurrent query requests.
        """
        if not query or not query.strip():
            return [0.0] * self._embedding_dim

        with get_embedding_execution_lock():
            model = self._get_model()
            import torch

            try:
                with torch.inference_mode():
                    embedding = model.encode(
                        query.strip(),
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                result = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                del embedding
                return result
            except (MemoryError, Exception) as e:
                logger.error(f"Error encoding query: {e}", exc_info=True)
                trim_memory()
                raise RuntimeError(f"Query embedding generation failed: {str(e)}") from e


def get_embedding_encoder(model_name: Optional[str] = None) -> EmbeddingEncoder:
    """Returns singleton instance of EmbeddingEncoder."""
    global _ENCODER_INSTANCE
    if _ENCODER_INSTANCE is None or (model_name and _ENCODER_INSTANCE.model_name != model_name):
        with _ENCODER_LOCK:
            if _ENCODER_INSTANCE is None or (model_name and _ENCODER_INSTANCE.model_name != model_name):
                _ENCODER_INSTANCE = EmbeddingEncoder(model_name=model_name)
    return _ENCODER_INSTANCE
