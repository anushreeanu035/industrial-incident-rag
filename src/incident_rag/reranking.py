"""Cross-encoder reranking module using Sentence Transformers CrossEncoder with lazy loading and memory controls."""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.mem_utils import (
    configure_runtime_memory_limits,
    get_process_memory_mb,
    trim_memory,
)
from incident_rag.schemas import RetrievalResultItem

logger = get_logger(__name__)

_RERANKER_INSTANCE: Optional["CrossEncoderReranker"] = None
_RERANKER_LOCK = threading.Lock()

# Process-level cross-encoder weight cache
_SHARED_RERANKERS: Dict[str, Any] = {}
_SHARED_RERANKERS_LOCK = threading.Lock()


class CrossEncoderReranker:
    """Reranks retrieved candidate chunks using a trained cross-encoder model loaded on demand."""

    def __init__(self, model_name: Optional[str] = None, enabled: Optional[bool] = None):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        self._enabled = enabled if enabled is not None else getattr(settings, "ENABLE_RERANKER", False)
        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False

    @property
    def is_enabled(self) -> bool:
        """Returns whether CrossEncoder reranking is enabled in configuration."""
        return getattr(self, "_enabled", False)

    @property
    def is_loaded(self) -> bool:
        """Returns whether this reranker currently has the cross-encoder model loaded in memory."""
        return hasattr(self, "_model") and self._model is not None

    def _get_model(self):
        """Thread-safe lazy initialization of CrossEncoder model when enabled."""
        if not getattr(self, "_enabled", True):
            return None

        if not hasattr(self, "_model"):
            self._model = None
        if not hasattr(self, "_lock"):
            self._lock = threading.Lock()
        if not hasattr(self, "_load_failed"):
            self._load_failed = False

        if self._model is None and not self._load_failed:
            with self._lock:
                if self._model is None and not self._load_failed:
                    # Check shared cache first
                    with _SHARED_RERANKERS_LOCK:
                        if self.model_name in _SHARED_RERANKERS:
                            self._model = _SHARED_RERANKERS[self.model_name]
                            return self._model

                    configure_runtime_memory_limits()
                    rss_before = get_process_memory_mb()
                    logger.info(
                        f"Initializing CrossEncoder reranker model: {self.model_name} "
                        f"(Current RSS: {rss_before:.1f} MB)"
                    )
                    start_t = time.perf_counter()
                    try:
                        import torch
                        from sentence_transformers import CrossEncoder

                        with torch.inference_mode():
                            model = CrossEncoder(self.model_name, device="cpu")

                        self._model = model
                        with _SHARED_RERANKERS_LOCK:
                            _SHARED_RERANKERS[self.model_name] = model

                        rss_after = trim_memory()
                        dur = (time.perf_counter() - start_t) * 1000.0
                        logger.info(
                            f"CrossEncoder model loaded successfully in {dur:.0f}ms. "
                            f"(RSS: {rss_after:.1f} MB, diff: +{rss_after - rss_before:.1f} MB)"
                        )
                    except (MemoryError, Exception) as e:
                        self._load_failed = True
                        logger.error(
                            f"Failed to load CrossEncoder model '{self.model_name}': {e}. "
                            "Reranker will gracefully fall back to reciprocal rank fusion ordering.",
                            exc_info=True,
                        )
                        trim_memory()
                        return None
        return self._model

    @property
    def model(self):
        """Underlying model instance (lazily loaded)."""
        return self._get_model()

    @model.setter
    def model(self, value):
        """Allows injecting a mock model for testing or overriding."""
        self._model = value
        self._load_failed = False
        if value is not None:
            self._enabled = True
            with _SHARED_RERANKERS_LOCK:
                _SHARED_RERANKERS[self.model_name] = value

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResultItem],
        top_k: int = 5,
    ) -> Tuple[List[RetrievalResultItem], float]:
        """Scores (query, passage) pairs and returns top_k candidates sorted by relevance.

        Falls back gracefully to fused rank ordering if reranking is disabled, model is unavailable,
        or memory limits are encountered.
        """
        if not candidates or not query.strip():
            return [], 0.0

        start_time = time.perf_counter()

        # If CrossEncoder is disabled in configuration, use RRF candidate ordering immediately
        # without loading model weights into memory
        if not getattr(self, "_enabled", True):
            logger.debug("CrossEncoder reranking disabled by configuration. Preserving fused rank order.")
            fallback = candidates[:top_k]
            for idx, item in enumerate(fallback, start=1):
                item.final_rank = idx
            return fallback, (time.perf_counter() - start_time) * 1000.0

        model = self._get_model()

        # Graceful fallback if model could not be loaded due to memory or runtime error
        if model is None:
            logger.warning("CrossEncoder model unavailable. Falling back to fused candidate order.")
            fallback = candidates[:top_k]
            for idx, item in enumerate(fallback, start=1):
                item.final_rank = idx
            return fallback, (time.perf_counter() - start_time) * 1000.0

        pairs = [[query, item.text] for item in candidates]

        try:
            import torch
            with torch.inference_mode():
                scores = model.predict(pairs, show_progress_bar=False, batch_size=min(len(pairs), 8))
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract equipment identifiers from query to prioritize exact equipment matches
            import re
            query_machines = set(re.findall(r"\b([A-Z]{1,4}-[0-9]{2,4}[A-Z]?)\b", query))

            # Attach scores to candidate copies
            scored_candidates = []
            for item, score in zip(candidates, scores):
                item_copy = item.model_copy()
                base_score = float(score)
                boost = 0.0
                if query_machines:
                    item_machines = set(re.findall(r"\b([A-Z]{1,4}-[0-9]{2,4}[A-Z]?)\b", item.text))
                    if item.metadata.machine_id:
                        item_machines.add(item.metadata.machine_id)
                    if query_machines.intersection(item_machines):
                        boost += 2.0  # Prioritize exact equipment match over other machines
                item_copy.rerank_score = base_score + boost
                scored_candidates.append(item_copy)

            # Sort descending by cross-encoder score
            scored_candidates.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -999.0, reverse=True)

            # Slice to top_k and update final_rank
            reranked_results = scored_candidates[:top_k]
            for rank_idx, item in enumerate(reranked_results, start=1):
                item.final_rank = rank_idx

            logger.info(
                f"Reranked {len(candidates)} candidates down to {len(reranked_results)} in {latency_ms:.1f}ms "
                f"(RSS: {get_process_memory_mb():.1f} MB)"
            )
            trim_memory()
            return reranked_results, latency_ms

        except Exception as e:
            logger.error(f"Error during cross-encoder reranking: {e}. Preserving fused rank order.")
            trim_memory()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback = candidates[:top_k]
            for idx, item in enumerate(fallback, start=1):
                item.final_rank = idx
            return fallback, latency_ms


def get_cross_encoder_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """Returns singleton instance of CrossEncoderReranker."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None or (model_name and _RERANKER_INSTANCE.model_name != model_name):
        with _RERANKER_LOCK:
            if _RERANKER_INSTANCE is None or (model_name and _RERANKER_INSTANCE.model_name != model_name):
                _RERANKER_INSTANCE = CrossEncoderReranker(model_name=model_name)
    return _RERANKER_INSTANCE
