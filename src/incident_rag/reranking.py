"""Cross-encoder reranking module using Sentence Transformers CrossEncoder."""

import time
from typing import List, Optional, Tuple
from sentence_transformers import CrossEncoder
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import RetrievalResultItem

logger = get_logger(__name__)

_RERANKER_INSTANCE: Optional["CrossEncoderReranker"] = None


class CrossEncoderReranker:
    """Reranks retrieved candidate chunks using a trained cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None):
        settings = get_settings()
        self.model_name = model_name or settings.RERANKER_MODEL
        logger.info(f"Loading CrossEncoder reranker model: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info("CrossEncoder model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResultItem],
        top_k: int = 5,
    ) -> Tuple[List[RetrievalResultItem], float]:
        """Scores (query, passage) pairs and returns top_k candidates sorted by relevance.

        Returns:
            Tuple of (reranked_items, rerank_latency_ms)
        """
        if not candidates or not query.strip():
            return [], 0.0

        start_time = time.perf_counter()
        pairs = [[query, item.text] for item in candidates]

        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach scores to candidate copies
            scored_candidates = []
            for item, score in zip(candidates, scores):
                item_copy = item.model_copy()
                item_copy.rerank_score = float(score)
                scored_candidates.append(item_copy)

            # Sort descending by cross-encoder score
            scored_candidates.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else -999.0, reverse=True)

            # Slice to top_k and update final_rank
            reranked_results = scored_candidates[:top_k]
            for rank_idx, item in enumerate(reranked_results, start=1):
                item.final_rank = rank_idx

            logger.info(
                f"Reranked {len(candidates)} candidates down to {len(reranked_results)} in {latency_ms:.1f}ms"
            )
            return reranked_results, latency_ms

        except Exception as e:
            logger.error(f"Error during cross-encoder reranking: {e}. Preserving fused rank order.")
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            fallback = candidates[:top_k]
            for idx, item in enumerate(fallback, start=1):
                item.final_rank = idx
            return fallback, latency_ms


def get_cross_encoder_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """Returns singleton instance of CrossEncoderReranker."""
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None or (model_name and _RERANKER_INSTANCE.model_name != model_name):
        _RERANKER_INSTANCE = CrossEncoderReranker(model_name=model_name)
    return _RERANKER_INSTANCE
