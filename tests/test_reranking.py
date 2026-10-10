"""Tests for cross-encoder reranking order and metadata preservation."""

from unittest.mock import MagicMock
import numpy as np
from incident_rag.reranking import CrossEncoderReranker
from incident_rag.schemas import ChunkMetadata, RetrievalResultItem


def make_candidate(chunk_id: str, text: str, initial_rank: int):
    return RetrievalResultItem(
        chunk_id=chunk_id,
        document_id="doc1",
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id="doc1",
            chunk_index=initial_rank,
            source_filename="test.md",
        ),
        dense_score=0.7,
        bm25_score=3.0,
        rrf_score=0.03,
        final_rank=initial_rank,
    )


def test_reranker_reordering():
    # Mock underlying sentence_transformers CrossEncoder model so test is fast and deterministic
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "test-model"
    mock_model = MagicMock()
    # Return high score for candidate 2 and lower score for candidate 1
    mock_model.predict.return_value = np.array([0.15, 0.95])
    reranker.model = mock_model

    c1 = make_candidate("c1", "Low relevance chunk", 1)
    c2 = make_candidate("c2", "High relevance chunk", 2)

    reranked, latency = reranker.rerank("query", [c1, c2], top_k=2)

    # c2 should now be ranked #1
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].final_rank == 1
    assert reranked[0].rerank_score == 0.95

    # Original metadata preserved
    assert reranked[0].dense_score == 0.7
    assert reranked[0].bm25_score == 3.0
    assert reranked[0].rrf_score == 0.03

    assert reranked[1].chunk_id == "c1"
    assert reranked[1].final_rank == 2
    assert reranked[1].rerank_score == 0.15
    assert latency >= 0.0


def test_reranker_error_fallback():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "test-model"
    mock_model = MagicMock()
    mock_model.predict.side_effect = RuntimeError("GPU out of memory")
    reranker.model = mock_model

    c1 = make_candidate("c1", "Chunk 1", 1)
    c2 = make_candidate("c2", "Chunk 2", 2)

    # Should not raise exception; falls back to original order
    reranked, latency = reranker.rerank("query", [c1, c2], top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c1"
    assert reranked[1].chunk_id == "c2"
