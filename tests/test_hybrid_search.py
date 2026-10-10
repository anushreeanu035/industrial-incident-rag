"""Tests for Reciprocal Rank Fusion (RRF) and Hybrid Search integration."""

from incident_rag.retrieval.fusion import reciprocal_rank_fusion
from incident_rag.schemas import ChunkMetadata, RetrievalResultItem


def make_item(chunk_id: str, doc_id: str, text: str, dense_score=None, bm25_score=None):
    return RetrievalResultItem(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        metadata=ChunkMetadata(
            chunk_id=chunk_id,
            document_id=doc_id,
            chunk_index=0,
            source_filename="test.md",
        ),
        dense_score=dense_score,
        bm25_score=bm25_score,
        final_rank=1,
    )


def test_rrf_mathematical_fusion():
    # Dense results: A is rank 1, B is rank 2
    dense_results = [
        make_item("A", "d1", "Chunk A", dense_score=0.9),
        make_item("B", "d2", "Chunk B", dense_score=0.8),
    ]

    # BM25 results: B is rank 1, C is rank 2
    bm25_results = [
        make_item("B", "d2", "Chunk B", bm25_score=4.5),
        make_item("C", "d3", "Chunk C", bm25_score=3.2),
    ]

    # RRF with k = 60:
    # A score = 1 / (60 + 1) = 1/61 = ~0.016393
    # B score = 1 / (60 + 2) + 1 / (60 + 1) = 1/62 + 1/61 = 0.016129 + 0.016393 = ~0.032522
    # C score = 1 / (60 + 2) = 1/62 = ~0.016129
    # B should be rank 1!

    fused = reciprocal_rank_fusion(dense_results, bm25_results, k=60)
    assert len(fused) == 3

    assert fused[0].chunk_id == "B"
    assert fused[0].final_rank == 1
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert abs(fused[0].rrf_score - (1/61 + 1/62)) < 1e-6

    assert fused[1].chunk_id == "A"
    assert fused[1].dense_rank == 1
    assert fused[1].bm25_rank is None

    assert fused[2].chunk_id == "C"
    assert fused[2].dense_rank is None
    assert fused[2].bm25_rank == 2
