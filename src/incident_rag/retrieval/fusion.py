"""Reciprocal Rank Fusion (RRF) algorithms for combining multi-modal retrieval lists."""

from typing import Dict, List, Optional
from incident_rag.schemas import RetrievalResultItem


def reciprocal_rank_fusion(
    dense_results: List[RetrievalResultItem],
    bm25_results: List[RetrievalResultItem],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[RetrievalResultItem]:
    """Combines dense and BM25 rank lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum( 1 / (k + rank_i(d)) )
    Preserves individual dense_score, dense_rank, bm25_score, bm25_rank.
    """
    fused_candidates: Dict[str, Dict] = {}

    # Process dense results
    for dense_rank, item in enumerate(dense_results, start=1):
        cid = item.chunk_id
        rrf_val = 1.0 / (k + dense_rank)

        if cid not in fused_candidates:
            fused_candidates[cid] = {
                "item": item.model_copy(),
                "rrf_score": rrf_val,
                "dense_score": item.dense_score,
                "dense_rank": dense_rank,
                "bm25_score": None,
                "bm25_rank": None,
            }
        else:
            fused_candidates[cid]["rrf_score"] += rrf_val
            fused_candidates[cid]["dense_score"] = item.dense_score
            fused_candidates[cid]["dense_rank"] = dense_rank

    # Process BM25 results
    for bm25_rank, item in enumerate(bm25_results, start=1):
        cid = item.chunk_id
        rrf_val = 1.0 / (k + bm25_rank)

        if cid not in fused_candidates:
            fused_candidates[cid] = {
                "item": item.model_copy(),
                "rrf_score": rrf_val,
                "dense_score": None,
                "dense_rank": None,
                "bm25_score": item.bm25_score,
                "bm25_rank": bm25_rank,
            }
        else:
            fused_candidates[cid]["rrf_score"] += rrf_val
            fused_candidates[cid]["bm25_score"] = item.bm25_score
            fused_candidates[cid]["bm25_rank"] = bm25_rank

    # Sort descending by fused RRF score
    sorted_candidates = sorted(
        fused_candidates.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    if top_k is not None:
        sorted_candidates = sorted_candidates[:top_k]

    results: List[RetrievalResultItem] = []
    for rank_idx, cand in enumerate(sorted_candidates, start=1):
        base_item = cand["item"]
        base_item.dense_score = cand["dense_score"]
        base_item.dense_rank = cand["dense_rank"]
        base_item.bm25_score = cand["bm25_score"]
        base_item.bm25_rank = cand["bm25_rank"]
        base_item.rrf_score = cand["rrf_score"]
        base_item.rrf_rank = rank_idx
        base_item.final_rank = rank_idx
        results.append(base_item)

    return results
