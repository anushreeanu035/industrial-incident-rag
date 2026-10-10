"""Deterministic retrieval ranking metrics: Recall@K, Precision@K, MRR, nDCG@K."""

import math
from typing import Any, Dict, List, Sequence, Set
from incident_rag.schemas import RetrievalMetricScores, RetrievalResultItem


def compute_recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Recall@K: fraction of relevant items retrieved in top K."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k).intersection(relevant_ids))
    return float(hits / len(relevant_ids))


def compute_precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Precision@K: fraction of top K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = len(set(top_k).intersection(relevant_ids))
    return float(hits / k)


def compute_reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Set[str]) -> float:
    """Calculates Reciprocal Rank (1/rank) for the first relevant retrieved item."""
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain at K with binary relevance."""
    if not relevant_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, item_id in enumerate(retrieved_ids[:k], start=1):
        if item_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0

    return float(dcg / idcg)


def is_chunk_relevant(item: RetrievalResultItem, target_filename: str, ground_truth_passages: List[str]) -> bool:
    """Evaluates whether a retrieved chunk is relevant to the benchmark target."""
    # Match on document source filename
    if item.metadata.source_filename == target_filename:
        return True

    # Match on key ground truth passage substrings
    item_text_lower = item.text.lower()
    for passage in ground_truth_passages:
        if passage.lower() in item_text_lower:
            return True

    return False


def evaluate_retrieval_for_query(
    results: List[RetrievalResultItem],
    target_filename: str,
    ground_truth_passages: List[str],
    k: int = 5,
) -> Dict[str, float]:
    """Computes all retrieval metrics for a single query."""
    # Create relevance set: chunk_id is relevant if is_chunk_relevant is True
    relevant_chunk_ids: Set[str] = set()
    retrieved_chunk_ids: List[str] = [item.chunk_id for item in results]

    for item in results:
        if is_chunk_relevant(item, target_filename, ground_truth_passages):
            relevant_chunk_ids.add(item.chunk_id)

    # If the target file exists in corpus, there must be at least 1 relevant chunk target
    # If no retrieved chunk matches, relevant_chunk_ids may be empty, in which case recall/mrr is 0.0
    if not relevant_chunk_ids:
        # Dummy target to reflect zero recall
        relevant_chunk_ids = {"__UNRETRIEVED_RELEVANT__"}

    recall_5 = compute_recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=5)
    recall_10 = compute_recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=10)
    precision_k = compute_precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=k)
    rr = compute_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids)
    ndcg_10 = compute_ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k=10)

    return {
        "recall_at_5": recall_5,
        "recall_at_10": recall_10,
        "precision_at_k": precision_k,
        "mrr": rr,
        "ndcg_at_10": ndcg_10,
    }
