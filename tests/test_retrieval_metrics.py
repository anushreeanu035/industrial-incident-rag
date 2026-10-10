"""Tests for deterministic retrieval ranking metrics."""

import math
import pytest
from incident_rag.evaluation.retrieval_metrics import (
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
)


def test_compute_recall_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "D", "X"}  # 3 relevant items, B and D are in top 5

    assert compute_recall_at_k(retrieved, relevant, k=3) == 1 / 3  # Only "B" in top 3
    assert compute_recall_at_k(retrieved, relevant, k=5) == 2 / 3  # "B" and "D" in top 5


def test_compute_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = {"B", "D"}

    assert compute_precision_at_k(retrieved, relevant, k=2) == 1 / 2  # "B" in top 2
    assert compute_precision_at_k(retrieved, relevant, k=4) == 2 / 4  # "B" and "D" in top 4
    assert compute_precision_at_k(retrieved, relevant, k=5) == 2 / 5


def test_compute_reciprocal_rank():
    assert compute_reciprocal_rank(["A", "B", "C"], {"A"}) == 1.0
    assert compute_reciprocal_rank(["A", "B", "C"], {"B"}) == 0.5
    assert compute_reciprocal_rank(["A", "B", "C"], {"C"}) == 1 / 3
    assert compute_reciprocal_rank(["A", "B", "C"], {"Z"}) == 0.0


def test_compute_ndcg_at_k():
    # If first item is relevant: DCG = 1 / log2(2) = 1.0, IDCG = 1.0 -> nDCG = 1.0
    assert compute_ndcg_at_k(["A", "B"], {"A"}, k=2) == 1.0

    # If second item is relevant: DCG = 1 / log2(3) = 0.6309..., IDCG = 1.0 -> nDCG = 0.6309...
    expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
    assert abs(compute_ndcg_at_k(["A", "B"], {"B"}, k=2) - expected_ndcg) < 1e-5

    # If no relevant items:
    assert compute_ndcg_at_k(["A", "B"], {"Z"}, k=2) == 0.0
