"""Tests for benchmark runner, report export, and evaluation metrics."""

from pathlib import Path
from unittest.mock import MagicMock
from incident_rag.evaluation.dataset import load_benchmark_dataset
from incident_rag.evaluation.reports import (
    get_benchmark_report,
    list_benchmark_reports,
    save_benchmark_report,
)
from incident_rag.evaluation.benchmark_runner import BenchmarkRunner
from incident_rag.schemas import (
    BenchmarkReport,
    ChunkMetadata,
    RetrievalMetricScores,
    RetrievalResponse,
    RetrievalResultItem,
    RetrievalStrategy,
    StrategyBenchmarkResult,
)


def test_load_benchmark_dataset():
    ds = load_benchmark_dataset()
    assert len(ds.test_cases) == 10
    assert len(ds.dataset_hash) == 64
    assert ds.version == "1.0.0"


def test_save_and_load_benchmark_report(tmp_path: Path):
    res = StrategyBenchmarkResult(
        strategy="hybrid_rrf",
        retrieval_metrics=RetrievalMetricScores(
            recall_at_5=0.8,
            recall_at_10=0.9,
            precision_at_k=0.7,
            mrr=0.85,
            ndcg_at_10=0.88,
        ),
        avg_latency_ms=45.2,
        p95_latency_ms=62.1,
        total_queries=10,
        successful_queries=10,
        failed_queries=0,
    )
    report = BenchmarkReport(
        run_id="test_run_123",
        timestamp="2026-10-09T10:00:00",
        dataset_version="1.0.0",
        dataset_hash="abc123hash",
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        llm_model="llama-3.3-70b-versatile",
        configuration={"top_k": 5},
        results=[res],
    )

    json_path, csv_path = save_benchmark_report(report, reports_dir=tmp_path)
    assert json_path.exists()
    assert csv_path.exists()

    loaded = get_benchmark_report("test_run_123", reports_dir=tmp_path)
    assert loaded is not None
    assert loaded.run_id == "test_run_123"
    assert loaded.results[0].strategy == "hybrid_rrf"

    all_reports = list_benchmark_reports(reports_dir=tmp_path)
    assert len(all_reports) == 1


def test_benchmark_runner_mock_pipeline():
    mock_pipeline = MagicMock()

    # Create dummy item matching target
    dummy_item = RetrievalResultItem(
        chunk_id="chk1",
        document_id="doc1",
        text="High-pressure steam turbine HP-17 bearing B-101 failure and V-204 orifice clog.",
        metadata=ChunkMetadata(
            chunk_id="chk1",
            document_id="doc1",
            chunk_index=0,
            machine_id="HP-17",
            source_filename="INC-2024-089_turbine_bearing_failure.md",
        ),
        final_rank=1,
    )

    mock_resp = RetrievalResponse(
        query="test query",
        strategy="bm25",
        results=[dummy_item],
        execution_time_ms=15.0,
    )
    mock_pipeline.retrieve.return_value = mock_resp

    runner = BenchmarkRunner(pipeline=mock_pipeline)
    report = runner.run_benchmark(strategies=[RetrievalStrategy.BM25], top_k=5, run_ragas=False)

    assert report.run_id.startswith("run_")
    assert len(report.results) == 1
    assert report.results[0].strategy == "bm25"
    assert report.results[0].total_queries == 10
    assert report.results[0].successful_queries == 10
