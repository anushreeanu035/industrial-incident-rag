"""Benchmark report generation, CSV export, and persistence."""

import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import BenchmarkReport

logger = get_logger(__name__)


def save_benchmark_report(
    report: BenchmarkReport,
    reports_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Saves benchmark report to structured JSON and CSV files."""
    settings = get_settings()
    target_dir = reports_dir or settings.resolve_path(settings.REPORTS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"benchmark_{report.run_id}.json"
    csv_path = target_dir / f"benchmark_{report.run_id}.csv"

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)

    # Save CSV
    csv_fields = [
        "run_id",
        "timestamp",
        "strategy",
        "recall_at_5",
        "recall_at_10",
        "precision_at_k",
        "mrr",
        "ndcg_at_10",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "avg_latency_ms",
        "p95_latency_ms",
        "successful_queries",
        "failed_queries",
        "embedding_model",
        "reranker_model",
        "llm_model",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        for res in report.results:
            row = {
                "run_id": report.run_id,
                "timestamp": report.timestamp,
                "strategy": res.strategy,
                "recall_at_5": round(res.retrieval_metrics.recall_at_5, 4),
                "recall_at_10": round(res.retrieval_metrics.recall_at_10, 4),
                "precision_at_k": round(res.retrieval_metrics.precision_at_k, 4),
                "mrr": round(res.retrieval_metrics.mrr, 4),
                "ndcg_at_10": round(res.retrieval_metrics.ndcg_at_10, 4),
                "faithfulness": res.ragas_metrics.faithfulness if res.ragas_metrics else "",
                "answer_relevancy": res.ragas_metrics.answer_relevancy if res.ragas_metrics else "",
                "context_precision": res.ragas_metrics.context_precision if res.ragas_metrics else "",
                "context_recall": res.ragas_metrics.context_recall if res.ragas_metrics else "",
                "avg_latency_ms": res.avg_latency_ms,
                "p95_latency_ms": res.p95_latency_ms,
                "successful_queries": res.successful_queries,
                "failed_queries": res.failed_queries,
                "embedding_model": report.embedding_model,
                "reranker_model": report.reranker_model,
                "llm_model": report.llm_model,
            }
            writer.writerow(row)

    logger.info(f"Saved benchmark report to {json_path} and {csv_path}")
    return json_path, csv_path


def list_benchmark_reports(reports_dir: Optional[Path] = None) -> List[BenchmarkReport]:
    """Retrieves all benchmark reports stored in the reports directory."""
    settings = get_settings()
    target_dir = reports_dir or settings.resolve_path(settings.REPORTS_DIR)
    if not target_dir.exists():
        return []

    reports = []
    for json_file in sorted(target_dir.glob("benchmark_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                reports.append(BenchmarkReport.model_validate(data))
        except Exception as e:
            logger.warning(f"Failed to read benchmark report {json_file}: {e}")

    return reports


def get_benchmark_report(run_id: str, reports_dir: Optional[Path] = None) -> Optional[BenchmarkReport]:
    """Retrieves a single benchmark report by run_id."""
    settings = get_settings()
    target_dir = reports_dir or settings.resolve_path(settings.REPORTS_DIR)
    json_path = target_dir / f"benchmark_{run_id}.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return BenchmarkReport.model_validate(json.load(f))
        except Exception as e:
            logger.error(f"Failed reading report {json_path}: {e}")
    return None
