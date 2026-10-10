"""Reproducible benchmark execution engine evaluating baseline and advanced RAG pipelines."""

from datetime import datetime
import time
from typing import Dict, List, Optional
import uuid
import numpy as np
from incident_rag.config import get_settings
from incident_rag.evaluation.dataset import load_benchmark_dataset
from incident_rag.evaluation.ragas_evaluator import RagasEvaluator
from incident_rag.evaluation.retrieval_metrics import evaluate_retrieval_for_query
from incident_rag.logging_config import get_logger
from incident_rag.pipeline import IncidentRAGPipeline
from incident_rag.schemas import (
    BenchmarkReport,
    InvestigationRequest,
    RagasMetricScores,
    RetrievalMetricScores,
    RetrievalRequest,
    RetrievalStrategy,
    StrategyBenchmarkResult,
)

logger = get_logger(__name__)


class BenchmarkRunner:
    """Executes deterministic retrieval benchmarks and generation evaluations across strategies."""

    def __init__(self, pipeline: IncidentRAGPipeline, ragas_evaluator: Optional[RagasEvaluator] = None):
        self.pipeline = pipeline
        self.settings = get_settings()
        self.ragas_evaluator = ragas_evaluator or RagasEvaluator()

    def run_benchmark(
        self,
        strategies: Optional[List[RetrievalStrategy]] = None,
        top_k: int = 5,
        run_ragas: bool = False,
    ) -> BenchmarkReport:
        """Executes full benchmark suite over the ground truth dataset."""
        dataset = load_benchmark_dataset()
        test_cases = dataset.test_cases
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.utcnow().isoformat()

        if strategies is None:
            strategies = [
                RetrievalStrategy.DENSE,
                RetrievalStrategy.BM25,
                RetrievalStrategy.HYBRID_RRF,
                RetrievalStrategy.HYBRID_RERANK,
                RetrievalStrategy.FULL_ADVANCED,
            ]

        logger.info(f"Starting benchmark {run_id} across {len(strategies)} strategies on {len(test_cases)} queries.")

        strategy_results: List[StrategyBenchmarkResult] = []

        for strat in strategies:
            strat_name = strat.value if hasattr(strat, "value") else str(strat)
            logger.info(f"Evaluating strategy: {strat_name}")

            latencies_ms: List[float] = []
            recall_5_list: List[float] = []
            recall_10_list: List[float] = []
            precision_k_list: List[float] = []
            mrr_list: List[float] = []
            ndcg_10_list: List[float] = []

            questions_for_ragas: List[str] = []
            answers_for_ragas: List[str] = []
            contexts_for_ragas: List[List[str]] = []
            ground_truths_for_ragas: List[str] = []

            successful_count = 0
            failed_count = 0

            for tc in test_cases:
                query = tc["query"]
                target_filename = tc["target_filename"]
                ground_truth_passages = tc.get("ground_truth_passages", [])
                machine_id = tc.get("machine_id")

                t0 = time.perf_counter()
                try:
                    req = RetrievalRequest(
                        query=query,
                        strategy=strat,
                        top_k=top_k,
                        machine_id_filter=None,  # Unfiltered benchmark evaluation
                    )
                    retrieval_res = self.pipeline.retrieve(req)
                    latency = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(latency)

                    # Compute retrieval ranking metrics
                    metrics = evaluate_retrieval_for_query(
                        results=retrieval_res.results,
                        target_filename=target_filename,
                        ground_truth_passages=ground_truth_passages,
                        k=top_k,
                    )

                    recall_5_list.append(metrics["recall_at_5"])
                    recall_10_list.append(metrics["recall_at_10"])
                    precision_k_list.append(metrics["precision_at_k"])
                    mrr_list.append(metrics["mrr"])
                    ndcg_10_list.append(metrics["ndcg_at_10"])

                    # If generation/ragas requested for advanced strategy
                    if run_ragas and strat == RetrievalStrategy.FULL_ADVANCED:
                        inv_req = InvestigationRequest(
                            query=query,
                            strategy=strat,
                            top_k=top_k,
                        )
                        inv_resp = self.pipeline.investigate(inv_req)
                        questions_for_ragas.append(query)
                        answers_for_ragas.append(inv_resp.executive_summary)
                        contexts_for_ragas.append([r.text for r in retrieval_res.results])
                        ground_truths_for_ragas.append(tc.get("reference_answer", ""))

                    successful_count += 1
                except Exception as e:
                    logger.error(f"Error evaluating query '{query}' under {strat_name}: {e}")
                    failed_count += 1
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    recall_5_list.append(0.0)
                    recall_10_list.append(0.0)
                    precision_k_list.append(0.0)
                    mrr_list.append(0.0)
                    ndcg_10_list.append(0.0)

            # Calculate aggregated metrics
            avg_lat = float(np.mean(latencies_ms)) if latencies_ms else 0.0
            p95_lat = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0

            retrieval_scores = RetrievalMetricScores(
                recall_at_5=float(np.mean(recall_5_list)) if recall_5_list else 0.0,
                recall_at_10=float(np.mean(recall_10_list)) if recall_10_list else 0.0,
                precision_at_k=float(np.mean(precision_k_list)) if precision_k_list else 0.0,
                mrr=float(np.mean(mrr_list)) if mrr_list else 0.0,
                ndcg_at_10=float(np.mean(ndcg_10_list)) if ndcg_10_list else 0.0,
            )

            # Ragas evaluation if applicable
            ragas_scores: Optional[RagasMetricScores] = None
            if run_ragas and questions_for_ragas:
                ragas_scores = self.ragas_evaluator.evaluate(
                    questions=questions_for_ragas,
                    answers=answers_for_ragas,
                    contexts=contexts_for_ragas,
                    ground_truths=ground_truths_for_ragas,
                )

            strategy_results.append(
                StrategyBenchmarkResult(
                    strategy=strat_name,
                    retrieval_metrics=retrieval_scores,
                    ragas_metrics=ragas_scores,
                    avg_latency_ms=round(avg_lat, 2),
                    p95_latency_ms=round(p95_lat, 2),
                    total_queries=len(test_cases),
                    successful_queries=successful_count,
                    failed_queries=failed_count,
                )
            )

        report = BenchmarkReport(
            run_id=run_id,
            timestamp=timestamp,
            dataset_version=dataset.version,
            dataset_hash=dataset.dataset_hash,
            embedding_model=self.settings.EMBEDDING_MODEL,
            reranker_model=self.settings.RERANKER_MODEL,
            llm_model=self.settings.GROQ_MODEL,
            configuration={
                "top_k": top_k,
                "dense_top_k": self.settings.DENSE_TOP_K,
                "bm25_top_k": self.settings.BM25_TOP_K,
                "rrf_k": self.settings.RRF_K,
                "chunk_size": self.settings.CHUNK_SIZE,
                "chunk_overlap": self.settings.CHUNK_OVERLAP,
                "query_rewriting_enabled": self.settings.QUERY_REWRITING_ENABLED,
            },
            results=strategy_results,
        )

        return report
