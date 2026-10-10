"""Ragas evaluation integration with provider validation and failure handling."""

from typing import Dict, List, Optional
from incident_rag.config import get_settings
from incident_rag.logging_config import get_logger
from incident_rag.schemas import RagasMetricScores

logger = get_logger(__name__)


class RagasEvaluator:
    """Evaluates RAG generation and context quality using Ragas metrics."""

    def __init__(self):
        self.settings = get_settings()

    def evaluate(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
    ) -> Optional[RagasMetricScores]:
        """Runs Ragas evaluation if supported environment/credentials are configured.

        Never invents scores; returns None if external evaluation dependencies are unconfigured.
        """
        # Ragas requires an OpenAI / LLM key to run LLM-as-a-judge metrics
        # If GROQ_API_KEY is configured with OpenAI-compatible wrapper or OPENAI_API_KEY is present
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "") or self.settings.GROQ_API_KEY

        if not api_key:
            logger.info("No LLM judge API key available for Ragas evaluation. Skipping.")
            return None

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            data_dict = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            metrics = [faithfulness, answer_relevancy, context_precision]

            if ground_truths and len(ground_truths) == len(questions):
                data_dict["ground_truth"] = ground_truths
                metrics.append(context_recall)

            dataset = Dataset.from_dict(data_dict)
            results = evaluate(dataset=dataset, metrics=metrics)

            return RagasMetricScores(
                faithfulness=float(results.get("faithfulness", 0.0)),
                answer_relevancy=float(results.get("answer_relevancy", 0.0)),
                context_precision=float(results.get("context_precision", 0.0)),
                context_recall=float(results.get("context_recall", 0.0)) if "context_recall" in results else None,
            )
        except Exception as e:
            logger.warning(f"Ragas evaluation could not be completed: {e}")
            return None
