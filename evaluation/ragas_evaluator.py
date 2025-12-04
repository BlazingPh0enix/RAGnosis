"""
RAGAS evaluation for DocuLens RAG system.

This module provides evaluation metrics using the RAGAS framework:
- Faithfulness: Is the answer grounded in the context?
- Answer Relevancy: Does the answer address the query?
- Context Relevancy: Is the retrieved context relevant?
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

from config.settings import settings
from config.logging_config import get_logger, log_execution_time, LogTimer

logger = get_logger(__name__)


# Default metrics to evaluate
DEFAULT_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
]


@dataclass
class EvaluationSample:
    """
    A single evaluation sample.
    
    Attributes:
        question: The query/question asked.
        answer: The generated answer.
        contexts: List of retrieved context strings.
        ground_truth: Optional ground truth answer for reference metrics.
    """
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for RAGAS Dataset."""
        result = {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
        }
        if self.ground_truth:
            result["ground_truth"] = self.ground_truth
        return result


@dataclass
class EvaluationResult:
    """
    Results from a RAGAS evaluation run.
    
    Attributes:
        samples: List of evaluated samples.
        metrics: Dictionary of metric name to score.
        per_sample_scores: List of per-sample metric scores.
        timestamp: When the evaluation was run.
        metadata: Additional metadata about the run.
    """
    samples: List[EvaluationSample]
    metrics: Dict[str, float]
    per_sample_scores: List[Dict[str, float]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "num_samples": len(self.samples),
            "metrics": self.metrics,
            "per_sample_scores": self.per_sample_scores,
            "metadata": self.metadata,
        }
    
    def save(self, path: Path):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @property
    def faithfulness_score(self) -> float:
        """Get faithfulness metric score."""
        return self.metrics.get("faithfulness", 0.0)
    
    @property
    def answer_relevancy_score(self) -> float:
        """Get answer relevancy metric score."""
        return self.metrics.get("answer_relevancy", 0.0)
    
    @property
    def context_precision_score(self) -> float:
        """Get context precision metric score."""
        return self.metrics.get("context_precision", 0.0)
    
    def summary(self) -> str:
        """Get a human-readable summary of results."""
        lines = [
            "=" * 50,
            "RAGAS Evaluation Results",
            "=" * 50,
            f"Timestamp: {self.timestamp}",
            f"Samples Evaluated: {len(self.samples)}",
            "",
            "Aggregate Metrics:",
        ]
        
        for name, score in self.metrics.items():
            lines.append(f"  {name}: {score:.4f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class RagasEvaluator:
    """
    Evaluator using RAGAS metrics for RAG system quality.
    
    This class wraps the RAGAS evaluation framework to measure:
    - Faithfulness: Whether answers are grounded in context
    - Answer Relevancy: Whether answers address the question
    - Context Precision: Quality of retrieved context
    """
    
    def __init__(
        self,
        metrics: Optional[List] = None,
        llm_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            metrics: List of RAGAS metrics to use. Defaults to standard set.
            llm_model: LLM model for evaluation (used by RAGAS internally).
        """
        self.metrics = metrics or DEFAULT_METRICS
        self.llm_model = llm_model
        self._results_history: List[EvaluationResult] = []
        
        metric_names = [m.name for m in self.metrics]
        logger.info(f"Initializing RagasEvaluator with metrics: {metric_names}")
        logger.debug(f"Using LLM model: {llm_model}")
    
    def create_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationSample:
        """
        Create an evaluation sample.
        
        Args:
            question: The query asked.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            ground_truth: Optional ground truth answer.
            
        Returns:
            EvaluationSample ready for evaluation.
        """
        return EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    
    def create_sample_from_query_result(
        self,
        query_result,
        ground_truth: Optional[str] = None,
    ) -> EvaluationSample:
        """
        Create an evaluation sample from a QueryResult.
        
        Args:
            query_result: QueryResult from the query engine.
            ground_truth: Optional ground truth answer.
            
        Returns:
            EvaluationSample ready for evaluation.
        """
        # Extract contexts from source nodes
        contexts = []
        for source in query_result.sources:
            text = source.get("text_preview", "")
            if text:
                contexts.append(text)
        
        return EvaluationSample(
            question=query_result.query,
            answer=query_result.response,
            contexts=contexts,
            ground_truth=ground_truth,
        )
    
    @log_execution_time()
    def evaluate(
        self,
        samples: List[EvaluationSample],
        raise_exceptions: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate a list of samples using RAGAS metrics.
        
        Args:
            samples: List of EvaluationSample objects.
            raise_exceptions: Whether to raise exceptions on errors.
            
        Returns:
            EvaluationResult with metrics and per-sample scores.
        """
        if not samples:
            logger.warning("No samples provided for evaluation")
            return EvaluationResult(
                samples=[],
                metrics={},
                metadata={"error": "No samples provided"},
            )
        
        logger.info(f"Starting RAGAS evaluation with {len(samples)} samples")
        
        # Convert samples to RAGAS Dataset format
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        
        # Add ground truth if available
        if any(s.ground_truth for s in samples):
            data["ground_truth"] = [s.ground_truth or "" for s in samples]
        
        dataset = Dataset.from_dict(data)
        
        # Run RAGAS evaluation
        try:
            logger.debug("Running RAGAS evaluation...")
            with LogTimer(logger, "RAGAS evaluation"):
                result = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    raise_exceptions=raise_exceptions,
                )
            
            # Extract aggregate metrics
            metrics = {}
            for metric in self.metrics:
                metric_name = metric.name
                if metric_name in result:
                    # RAGAS result values can be lists or scalars
                    value = result[metric_name]
                    if isinstance(value, list):
                        # Take mean of list values
                        metrics[metric_name] = sum(value) / len(value) if value else 0.0
                    else:
                        metrics[metric_name] = float(value)
            
            # Extract per-sample scores from the result dataframe
            per_sample_scores = []
            if hasattr(result, 'to_pandas'):
                df = result.to_pandas()
                for _, row in df.iterrows():
                    sample_scores = {}
                    for metric in self.metrics:
                        if metric.name in row:
                            sample_scores[metric.name] = float(row[metric.name])
                    per_sample_scores.append(sample_scores)
            
            eval_result = EvaluationResult(
                samples=samples,
                metrics=metrics,
                per_sample_scores=per_sample_scores,
                metadata={
                    "llm_model": self.llm_model,
                    "metrics_used": [m.name for m in self.metrics],
                },
            )
            
            self._results_history.append(eval_result)
            logger.info(f"RAGAS evaluation completed. Metrics: {metrics}")
            return eval_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            if raise_exceptions:
                raise
            return EvaluationResult(
                samples=samples,
                metrics={},
                metadata={"error": str(e)},
            )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: The query asked.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            ground_truth: Optional ground truth answer.
            
        Returns:
            EvaluationResult for the single sample.
        """
        sample = self.create_sample(question, answer, contexts, ground_truth)
        return self.evaluate([sample])
    
    def get_results_history(self) -> List[EvaluationResult]:
        """Get history of all evaluation results."""
        return self._results_history
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Calculate average metrics across all historical results.
        
        Returns:
            Dictionary of metric name to average score.
        """
        if not self._results_history:
            return {}
        
        metric_sums: Dict[str, float] = {}
        metric_counts: Dict[str, int] = {}
        
        for result in self._results_history:
            for name, score in result.metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + score
                metric_counts[name] = metric_counts.get(name, 0) + 1
        
        return {
            name: metric_sums[name] / metric_counts[name]
            for name in metric_sums
        }


def create_evaluator(
    metrics: Optional[List] = None,
    llm_model: str = "gpt-4o-mini",
) -> RagasEvaluator:
    """
    Factory function to create a RagasEvaluator.
    
    Args:
        metrics: List of RAGAS metrics to use.
        llm_model: LLM model for evaluation.
        
    Returns:
        Configured RagasEvaluator instance.
    """
    logger.info(f"Creating RagasEvaluator (llm_model={llm_model})")
    return RagasEvaluator(metrics=metrics, llm_model=llm_model)
