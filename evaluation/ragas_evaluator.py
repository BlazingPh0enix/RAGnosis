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
    """A single evaluation sample for RAGAS."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"question": self.question, "answer": self.answer, "contexts": self.contexts}
        if self.ground_truth:
            result["ground_truth"] = self.ground_truth
        return result


@dataclass
class EvaluationResult:
    """Results from a RAGAS evaluation run."""
    samples: List[EvaluationSample]
    metrics: Dict[str, float]
    per_sample_scores: List[Dict[str, float]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "num_samples": len(self.samples),
                "metrics": self.metrics, "per_sample_scores": self.per_sample_scores,
                "metadata": self.metadata}
    
    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @property
    def faithfulness_score(self) -> float:
        return self.metrics.get("faithfulness", 0.0)
    
    @property
    def answer_relevancy_score(self) -> float:
        return self.metrics.get("answer_relevancy", 0.0)
    
    @property
    def context_precision_score(self) -> float:
        return self.metrics.get("context_precision", 0.0)
    
    def summary(self) -> str:
        lines = ["=" * 50, "RAGAS Evaluation Results", "=" * 50,
                 f"Samples: {len(self.samples)}", "", "Metrics:"]
        lines.extend(f"  {k}: {v:.4f}" for k, v in self.metrics.items())
        lines.append("=" * 50)
        return "\n".join(lines)


class RagasEvaluator:
    """Evaluator using RAGAS metrics for RAG quality (faithfulness, relevancy, precision)."""
    
    def __init__(self, metrics: Optional[List] = None, llm_model: str = "gpt-5-nano-2025-08-07"):
        self.metrics = metrics or DEFAULT_METRICS
        self.llm_model = llm_model
        self._results_history: List[EvaluationResult] = []
        logger.info(f"RagasEvaluator initialized with metrics: {[m.name for m in self.metrics]}")
    
    def create_sample(
        self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None,
    ) -> EvaluationSample:
        """Create an evaluation sample."""
        return EvaluationSample(question=question, answer=answer, contexts=contexts, ground_truth=ground_truth)
    
    def create_sample_from_query_result(self, query_result, ground_truth: Optional[str] = None) -> EvaluationSample:
        """Create an evaluation sample from a QueryResult."""
        contexts = [s.get("text_preview", "") for s in query_result.sources if s.get("text_preview")]
        return EvaluationSample(
            question=query_result.query, answer=query_result.response,
            contexts=contexts, ground_truth=ground_truth
        )
    
    @log_execution_time()
    def evaluate(self, samples: List[EvaluationSample], raise_exceptions: bool = False) -> EvaluationResult:
        """Evaluate a list of samples using RAGAS metrics."""
        if not samples:
            logger.warning("No samples provided")
            return EvaluationResult(samples=[], metrics={}, metadata={"error": "No samples"})
        
        logger.info(f"Starting RAGAS evaluation with {len(samples)} samples")
        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
        }
        if any(s.ground_truth for s in samples):
            data["ground_truth"] = [s.ground_truth or "" for s in samples]
        
        dataset = Dataset.from_dict(data)
        
        try:
            with LogTimer(logger, "RAGAS evaluation"):
                result = evaluate(dataset=dataset, metrics=self.metrics, raise_exceptions=raise_exceptions)
            
            metrics = {}
            for m in self.metrics:
                if m.name in result:
                    val = result[m.name]
                    # Ensure we don't call float() on a list or None; handle empty lists safely
                    if val is None:
                        metrics[m.name] = 0.0
                    elif isinstance(val, list):
                        metrics[m.name] = sum(val) / len(val) if len(val) > 0 else 0.0
                    else:
                        metrics[m.name] = float(val)
            
            per_sample = []
            if hasattr(result, 'to_pandas'):
                for _, row in result.to_pandas().iterrows():
                    per_sample.append({m.name: float(row[m.name]) for m in self.metrics if m.name in row})
            
            eval_result = EvaluationResult(
                samples=samples, metrics=metrics, per_sample_scores=per_sample,
                metadata={"llm_model": self.llm_model, "metrics_used": [m.name for m in self.metrics]},
            )
            self._results_history.append(eval_result)
            logger.info(f"Evaluation completed. Metrics: {metrics}")
            return eval_result
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}", exc_info=True)
            if raise_exceptions:
                raise
            return EvaluationResult(samples=samples, metrics={}, metadata={"error": str(e)})
    
    def evaluate_single(
        self, question: str, answer: str, contexts: List[str], ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a single question-answer pair."""
        return self.evaluate([self.create_sample(question, answer, contexts, ground_truth)])
    
    def get_results_history(self) -> List[EvaluationResult]:
        return self._results_history
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Calculate average metrics across all historical results."""
        if not self._results_history:
            return {}
        sums, counts = {}, {}
        for r in self._results_history:
            for name, score in r.metrics.items():
                sums[name] = sums.get(name, 0.0) + score
                counts[name] = counts.get(name, 0) + 1
        return {name: sums[name] / counts[name] for name in sums}


def create_evaluator(metrics: Optional[List] = None, llm_model: str = "gpt-5-nano-2025-08-07") -> RagasEvaluator:
    """Factory function to create a RagasEvaluator."""
    return RagasEvaluator(metrics=metrics, llm_model=llm_model)
