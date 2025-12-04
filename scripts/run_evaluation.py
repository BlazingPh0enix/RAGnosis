"""
Evaluation runner script for DocuLens.

This script runs benchmark queries against the RAG system and
evaluates the results using RAGAS metrics.

Usage:
    python -m scripts.run_evaluation --help
    python -m scripts.run_evaluation --suite default
    python -m scripts.run_evaluation --suite quick --output results.json
    python -m scripts.run_evaluation --query "What is the main topic?"
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from config.settings import get_settings
from evaluation.benchmark_queries import (
    BenchmarkQuery,
    BenchmarkSuite,
    get_benchmark_suite,
    DifficultyLevel,
    QueryType,
)
from evaluation.ragas_evaluator import (
    RagasEvaluator,
    EvaluationResult,
    EvaluationSample,
    create_evaluator,
)
from retrieval.query_engine import DocuLensQueryEngine, create_query_engine


def run_single_query(
    query_engine: DocuLensQueryEngine,
    evaluator: RagasEvaluator,
    query: str,
    ground_truth: Optional[str] = None,
    verbose: bool = True,
) -> EvaluationResult:
    """
    Run a single query and evaluate the response.
    
    Args:
        query_engine: The DocuLens query engine
        evaluator: The RAGAS evaluator
        query: The query to run
        ground_truth: Optional ground truth answer
        verbose: Whether to print progress
    
    Returns:
        EvaluationResult with metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
    
    # Run query
    result = query_engine.query(query)
    
    # Extract context from source nodes
    contexts = []
    for node_with_score in result.source_nodes:
        text = node_with_score.node.get_content()
        if text:
            contexts.append(text)
    
    if verbose:
        response_preview = result.response[:500] + "..." if len(result.response) > 500 else result.response
        print(f"\nResponse: {response_preview}")
        print(f"\nSources: {len(result.source_nodes)} nodes retrieved")
    
    # Evaluate
    eval_result = evaluator.evaluate_single(
        question=query,
        answer=result.response,
        contexts=contexts,
        ground_truth=ground_truth,
    )
    
    if verbose:
        print(f"\n--- Metrics ---")
        for metric, value in eval_result.metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
    
    return eval_result


def run_benchmark_suite(
    query_engine: DocuLensQueryEngine,
    evaluator: RagasEvaluator,
    suite: BenchmarkSuite,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a full benchmark suite and generate report.
    
    Args:
        query_engine: The DocuLens query engine
        evaluator: The RAGAS evaluator
        suite: The benchmark suite to run
        verbose: Whether to print progress
    
    Returns:
        Report dictionary with all results and aggregate metrics
    """
    if verbose:
        stats = suite.get_statistics()
        print(f"\n{'#'*60}")
        print(f"Running Benchmark Suite: {suite.name}")
        print(f"Total queries: {stats['total_queries']}")
        print(f"By type: {stats['by_type']}")
        print(f"By difficulty: {stats['by_difficulty']}")
        print(f"{'#'*60}")
    
    samples: List[EvaluationSample] = []
    sample_metadata: List[Dict[str, Any]] = []
    failed_queries: List[Dict[str, Any]] = []
    
    for i, benchmark_query in enumerate(suite.queries, 1):
        if verbose:
            print(f"\n[{i}/{len(suite.queries)}] Running: {benchmark_query.query_type.value} - {benchmark_query.difficulty.value}")
        
        try:
            # Run query
            query_result = query_engine.query(benchmark_query.query)
            
            # Extract contexts
            contexts = []
            for node_with_score in query_result.source_nodes:
                text = node_with_score.node.get_content()
                if text:
                    contexts.append(text)
            
            # Create sample for batch evaluation
            sample = evaluator.create_sample(
                question=benchmark_query.query,
                answer=query_result.response,
                contexts=contexts,
                ground_truth=benchmark_query.ground_truth,
            )
            samples.append(sample)
            
            # Track metadata for this sample
            sample_metadata.append({
                "query_type": benchmark_query.query_type.value,
                "difficulty": benchmark_query.difficulty.value,
                "tags": benchmark_query.tags,
                "num_sources": len(query_result.source_nodes),
            })
            
            if verbose:
                print(f"    Response: {query_result.response[:100]}...")
        
        except Exception as e:
            print(f"    ERROR: {e}")
            failed_queries.append({
                "query": benchmark_query.query,
                "error": str(e),
            })
    
    # Run batch evaluation
    if verbose:
        print(f"\n{'#'*60}")
        print("Running RAGAS Evaluation...")
        print(f"{'#'*60}")
    
    eval_result = evaluator.evaluate(samples)
    
    # Build report
    report = {
        "suite_name": suite.name,
        "suite_version": suite.version,
        "timestamp": datetime.now().isoformat(),
        "num_queries": len(suite.queries),
        "num_successful": len(samples),
        "num_failed": len(failed_queries),
        "aggregate_metrics": eval_result.metrics,
        "per_sample_scores": eval_result.per_sample_scores,
        "sample_metadata": sample_metadata,
        "failed_queries": failed_queries,
    }
    
    if verbose:
        print(f"\n{'#'*60}")
        print("AGGREGATE METRICS")
        print(f"{'#'*60}")
        for metric, value in eval_result.metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.4f}")
    
    return report


def save_results(
    report: Dict[str, Any],
    output_path: Path,
    format: str = "json",
) -> None:
    """
    Save evaluation results to file.
    
    Args:
        report: The evaluation report dictionary
        output_path: Path to save results
        format: Output format ("json" or "csv")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to: {output_path}")
    
    elif format == "csv":
        import csv
        
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            # For single result
            if "samples" in report:
                samples = report["samples"]
                per_sample = report.get("per_sample_scores", [])
                
                # Header
                metric_keys = list(per_sample[0].keys()) if per_sample else []
                headers = ["question", "answer_preview"] + metric_keys
                writer.writerow(headers)
                
                # Data rows
                for i, sample in enumerate(samples):
                    row = [
                        sample.get("question", ""),
                        sample.get("answer", "")[:100] + "..." if len(sample.get("answer", "")) > 100 else sample.get("answer", ""),
                    ]
                    if i < len(per_sample):
                        row.extend(per_sample[i].get(k, "") for k in metric_keys)
                    writer.writerow(row)
            else:
                # For benchmark report
                per_sample = report.get("per_sample_scores", [])
                metadata = report.get("sample_metadata", [])
                
                # Header
                metric_keys = list(per_sample[0].keys()) if per_sample else []
                headers = ["query_type", "difficulty"] + metric_keys
                writer.writerow(headers)
                
                # Data rows
                for i, scores in enumerate(per_sample):
                    meta = metadata[i] if i < len(metadata) else {}
                    row = [
                        meta.get("query_type", ""),
                        meta.get("difficulty", ""),
                    ]
                    row.extend(scores.get(k, "") for k in metric_keys)
                    writer.writerow(row)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run DocuLens RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run full benchmark suite:
    python -m scripts.run_evaluation --suite default

  Run quick benchmark:
    python -m scripts.run_evaluation --suite quick

  Run single query:
    python -m scripts.run_evaluation --query "What is the main topic?"

  Save results to file:
    python -m scripts.run_evaluation --suite quick --output results.json

  Filter by difficulty:
    python -m scripts.run_evaluation --suite default --difficulty easy

  Filter by query type:
    python -m scripts.run_evaluation --suite default --type text_only
        """,
    )
    
    parser.add_argument(
        "--suite",
        type=str,
        choices=["default", "quick"],
        help="Benchmark suite to run",
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to evaluate",
    )
    
    parser.add_argument(
        "--ground-truth",
        type=str,
        help="Ground truth answer for single query (enables context_recall)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for results (supports .json and .csv)",
    )
    
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Filter benchmark queries by difficulty",
    )
    
    parser.add_argument(
        "--type",
        type=str,
        choices=["text_only", "table_based", "image_based", "chart_based", "multimodal", "reasoning", "factual", "comparative"],
        help="Filter benchmark queries by type",
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        help="Qdrant collection name to query",
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="List all queries in the benchmark suite without running them",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.suite and not args.query and not args.list_queries:
        parser.error("Either --suite, --query, or --list-queries is required")
    
    verbose = not args.quiet
    
    # List queries mode
    if args.list_queries:
        suite = get_benchmark_suite(args.suite or "default")
        print(f"\nBenchmark Suite: {suite.name}")
        print(f"{'='*60}")
        
        for i, q in enumerate(suite.queries, 1):
            print(f"\n{i}. [{q.query_type.value}] [{q.difficulty.value}]")
            print(f"   {q.query}")
            if q.tags:
                print(f"   Tags: {', '.join(q.tags)}")
        
        stats = suite.get_statistics()
        print(f"\n{'='*60}")
        print(f"Total: {stats['total_queries']} queries")
        return
    
    # Initialize components
    settings = get_settings()
    collection_name = args.collection or settings.QDRANT_COLLECTION
    
    if verbose:
        print(f"Initializing DocuLens evaluation...")
        print(f"Collection: {collection_name}")
    
    try:
        query_engine = create_query_engine(collection_name=collection_name)
        evaluator = create_evaluator()
    except Exception as e:
        print(f"ERROR: Failed to initialize: {e}")
        print("Make sure Qdrant is running and the collection exists.")
        sys.exit(1)
    
    # Run evaluation
    if args.query:
        # Single query mode
        result = run_single_query(
            query_engine=query_engine,
            evaluator=evaluator,
            query=args.query,
            ground_truth=args.ground_truth,
            verbose=verbose,
        )
        
        if args.output:
            output_path = Path(args.output)
            report = result.to_dict()
            save_results(report, output_path, format="csv" if output_path.suffix == ".csv" else "json")
    
    else:
        # Benchmark suite mode
        suite = get_benchmark_suite(args.suite)
        
        # Apply filters
        if args.difficulty:
            filtered = suite.filter_by_difficulty(DifficultyLevel(args.difficulty))
            suite = BenchmarkSuite(
                name=f"{suite.name} (filtered: {args.difficulty})",
                description=suite.description,
                queries=filtered,
            )
        
        if args.type:
            filtered = suite.filter_by_type(QueryType(args.type))
            suite = BenchmarkSuite(
                name=f"{suite.name} (filtered: {args.type})",
                description=suite.description,
                queries=filtered,
            )
        
        if not suite.queries:
            print("No queries match the specified filters.")
            sys.exit(1)
        
        report = run_benchmark_suite(
            query_engine=query_engine,
            evaluator=evaluator,
            suite=suite,
            verbose=verbose,
        )
        
        if args.output:
            output_path = Path(args.output)
            save_results(report, output_path, format="csv" if output_path.suffix == ".csv" else "json")
        
        # Summary
        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION COMPLETE")
            print(f"{'='*60}")
            print(f"Total queries: {report['num_queries']}")
            print(f"Successful: {report['num_successful']}")
            print(f"Failed: {report['num_failed']}")
            print(f"\nAggregate Metrics:")
            for metric, value in report['aggregate_metrics'].items():
                if value is not None:
                    print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
