"""
Benchmark query suite for DocuLens evaluation.

This module provides a collection of test queries for evaluating
the RAG system across different content types and complexity levels.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QueryType(Enum):
    """Types of queries for categorization."""
    TEXT_ONLY = "text_only"
    TABLE_BASED = "table_based"
    IMAGE_BASED = "image_based"
    CHART_BASED = "chart_based"
    MULTIMODAL = "multimodal"
    REASONING = "reasoning"
    FACTUAL = "factual"
    COMPARATIVE = "comparative"


class DifficultyLevel(Enum):
    """Difficulty levels for benchmark queries."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class BenchmarkQuery:
    """A single benchmark query with metadata."""
    
    query: str
    query_type: QueryType
    difficulty: DifficultyLevel
    ground_truth: Optional[str] = None
    expected_sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    description: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "difficulty": self.difficulty.value,
            "ground_truth": self.ground_truth,
            "expected_sources": self.expected_sources,
            "tags": self.tags,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkQuery":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            query_type=QueryType(data["query_type"]),
            difficulty=DifficultyLevel(data["difficulty"]),
            ground_truth=data.get("ground_truth"),
            expected_sources=data.get("expected_sources", []),
            tags=data.get("tags", []),
            description=data.get("description"),
        )


@dataclass
class BenchmarkSuite:
    """A collection of benchmark queries."""
    
    name: str
    description: str
    queries: list[BenchmarkQuery] = field(default_factory=list)
    version: str = "1.0.0"
    
    def add_query(self, query: BenchmarkQuery) -> None:
        """Add a query to the suite."""
        self.queries.append(query)
    
    def filter_by_type(self, query_type: QueryType) -> list[BenchmarkQuery]:
        """Filter queries by type."""
        return [q for q in self.queries if q.query_type == query_type]
    
    def filter_by_difficulty(self, difficulty: DifficultyLevel) -> list[BenchmarkQuery]:
        """Filter queries by difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]
    
    def filter_by_tags(self, tags: list[str]) -> list[BenchmarkQuery]:
        """Filter queries that have any of the specified tags."""
        return [q for q in self.queries if any(t in q.tags for t in tags)]
    
    def get_statistics(self) -> dict:
        """Get statistics about the benchmark suite."""
        stats = {
            "total_queries": len(self.queries),
            "by_type": {},
            "by_difficulty": {},
            "with_ground_truth": sum(1 for q in self.queries if q.ground_truth),
        }
        
        for qt in QueryType:
            count = len(self.filter_by_type(qt))
            if count > 0:
                stats["by_type"][qt.value] = count
        
        for dl in DifficultyLevel:
            count = len(self.filter_by_difficulty(dl))
            if count > 0:
                stats["by_difficulty"][dl.value] = count
        
        return stats
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "queries": [q.to_dict() for q in self.queries],
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkSuite":
        """Create from dictionary."""
        suite = cls(
            name=data["name"],
            description=data["description"],
            version=data.get("version", "1.0.0"),
        )
        for q_data in data.get("queries", []):
            suite.add_query(BenchmarkQuery.from_dict(q_data))
        return suite


# =============================================================================
# Pre-defined Benchmark Queries
# =============================================================================

# Text-only queries (general document understanding)
TEXT_QUERIES = [
    BenchmarkQuery(
        query="What is the main topic or purpose of this document?",
        query_type=QueryType.TEXT_ONLY,
        difficulty=DifficultyLevel.EASY,
        tags=["overview", "summary"],
        description="Basic document understanding query",
    ),
    BenchmarkQuery(
        query="Summarize the key points discussed in the document.",
        query_type=QueryType.TEXT_ONLY,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["summary", "comprehension"],
        description="Summarization capability test",
    ),
    BenchmarkQuery(
        query="What are the main conclusions or recommendations?",
        query_type=QueryType.TEXT_ONLY,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["conclusions", "recommendations"],
        description="Tests extraction of conclusions",
    ),
    BenchmarkQuery(
        query="What methodology or approach is described in the document?",
        query_type=QueryType.TEXT_ONLY,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["methodology", "technical"],
        description="Technical content extraction",
    ),
    BenchmarkQuery(
        query="Are there any limitations or caveats mentioned?",
        query_type=QueryType.TEXT_ONLY,
        difficulty=DifficultyLevel.HARD,
        tags=["limitations", "critical-analysis"],
        description="Tests nuanced understanding",
    ),
]

# Table-based queries
TABLE_QUERIES = [
    BenchmarkQuery(
        query="What data is presented in the tables?",
        query_type=QueryType.TABLE_BASED,
        difficulty=DifficultyLevel.EASY,
        tags=["tables", "data"],
        description="Basic table understanding",
    ),
    BenchmarkQuery(
        query="What are the key metrics or values shown in the tables?",
        query_type=QueryType.TABLE_BASED,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["tables", "metrics"],
        description="Table data extraction",
    ),
    BenchmarkQuery(
        query="Compare the values across different rows or columns in the table.",
        query_type=QueryType.TABLE_BASED,
        difficulty=DifficultyLevel.HARD,
        tags=["tables", "comparison"],
        description="Table comparison analysis",
    ),
    BenchmarkQuery(
        query="What trends can be observed from the tabular data?",
        query_type=QueryType.TABLE_BASED,
        difficulty=DifficultyLevel.HARD,
        tags=["tables", "trends", "analysis"],
        description="Trend analysis from tables",
    ),
]

# Image and chart-based queries
IMAGE_QUERIES = [
    BenchmarkQuery(
        query="Describe the images or diagrams in the document.",
        query_type=QueryType.IMAGE_BASED,
        difficulty=DifficultyLevel.EASY,
        tags=["images", "diagrams"],
        description="Basic image description",
    ),
    BenchmarkQuery(
        query="What do the charts or graphs show?",
        query_type=QueryType.CHART_BASED,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["charts", "graphs", "visualization"],
        description="Chart interpretation",
    ),
    BenchmarkQuery(
        query="What trends are visible in the charts?",
        query_type=QueryType.CHART_BASED,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["charts", "trends"],
        description="Visual trend analysis",
    ),
    BenchmarkQuery(
        query="How do the figures support the main arguments in the text?",
        query_type=QueryType.IMAGE_BASED,
        difficulty=DifficultyLevel.HARD,
        tags=["images", "reasoning", "integration"],
        description="Figure-text integration understanding",
    ),
]

# Multi-modal queries (combining text, tables, and images)
MULTIMODAL_QUERIES = [
    BenchmarkQuery(
        query="How do the tables and charts relate to the textual content?",
        query_type=QueryType.MULTIMODAL,
        difficulty=DifficultyLevel.MEDIUM,
        tags=["multimodal", "integration"],
        description="Cross-modal understanding",
    ),
    BenchmarkQuery(
        query="Provide a comprehensive summary including information from text, tables, and figures.",
        query_type=QueryType.MULTIMODAL,
        difficulty=DifficultyLevel.HARD,
        tags=["multimodal", "summary", "comprehensive"],
        description="Full multi-modal synthesis",
    ),
    BenchmarkQuery(
        query="What evidence from different parts of the document supports the main conclusion?",
        query_type=QueryType.MULTIMODAL,
        difficulty=DifficultyLevel.HARD,
        tags=["multimodal", "evidence", "reasoning"],
        description="Evidence synthesis across modalities",
    ),
]

# Reasoning queries
REASONING_QUERIES = [
    BenchmarkQuery(
        query="Based on the document, what can be inferred about future implications?",
        query_type=QueryType.REASONING,
        difficulty=DifficultyLevel.HARD,
        tags=["inference", "reasoning", "implications"],
        description="Inference and reasoning test",
    ),
    BenchmarkQuery(
        query="What assumptions underlie the analysis presented?",
        query_type=QueryType.REASONING,
        difficulty=DifficultyLevel.HARD,
        tags=["assumptions", "critical-thinking"],
        description="Critical analysis of assumptions",
    ),
    BenchmarkQuery(
        query="How would the conclusions change if the data were different?",
        query_type=QueryType.REASONING,
        difficulty=DifficultyLevel.HARD,
        tags=["reasoning", "hypothetical"],
        description="Hypothetical reasoning test",
    ),
]

# Factual queries
FACTUAL_QUERIES = [
    BenchmarkQuery(
        query="What specific numbers or statistics are mentioned?",
        query_type=QueryType.FACTUAL,
        difficulty=DifficultyLevel.EASY,
        tags=["factual", "numbers", "statistics"],
        description="Factual number extraction",
    ),
    BenchmarkQuery(
        query="What dates or time periods are referenced?",
        query_type=QueryType.FACTUAL,
        difficulty=DifficultyLevel.EASY,
        tags=["factual", "dates", "timeline"],
        description="Temporal information extraction",
    ),
    BenchmarkQuery(
        query="Who are the authors or key people mentioned?",
        query_type=QueryType.FACTUAL,
        difficulty=DifficultyLevel.EASY,
        tags=["factual", "people", "authors"],
        description="Entity extraction",
    ),
]


def create_default_benchmark_suite() -> BenchmarkSuite:
    """Create the default benchmark suite with all pre-defined queries."""
    suite = BenchmarkSuite(
        name="DocuLens Default Benchmark",
        description="Comprehensive benchmark suite for evaluating DocuLens RAG system",
        version="1.0.0",
    )
    
    # Add all query categories
    for queries in [
        TEXT_QUERIES,
        TABLE_QUERIES,
        IMAGE_QUERIES,
        MULTIMODAL_QUERIES,
        REASONING_QUERIES,
        FACTUAL_QUERIES,
    ]:
        for query in queries:
            suite.add_query(query)
    
    return suite


def create_quick_benchmark_suite() -> BenchmarkSuite:
    """Create a smaller benchmark suite for quick evaluation."""
    suite = BenchmarkSuite(
        name="DocuLens Quick Benchmark",
        description="Quick benchmark suite for fast evaluation",
        version="1.0.0",
    )
    
    # Select representative queries from each category
    quick_queries = [
        TEXT_QUERIES[0],      # Easy text
        TEXT_QUERIES[1],      # Medium text
        TABLE_QUERIES[1],     # Medium table
        IMAGE_QUERIES[1],     # Medium chart
        MULTIMODAL_QUERIES[0], # Medium multimodal
        FACTUAL_QUERIES[0],   # Easy factual
    ]
    
    for query in quick_queries:
        suite.add_query(query)
    
    return suite


def create_custom_benchmark_suite(
    name: str,
    queries: list[dict],
    description: str = "",
) -> BenchmarkSuite:
    """
    Create a custom benchmark suite from query dictionaries.
    
    Args:
        name: Name of the benchmark suite
        queries: List of query dictionaries with keys:
                 - query (required): The query text
                 - query_type (optional): QueryType value, defaults to TEXT_ONLY
                 - difficulty (optional): DifficultyLevel value, defaults to MEDIUM
                 - ground_truth (optional): Expected answer
                 - tags (optional): List of tags
        description: Description of the benchmark suite
    
    Returns:
        BenchmarkSuite with the custom queries
    """
    suite = BenchmarkSuite(
        name=name,
        description=description,
    )
    
    for q_dict in queries:
        query = BenchmarkQuery(
            query=q_dict["query"],
            query_type=QueryType(q_dict.get("query_type", "text_only")),
            difficulty=DifficultyLevel(q_dict.get("difficulty", "medium")),
            ground_truth=q_dict.get("ground_truth"),
            expected_sources=q_dict.get("expected_sources", []),
            tags=q_dict.get("tags", []),
            description=q_dict.get("description"),
        )
        suite.add_query(query)
    
    return suite


# Factory function for easy access
def get_benchmark_suite(suite_type: str = "default") -> BenchmarkSuite:
    """
    Get a pre-defined benchmark suite.
    
    Args:
        suite_type: Type of suite - "default" for full suite, "quick" for smaller suite
    
    Returns:
        BenchmarkSuite instance
    """
    if suite_type == "quick":
        return create_quick_benchmark_suite()
    return create_default_benchmark_suite()
