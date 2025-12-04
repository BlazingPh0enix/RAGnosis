"""
Cross-modal reranking for DocuLens.

This module provides reranking capabilities to improve retrieval quality
by scoring retrieved nodes against the query using a cross-encoder model.
Supports both text and image summary nodes.

Excellence Track Feature: Cross-modal reranking combining vision-text embeddings.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

from llama_index.core.schema import NodeWithScore, TextNode

from retrieval.retriever import RetrievalResult
from config.logging_config import get_logger, log_execution_time, LogTimer

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
    from openai import OpenAI

logger = get_logger(__name__)


# Default reranker settings
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANK_TOP_K = 5


@dataclass
class RerankResult:
    """
    Result from a reranking operation.
    
    Attributes:
        query: The original query string.
        nodes: Reranked list of nodes with updated scores.
        original_nodes: Original nodes before reranking.
        rerank_time_ms: Time taken for reranking in milliseconds.
        model_name: Name of the reranker model used.
    """
    query: str
    nodes: List[NodeWithScore]
    original_nodes: List[NodeWithScore]
    rerank_time_ms: float = 0.0
    model_name: str = ""
    
    @property
    def text_nodes(self) -> List[NodeWithScore]:
        """Get only text and table nodes."""
        return [
            n for n in self.nodes
            if n.node.metadata.get("content_type") in ("text", "table")
        ]
    
    @property
    def image_nodes(self) -> List[NodeWithScore]:
        """Get only image summary nodes."""
        return [
            n for n in self.nodes
            if n.node.metadata.get("content_type") == "image_summary"
        ]
    
    @property
    def rank_changes(self) -> List[Dict[str, Any]]:
        """
        Calculate rank changes for each node after reranking.
        
        Returns:
            List of dicts with node_id, original_rank, new_rank, change.
        """
        changes = []
        
        # Build original rank map
        original_ranks = {
            n.node.node_id: i for i, n in enumerate(self.original_nodes)
        }
        
        for new_rank, node in enumerate(self.nodes):
            node_id = node.node.node_id
            original_rank = original_ranks.get(node_id, -1)
            
            changes.append({
                "node_id": node_id,
                "original_rank": original_rank,
                "new_rank": new_rank,
                "change": original_rank - new_rank if original_rank >= 0 else 0,
                "content_type": node.node.metadata.get("content_type", "text"),
                "original_score": self.original_nodes[original_rank].score if original_rank >= 0 else None,
                "rerank_score": node.score,
            })
        
        return changes
    
    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Get source information for all reranked nodes."""
        sources = []
        for node_with_score in self.nodes:
            node = node_with_score.node
            metadata = node.metadata
            node_text = node.get_content()
            
            source = {
                "page_number": metadata.get("page_number", 1),
                "content_type": metadata.get("content_type", "text"),
                "source_document": metadata.get("source_document", "unknown"),
                "score": node_with_score.score,
                "text_preview": node_text[:200] + "..." if len(node_text) > 200 else node_text,
            }
            
            if metadata.get("content_type") == "image_summary":
                source["image_name"] = metadata.get("image_name", "")
            
            sources.append(source)
        
        return sources
    
    def get_context_for_llm(self) -> str:
        """Format reranked nodes as context for LLM."""
        context_parts = []
        
        for node_with_score in self.nodes:
            node = node_with_score.node
            metadata = node.metadata
            page_num = metadata.get("page_number", 1)
            content_type = metadata.get("content_type", "text")
            
            if content_type == "image_summary":
                image_name = metadata.get("image_name", "image")
                prefix = f"[Page {page_num}, Image: {image_name}]"
            elif content_type == "table":
                prefix = f"[Page {page_num}, Table]"
            else:
                prefix = f"[Page {page_num}]"
            
            node_text = node.get_content()
            context_parts.append(f"{prefix}\n{node_text}")
        
        return "\n\n---\n\n".join(context_parts)


class CrossModalReranker:
    """
    Cross-modal reranker using a cross-encoder model.
    
    This reranker scores query-document pairs using a BGE reranker model
    to improve retrieval quality. It works with both text nodes and
    image summary nodes (which are text representations of images).
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        top_k: int = DEFAULT_RERANK_TOP_K,
        use_gpu: bool = False,
        batch_size: int = 32,
    ):
        """
        Initialize the cross-modal reranker.
        
        Args:
            model_name: HuggingFace model name for the reranker.
            top_k: Number of top results to return after reranking.
            use_gpu: Whether to use GPU for inference.
            batch_size: Batch size for reranking.
        """
        self.model_name = model_name
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        
        logger.info(f"Initializing CrossModalReranker (model={model_name}, top_k={top_k})")
        logger.debug(f"GPU: {use_gpu}, batch_size: {batch_size}")
        
        # Lazy load the model
        self._model: Optional["CrossEncoder"] = None
        self._tokenizer = None
    
    def _load_model(self) -> None:
        """Lazy load the reranker model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                device = "cuda" if self.use_gpu else "cpu"
                logger.info(f"Loading reranker model: {self.model_name} on {device}")
                self._model = CrossEncoder(
                    self.model_name,
                    device=device,
                    max_length=512,
                )
                logger.info(f"Reranker model loaded successfully")
            except ImportError:
                logger.error("sentence-transformers is required for reranking")
                raise ImportError(
                    "sentence-transformers is required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
    
    def _get_node_text(self, node: NodeWithScore) -> str:
        """
        Extract text content from a node for reranking.
        
        For image summary nodes, prepend metadata about the image.
        """
        content = node.node.get_content()
        metadata = node.node.metadata
        content_type = metadata.get("content_type", "text")
        
        # Add context prefix for image summaries
        if content_type == "image_summary":
            image_name = metadata.get("image_name", "image")
            page_num = metadata.get("page_number", 1)
            prefix = f"[Image {image_name} from page {page_num}] "
            return prefix + content
        
        return content
    
    @log_execution_time()
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank retrieved nodes using the cross-encoder model.
        
        Args:
            query: The query string.
            nodes: List of retrieved nodes to rerank.
            top_k: Override default top_k for this reranking.
            
        Returns:
            RerankResult with reranked nodes.
        """
        if not nodes:
            logger.debug("No nodes to rerank, returning empty result")
            return RerankResult(
                query=query,
                nodes=[],
                original_nodes=[],
                model_name=self.model_name,
            )
        
        # Load model if needed
        self._load_model()
        assert self._model is not None, "Model failed to load"
        
        top_k = top_k or self.top_k
        logger.info(f"Reranking {len(nodes)} nodes (top_k={top_k})")
        start_time = time.time()
        
        # Prepare query-document pairs
        pairs = [(query, self._get_node_text(node)) for node in nodes]
        
        # Score all pairs
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        # Create node-score pairs and sort
        scored_nodes = list(zip(nodes, scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Create new NodeWithScore objects with updated scores
        reranked_nodes = []
        for node, score in scored_nodes[:top_k]:
            new_node = NodeWithScore(
                node=node.node,
                score=float(score),
            )
            reranked_nodes.append(new_node)
        
        rerank_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Reranking completed in {rerank_time_ms:.2f}ms")
        
        return RerankResult(
            query=query,
            nodes=reranked_nodes,
            original_nodes=nodes,
            rerank_time_ms=rerank_time_ms,
            model_name=self.model_name,
        )
    
    def rerank_retrieval_result(
        self,
        result: RetrievalResult,
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank nodes from a RetrievalResult.
        
        Args:
            result: RetrievalResult from the retriever.
            top_k: Override default top_k.
            
        Returns:
            RerankResult with reranked nodes.
        """
        return self.rerank(
            query=result.query,
            nodes=result.nodes,
            top_k=top_k,
        )
    
    def rerank_multimodal(
        self,
        query: str,
        text_nodes: List[NodeWithScore],
        image_nodes: List[NodeWithScore],
        text_weight: float = 1.0,
        image_weight: float = 1.2,
        top_k: Optional[int] = None,
    ) -> RerankResult:
        """
        Rerank with configurable weights for different content types.
        
        This allows boosting image or text content based on query type.
        
        Args:
            query: The query string.
            text_nodes: Text/table nodes to rerank.
            image_nodes: Image summary nodes to rerank.
            text_weight: Score multiplier for text nodes.
            image_weight: Score multiplier for image nodes.
            top_k: Number of results to return.
            
        Returns:
            RerankResult with weighted reranked nodes.
        """
        if not text_nodes and not image_nodes:
            return RerankResult(
                query=query,
                nodes=[],
                original_nodes=[],
                model_name=self.model_name,
            )
        
        # Load model if needed
        self._load_model()
        assert self._model is not None, "Model failed to load"
        
        top_k = top_k or self.top_k
        start_time = time.time()
        
        all_nodes = text_nodes + image_nodes
        original_nodes = list(all_nodes)
        
        # Prepare query-document pairs
        pairs = [(query, self._get_node_text(node)) for node in all_nodes]
        
        # Score all pairs
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        
        # Apply content-type weights
        weighted_scores = []
        for i, (node, score) in enumerate(zip(all_nodes, scores)):
            content_type = node.node.metadata.get("content_type", "text")
            
            if content_type == "image_summary":
                weighted_score = float(score) * image_weight
            else:
                weighted_score = float(score) * text_weight
            
            weighted_scores.append(weighted_score)
        
        # Create scored pairs and sort
        scored_nodes = list(zip(all_nodes, weighted_scores))
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Create reranked nodes
        reranked_nodes = []
        for node, score in scored_nodes[:top_k]:
            new_node = NodeWithScore(
                node=node.node,
                score=score,
            )
            reranked_nodes.append(new_node)
        
        rerank_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Multimodal reranking completed in {rerank_time_ms:.2f}ms")
        
        return RerankResult(
            query=query,
            nodes=reranked_nodes,
            original_nodes=original_nodes,
            rerank_time_ms=rerank_time_ms,
            model_name=self.model_name,
        )


class LLMReranker:
    """
    LLM-based reranker using GPT models for semantic reranking.
    
    This is an alternative to cross-encoder reranking that uses
    an LLM to score relevance. Slower but potentially more accurate
    for complex queries.
    """
    
    def __init__(
        self,
        model: str = "gpt-5-nano-2025-08-07",
        top_k: int = DEFAULT_RERANK_TOP_K,
    ):
        """
        Initialize the LLM reranker.
        
        Args:
            model: OpenAI model to use for reranking.
            top_k: Number of top results to return.
        """
        self.model = model
        self.top_k = top_k
        self._client: Optional["OpenAI"] = None
        logger.info(f"Initializing LLMReranker (model={model}, top_k={top_k})")
    
    def _get_client(self) -> "OpenAI":
        """Lazy load OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    @log_execution_time()
    def rerank(
        self,
        query: str,
        nodes: List[NodeWithScore],
        top_k: Optional[int] = None,
    ) -> Optional[RerankResult]:
        """
        Rerank nodes using LLM-based scoring.
        
        Args:
            query: The query string.
            nodes: List of nodes to rerank.
            top_k: Override default top_k.
            
        Returns:
            RerankResult with reranked nodes.
        """
        if not nodes:
            logger.debug("No nodes to rerank, returning empty result")
            return RerankResult(
                query=query,
                nodes=[],
                original_nodes=[],
                model_name=self.model,
            )
        
        top_k = top_k or self.top_k
        logger.info(f"LLM reranking {len(nodes)} nodes (top_k={top_k}, model={self.model})")
        start_time = time.time()
        client = self._get_client()
        
        # Score each node
        scored_nodes = []
        for node in nodes:
            content = node.node.get_content()[:500]  # Truncate for efficiency
            
            prompt = f"""Rate how relevant this document passage is to the query on a scale of 0-10.
            
Query: {query}

Passage: {content}

Return only a number from 0-10."""
            
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=5,
                    temperature=0,
                )
                
                content_response = response.choices[0].message.content
                score_str = content_response.strip() if content_response else "5"
                score = float(score_str) / 10.0  # Normalize to 0-1
            except (ValueError, AttributeError):
                score = 0.5  # Default score on parse error
            
            scored_nodes.append((node, score))
        
        # Sort by score
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        # Create reranked nodes
        reranked_nodes = []
        for node, score in scored_nodes[:top_k]:
            new_node = NodeWithScore(
                node=node.node,
                score=score,
            )
            reranked_nodes.append(new_node)
        
        rerank_time_ms = (time.time() - start_time) * 1000
        logger.info(f"LLM reranking completed in {rerank_time_ms:.2f}ms")
        
        return RerankResult(
            query=query,
            nodes=reranked_nodes,
            original_nodes=nodes,
            rerank_time_ms=rerank_time_ms,
            model_name=self.model,
        )


def create_reranker(
    reranker_type: str = "cross-encoder",
    model_name: Optional[str] = None,
    top_k: int = DEFAULT_RERANK_TOP_K,
    use_gpu: bool = False,
) -> CrossModalReranker | LLMReranker:
    """
    Factory function to create a reranker.
    
    Args:
        reranker_type: Type of reranker - "cross-encoder" or "llm".
        model_name: Model name to use.
        top_k: Number of results to return after reranking.
        use_gpu: Whether to use GPU (cross-encoder only).
        
    Returns:
        Configured reranker instance.
    """
    logger.info(f"Creating reranker (type={reranker_type}, top_k={top_k})")
    
    if reranker_type == "llm":
        return LLMReranker(
            model=model_name or "gpt-5-nano-2025-08-07",
            top_k=top_k,
        )
    else:
        return CrossModalReranker(
            model_name=model_name or DEFAULT_RERANKER_MODEL,
            top_k=top_k,
            use_gpu=use_gpu,
        )
