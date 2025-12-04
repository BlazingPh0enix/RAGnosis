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
    """Result from a reranking operation."""
    query: str
    nodes: List[NodeWithScore]
    original_nodes: List[NodeWithScore]
    rerank_time_ms: float = 0.0
    model_name: str = ""
    
    def get_nodes_by_type(self, content_type: str) -> List[NodeWithScore]:
        """Get nodes filtered by content type ('text', 'table', 'image_summary')."""
        types = ("text", "table") if content_type == "text" else (content_type,)
        return [n for n in self.nodes if n.node.metadata.get("content_type") in types]
    
    @property
    def text_nodes(self) -> List[NodeWithScore]:
        return self.get_nodes_by_type("text")
    
    @property
    def image_nodes(self) -> List[NodeWithScore]:
        return self.get_nodes_by_type("image_summary")
    
    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Get source information for all reranked nodes."""
        return [self._format_source(n) for n in self.nodes]
    
    def _format_source(self, node_with_score: NodeWithScore) -> Dict[str, Any]:
        node, metadata = node_with_score.node, node_with_score.node.metadata
        text = node.get_content()
        source = {
            "page_number": metadata.get("page_number", 1),
            "content_type": metadata.get("content_type", "text"),
            "source_document": metadata.get("source_document", "unknown"),
            "score": node_with_score.score,
            "text_preview": text[:200] + "..." if len(text) > 200 else text,
        }
        if metadata.get("content_type") == "image_summary":
            source["image_name"] = metadata.get("image_name", "")
        return source
    
    def get_context_for_llm(self) -> str:
        """Format reranked nodes as context for LLM."""
        parts = []
        for n in self.nodes:
            meta = n.node.metadata
            page, ctype = meta.get("page_number", 1), meta.get("content_type", "text")
            if ctype == "image_summary":
                prefix = f"[Page {page}, Image: {meta.get('image_name', 'image')}]"
            elif ctype == "table":
                prefix = f"[Page {page}, Table]"
            else:
                prefix = f"[Page {page}]"
            parts.append(f"{prefix}\n{n.node.get_content()}")
        return "\n\n---\n\n".join(parts)


class CrossModalReranker:
    """Cross-modal reranker using a cross-encoder model (BGE reranker)."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        top_k: int = DEFAULT_RERANK_TOP_K,
        use_gpu: bool = False,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._model: Optional["CrossEncoder"] = None
        logger.info(f"Initializing CrossModalReranker (model={model_name}, top_k={top_k})")
    
    def _load_model(self) -> None:
        """Lazy load the reranker model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            device = "cuda" if self.use_gpu else "cpu"
            logger.info(f"Loading reranker model: {self.model_name} on {device}")
            self._model = CrossEncoder(self.model_name, device=device, max_length=512)
        except ImportError:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
    
    def _get_node_text(self, node: NodeWithScore) -> str:
        """Extract text content from a node, adding context for image summaries."""
        content, meta = node.node.get_content(), node.node.metadata
        if meta.get("content_type") == "image_summary":
            return f"[Image {meta.get('image_name', 'image')} from page {meta.get('page_number', 1)}] {content}"
        return content
    
    @log_execution_time()
    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: Optional[int] = None) -> RerankResult:
        """Rerank retrieved nodes using the cross-encoder model."""
        if not nodes:
            return RerankResult(query=query, nodes=[], original_nodes=[], model_name=self.model_name)
        
        self._load_model()
        top_k = top_k or self.top_k
        logger.info(f"Reranking {len(nodes)} nodes (top_k={top_k})")
        start_time = time.time()
        
        pairs = [(query, self._get_node_text(node)) for node in nodes]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)  # type: ignore
        
        scored_nodes = sorted(zip(nodes, scores), key=lambda x: x[1], reverse=True)
        reranked = [NodeWithScore(node=n.node, score=float(s)) for n, s in scored_nodes[:top_k]]
        
        return RerankResult(
            query=query, nodes=reranked, original_nodes=nodes,
            rerank_time_ms=(time.time() - start_time) * 1000, model_name=self.model_name
        )
    
    def rerank_retrieval_result(self, result: RetrievalResult, top_k: Optional[int] = None) -> RerankResult:
        """Rerank nodes from a RetrievalResult."""
        return self.rerank(query=result.query, nodes=result.nodes, top_k=top_k)
    
    def rerank_multimodal(
        self, query: str, text_nodes: List[NodeWithScore], image_nodes: List[NodeWithScore],
        text_weight: float = 1.0, image_weight: float = 1.2, top_k: Optional[int] = None,
    ) -> RerankResult:
        """Rerank with configurable weights for different content types."""
        all_nodes = text_nodes + image_nodes
        if not all_nodes:
            return RerankResult(query=query, nodes=[], original_nodes=[], model_name=self.model_name)
        
        self._load_model()
        top_k, start_time = top_k or self.top_k, time.time()
        
        pairs = [(query, self._get_node_text(node)) for node in all_nodes]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)  # type: ignore
        
        # Apply weights based on content type
        weighted = [
            (n, float(s) * (image_weight if n.node.metadata.get("content_type") == "image_summary" else text_weight))
            for n, s in zip(all_nodes, scores)
        ]
        scored_nodes = sorted(weighted, key=lambda x: x[1], reverse=True)
        reranked = [NodeWithScore(node=n.node, score=s) for n, s in scored_nodes[:top_k]]
        
        return RerankResult(
            query=query, nodes=reranked, original_nodes=all_nodes,
            rerank_time_ms=(time.time() - start_time) * 1000, model_name=self.model_name
        )


class LLMReranker:
    """LLM-based reranker using GPT models for semantic reranking."""
    
    def __init__(self, model: str = "gpt-5-nano-2025-08-07", top_k: int = DEFAULT_RERANK_TOP_K):
        self.model, self.top_k = model, top_k
        self._client: Optional["OpenAI"] = None
        logger.info(f"Initializing LLMReranker (model={model}, top_k={top_k})")
    
    def _get_client(self) -> "OpenAI":
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    @log_execution_time()
    def rerank(self, query: str, nodes: List[NodeWithScore], top_k: Optional[int] = None) -> RerankResult:
        """Rerank nodes using LLM-based scoring."""
        if not nodes:
            return RerankResult(query=query, nodes=[], original_nodes=[], model_name=self.model)
        
        top_k, start_time = top_k or self.top_k, time.time()
        logger.info(f"LLM reranking {len(nodes)} nodes (top_k={top_k})")
        client = self._get_client()
        
        scored_nodes = []
        for node in nodes:
            content = node.node.get_content()[:500]
            prompt = f"Rate relevance 0-10.\nQuery: {query}\nPassage: {content}\nReturn only a number."
            try:
                resp = client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    max_tokens=5, temperature=0
                )
                score = float((resp.choices[0].message.content or "5").strip()) / 10.0
            except (ValueError, AttributeError):
                score = 0.5
            scored_nodes.append((node, score))
        
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        reranked = [NodeWithScore(node=n.node, score=s) for n, s in scored_nodes[:top_k]]
        
        return RerankResult(
            query=query, nodes=reranked, original_nodes=nodes,
            rerank_time_ms=(time.time() - start_time) * 1000, model_name=self.model
        )


def create_reranker(
    reranker_type: str = "cross-encoder", model_name: Optional[str] = None,
    top_k: int = DEFAULT_RERANK_TOP_K, use_gpu: bool = False,
) -> CrossModalReranker | LLMReranker:
    """Factory function to create a reranker ('cross-encoder' or 'llm')."""
    logger.info(f"Creating reranker (type={reranker_type}, top_k={top_k})")
    if reranker_type == "llm":
        return LLMReranker(model=model_name or "gpt-5-nano-2025-08-07", top_k=top_k)
    return CrossModalReranker(model_name=model_name or DEFAULT_RERANKER_MODEL, top_k=top_k, use_gpu=use_gpu)
