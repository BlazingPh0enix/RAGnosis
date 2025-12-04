"""
Hybrid retrieval system for DocuLens.

This module provides retriever classes that support retrieving
both text/table nodes and image summary nodes from the Qdrant index.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import VectorIndexRetriever

from index.indexer import Indexer, create_indexer
from index.embeddings import create_embedding_service, configure_global_embeddings
from config.logging_config import get_logger, log_execution_time, LogTimer

logger = get_logger(__name__)


# Default retrieval settings
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.0


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    nodes: List[NodeWithScore]
    
    def _filter_by_type(self, types: tuple) -> List[NodeWithScore]:
        return [n for n in self.nodes if n.node.metadata.get("content_type") in types]
    
    @property
    def text_nodes(self) -> List[NodeWithScore]:
        return self._filter_by_type(("text", "table"))
    
    @property
    def image_nodes(self) -> List[NodeWithScore]:
        return self._filter_by_type(("image_summary",))
    
    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Get source information for all retrieved nodes."""
        sources = []
        for n in self.nodes:
            meta, text = n.node.metadata, n.node.get_content()
            source = {
                "page_number": meta.get("page_number", 1),
                "content_type": meta.get("content_type", "text"),
                "source_document": meta.get("source_document", "unknown"),
                "score": n.score,
                "text_preview": text[:200] + "..." if len(text) > 200 else text,
            }
            if meta.get("content_type") == "image_summary":
                source["image_name"] = meta.get("image_name", "")
            sources.append(source)
        return sources
    
    def get_context_for_llm(self) -> str:
        """Format retrieved nodes as context for LLM."""
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


class DocumentRetriever:
    """Hybrid retriever for multi-modal document content."""
    
    def __init__(
        self, index: VectorStoreIndex, top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.index, self.top_k, self.similarity_threshold = index, top_k, similarity_threshold
        self._retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        logger.info(f"DocumentRetriever initialized (top_k={top_k}, threshold={similarity_threshold})")
    
    @log_execution_time()
    def retrieve(
        self, query: str, top_k: Optional[int] = None, filter_content_types: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """Retrieve relevant nodes for a query."""
        effective_top_k = top_k or self.top_k
        self._retriever.similarity_top_k = effective_top_k
        
        with LogTimer(logger, "Vector retrieval"):
            nodes = self._retriever.retrieve(query)
        
        # Filter by threshold
        if self.similarity_threshold > 0:
            nodes = [n for n in nodes if n.score is not None and n.score >= self.similarity_threshold]
        
        # Filter by content type
        if filter_content_types:
            nodes = [n for n in nodes if n.node.metadata.get("content_type") in filter_content_types]
        
        logger.info(f"Retrieved {len(nodes)} nodes")
        return RetrievalResult(query=query, nodes=nodes)
    
    def retrieve_text_only(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return self.retrieve(query, top_k, filter_content_types=["text", "table"])
    
    def retrieve_images_only(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        return self.retrieve(query, top_k, filter_content_types=["image_summary"])
    
    def retrieve_multimodal(self, query: str, text_top_k: int = 3, image_top_k: int = 2) -> RetrievalResult:
        """Retrieve a balanced mix of text and image nodes."""
        text_nodes = self.retrieve(query, text_top_k * 2, ["text", "table"]).nodes[:text_top_k]
        image_nodes = self.retrieve(query, image_top_k * 2, ["image_summary"]).nodes[:image_top_k]
        all_nodes = sorted(text_nodes + image_nodes, key=lambda x: x.score or 0, reverse=True)
        return RetrievalResult(query=query, nodes=all_nodes)


def create_retriever(
    indexer: Optional[Indexer] = None, index: Optional[VectorStoreIndex] = None,
    top_k: int = DEFAULT_TOP_K, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    collection_name: Optional[str] = None, embedding_model: Optional[str] = None,
) -> DocumentRetriever:
    """Factory function to create a DocumentRetriever."""
    logger.info(f"Creating DocumentRetriever (top_k={top_k})")
    if index is None:
        if indexer is None:
            indexer = create_indexer(collection_name=collection_name, embedding_model=embedding_model)
        index = indexer.load_index()
    return DocumentRetriever(index=index, top_k=top_k, similarity_threshold=similarity_threshold)


def load_retriever(
    collection_name: Optional[str] = None, embedding_model: Optional[str] = None, top_k: int = DEFAULT_TOP_K,
) -> DocumentRetriever:
    """Convenience function to load a retriever from existing Qdrant index."""
    embedding_service = create_embedding_service(model_name=embedding_model)
    configure_global_embeddings(model_name=embedding_service.model_name)
    return create_retriever(collection_name=collection_name, embedding_model=embedding_model, top_k=top_k)
