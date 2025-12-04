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
    """
    Result from a retrieval operation.
    
    Attributes:
        query: The original query string.
        nodes: List of retrieved nodes with scores.
        text_nodes: Filtered list of text/table nodes.
        image_nodes: Filtered list of image summary nodes.
    """
    query: str
    nodes: List[NodeWithScore]
    
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
    def sources(self) -> List[Dict[str, Any]]:
        """
        Get source information for all retrieved nodes.
        
        Returns:
            List of source dictionaries with page numbers and content types.
        """
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
            
            # Add image name for image summaries
            if metadata.get("content_type") == "image_summary":
                source["image_name"] = metadata.get("image_name", "")
            
            sources.append(source)
        
        return sources
    
    def get_context_for_llm(self) -> str:
        """
        Format retrieved nodes as context for LLM.
        
        Returns:
            Formatted context string with page citations.
        """
        context_parts = []
        
        for node_with_score in self.nodes:
            node = node_with_score.node
            metadata = node.metadata
            page_num = metadata.get("page_number", 1)
            content_type = metadata.get("content_type", "text")
            
            # Add content type indicator
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


class DocumentRetriever:
    """
    Hybrid retriever for multi-modal document content.
    
    This retriever supports fetching text, table, and image summary
    nodes from the Qdrant vector index.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        """
        Initialize the document retriever.
        
        Args:
            index: VectorStoreIndex to retrieve from.
            top_k: Number of top results to retrieve.
            similarity_threshold: Minimum similarity score (0.0 to 1.0).
        """
        self.index = index
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Initializing DocumentRetriever (top_k={top_k}, threshold={similarity_threshold})")
        
        # Create underlying retriever
        self._retriever = self._create_retriever()
    
    def _create_retriever(self) -> VectorIndexRetriever:
        """Create the underlying LlamaIndex retriever."""
        return VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )
    
    @log_execution_time()
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_content_types: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query: Query string.
            top_k: Override default top_k for this query.
            filter_content_types: Optional list of content types to filter
                                  (e.g., ["text", "table", "image_summary"]).
        
        Returns:
            RetrievalResult containing retrieved nodes.
        """
        effective_top_k = top_k or self.top_k
        logger.debug(f"Retrieving for query: '{query[:50]}...' (top_k={effective_top_k})")
        
        # Update top_k if specified
        if top_k is not None:
            self._retriever.similarity_top_k = top_k
        else:
            self._retriever.similarity_top_k = self.top_k
        
        # Retrieve nodes
        with LogTimer(logger, "Vector retrieval"):
            nodes_with_scores = self._retriever.retrieve(query)
        
        logger.debug(f"Initial retrieval returned {len(nodes_with_scores)} nodes")
        
        # Filter by similarity threshold
        if self.similarity_threshold > 0:
            nodes_with_scores = [
                n for n in nodes_with_scores
                if n.score is not None and n.score >= self.similarity_threshold
            ]
            logger.debug(f"After threshold filter: {len(nodes_with_scores)} nodes")
        
        # Filter by content type if specified
        if filter_content_types:
            nodes_with_scores = [
                n for n in nodes_with_scores
                if n.node.metadata.get("content_type") in filter_content_types
            ]
            logger.debug(f"After content type filter ({filter_content_types}): {len(nodes_with_scores)} nodes")
        
        logger.info(f"Retrieved {len(nodes_with_scores)} nodes for query")
        return RetrievalResult(query=query, nodes=nodes_with_scores)
    
    def retrieve_text_only(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve only text and table nodes.
        
        Args:
            query: Query string.
            top_k: Override default top_k.
            
        Returns:
            RetrievalResult with only text/table nodes.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_content_types=["text", "table"],
        )
    
    def retrieve_images_only(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve only image summary nodes.
        
        Args:
            query: Query string.
            top_k: Override default top_k.
            
        Returns:
            RetrievalResult with only image summary nodes.
        """
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_content_types=["image_summary"],
        )
    
    def retrieve_multimodal(
        self,
        query: str,
        text_top_k: int = 3,
        image_top_k: int = 2,
    ) -> RetrievalResult:
        """
        Retrieve a balanced mix of text and image nodes.
        
        This method ensures both text and image content are represented
        in the results by retrieving separately and merging.
        
        Args:
            query: Query string.
            text_top_k: Number of text/table nodes to retrieve.
            image_top_k: Number of image summary nodes to retrieve.
            
        Returns:
            RetrievalResult with mixed content types.
        """
        # Retrieve text nodes
        text_result = self.retrieve(
            query=query,
            top_k=text_top_k * 2,  # Retrieve more, then filter
            filter_content_types=["text", "table"],
        )
        text_nodes = text_result.nodes[:text_top_k]
        
        # Retrieve image nodes
        image_result = self.retrieve(
            query=query,
            top_k=image_top_k * 2,
            filter_content_types=["image_summary"],
        )
        image_nodes = image_result.nodes[:image_top_k]
        
        # Merge and sort by score
        all_nodes = text_nodes + image_nodes
        all_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        return RetrievalResult(query=query, nodes=all_nodes)


def create_retriever(
    indexer: Optional[Indexer] = None,
    index: Optional[VectorStoreIndex] = None,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> DocumentRetriever:
    """
    Factory function to create a DocumentRetriever.
    
    Either provide an existing index or indexer, or the function will
    create a new indexer and load the index from Qdrant.
    
    Args:
        indexer: Optional Indexer instance to use.
        index: Optional VectorStoreIndex to use directly.
        top_k: Number of results to retrieve.
        similarity_threshold: Minimum similarity score.
        collection_name: Qdrant collection name (if creating new indexer).
        embedding_model: Embedding model name (if creating new indexer).
        
    Returns:
        Configured DocumentRetriever instance.
    """
    logger.info(f"Creating DocumentRetriever (top_k={top_k})")
    
    if index is None:
        if indexer is None:
            # Create new indexer and load index
            logger.debug("Creating new indexer to load index")
            indexer = create_indexer(
                collection_name=collection_name,
                embedding_model=embedding_model,
            )
        index = indexer.load_index()
    
    return DocumentRetriever(
        index=index,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def load_retriever(
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
) -> DocumentRetriever:
    """
    Convenience function to load a retriever from an existing Qdrant index.
    
    Args:
        collection_name: Qdrant collection name.
        embedding_model: Embedding model name.
        top_k: Number of results to retrieve.
        
    Returns:
        DocumentRetriever loaded from Qdrant.
    """
    # Configure global embeddings
    embedding_service = create_embedding_service(model_name=embedding_model)
    configure_global_embeddings(model_name=embedding_service.model_name)
    
    return create_retriever(
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=top_k,
    )
