"""
Query engine wrapper for DocuLens.

This module provides a query engine that wraps LlamaIndex's query
capabilities with custom retrieval and response configuration.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.llms.openai import OpenAI

from config.settings import settings
from retrieval.retriever import (
    DocumentRetriever,
    RetrievalResult,
    create_retriever,
    DEFAULT_TOP_K,
)
from index.indexer import Indexer, create_indexer
from index.embeddings import configure_global_embeddings, create_embedding_service


# Default LLM settings
DEFAULT_LLM_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_TEMPERATURE = 0.1


@dataclass
class QueryResult:
    """
    Result from a query operation.
    
    Attributes:
        query: The original query string.
        response: The generated response text.
        source_nodes: List of source nodes used to generate the response.
        retrieval_result: The underlying RetrievalResult.
        metadata: Additional metadata about the query.
    """
    query: str
    response: str
    source_nodes: List[NodeWithScore]
    retrieval_result: Optional[RetrievalResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def sources(self) -> List[Dict[str, Any]]:
        """Get formatted source information."""
        if self.retrieval_result:
            return self.retrieval_result.sources
        
        # Fallback: extract from source_nodes directly
        sources = []
        for node_with_score in self.source_nodes:
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
    
    @property
    def cited_pages(self) -> List[int]:
        """Get list of unique page numbers from sources."""
        pages = set()
        for source in self.sources:
            pages.add(source.get("page_number", 1))
        return sorted(pages)


class DocuLensQueryEngine:
    """
    Query engine for DocuLens multi-modal RAG.
    
    This engine combines retrieval and response generation,
    providing a simple interface for asking questions about
    indexed documents.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        retriever: Optional[DocumentRetriever] = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        response_mode: ResponseMode = ResponseMode.COMPACT,
    ):
        """
        Initialize the query engine.
        
        Args:
            index: VectorStoreIndex to query.
            retriever: Optional custom DocumentRetriever.
            llm_model: LLM model for response generation.
            temperature: LLM temperature setting.
            top_k: Number of nodes to retrieve.
            response_mode: LlamaIndex response synthesis mode.
        """
        self.index = index
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        self.response_mode = response_mode
        
        # Create retriever if not provided
        if retriever is None:
            self.retriever = DocumentRetriever(index=index, top_k=top_k)
        else:
            self.retriever = retriever
        
        # Create LLM
        self.llm = self._create_llm()
        
        # Create underlying query engine
        self._query_engine = self._create_query_engine()
    
    def _create_llm(self) -> OpenAI:
        """Create the LLM instance."""
        return OpenAI(
            model=self.llm_model,
            temperature=self.temperature,
            api_key=settings.OPENAI_API_KEY,
        )
    
    def _create_query_engine(self) -> RetrieverQueryEngine:
        """Create the underlying LlamaIndex query engine."""
        # Create vector retriever
        vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=self.top_k,
        )
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            response_mode=self.response_mode,
        )
        
        return RetrieverQueryEngine(
            retriever=vector_retriever,
            response_synthesizer=response_synthesizer,
        )
    
    def query(
        self,
        query_str: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """
        Query the index and generate a response.
        
        Args:
            query_str: The question to ask.
            top_k: Override default top_k for this query.
            
        Returns:
            QueryResult with response and sources.
        """
        # First, retrieve nodes using our custom retriever
        retrieval_result = self.retriever.retrieve(
            query=query_str,
            top_k=top_k or self.top_k,
        )
        
        # Use the LlamaIndex query engine for response generation
        response = self._query_engine.query(query_str)
        
        return QueryResult(
            query=query_str,
            response=str(response),
            source_nodes=list(response.source_nodes) if response.source_nodes else [],
            retrieval_result=retrieval_result,
            metadata={
                "llm_model": self.llm_model,
                "top_k": top_k or self.top_k,
            },
        )
    
    def retrieve_only(
        self,
        query_str: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve relevant nodes without generating a response.
        
        Args:
            query_str: The query string.
            top_k: Number of results to retrieve.
            
        Returns:
            RetrievalResult with retrieved nodes.
        """
        return self.retriever.retrieve(
            query=query_str,
            top_k=top_k or self.top_k,
        )
    
    def query_multimodal(
        self,
        query_str: str,
        text_top_k: int = 3,
        image_top_k: int = 2,
    ) -> QueryResult:
        """
        Query with balanced multimodal retrieval.
        
        This ensures both text and image content are considered.
        
        Args:
            query_str: The question to ask.
            text_top_k: Number of text/table nodes.
            image_top_k: Number of image nodes.
            
        Returns:
            QueryResult with multimodal context.
        """
        # Get multimodal retrieval result
        retrieval_result = self.retriever.retrieve_multimodal(
            query=query_str,
            text_top_k=text_top_k,
            image_top_k=image_top_k,
        )
        
        # Generate response using query engine
        response = self._query_engine.query(query_str)
        
        return QueryResult(
            query=query_str,
            response=str(response),
            source_nodes=list(response.source_nodes) if response.source_nodes else [],
            retrieval_result=retrieval_result,
            metadata={
                "llm_model": self.llm_model,
                "text_top_k": text_top_k,
                "image_top_k": image_top_k,
                "mode": "multimodal",
            },
        )


def create_query_engine(
    indexer: Optional[Indexer] = None,
    index: Optional[VectorStoreIndex] = None,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
) -> DocuLensQueryEngine:
    """
    Factory function to create a DocuLensQueryEngine.
    
    Args:
        indexer: Optional Indexer instance.
        index: Optional VectorStoreIndex.
        collection_name: Qdrant collection name.
        embedding_model: Embedding model name.
        llm_model: LLM model for response generation.
        temperature: LLM temperature.
        top_k: Number of nodes to retrieve.
        
    Returns:
        Configured DocuLensQueryEngine.
    """
    # Configure embeddings
    embedding_service = create_embedding_service(model_name=embedding_model)
    configure_global_embeddings(model_name=embedding_service.model_name)
    
    # Get or create index
    if index is None:
        if indexer is None:
            indexer = create_indexer(
                collection_name=collection_name,
                embedding_model=embedding_model,
            )
        index = indexer.load_index()
    
    return DocuLensQueryEngine(
        index=index,
        llm_model=llm_model,
        temperature=temperature,
        top_k=top_k,
    )


def load_query_engine(
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL,
    top_k: int = DEFAULT_TOP_K,
) -> DocuLensQueryEngine:
    """
    Convenience function to load a query engine from existing Qdrant index.
    
    Args:
        collection_name: Qdrant collection name.
        embedding_model: Embedding model name.
        llm_model: LLM model for response generation.
        top_k: Number of nodes to retrieve.
        
    Returns:
        DocuLensQueryEngine ready to use.
    """
    return create_query_engine(
        collection_name=collection_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        top_k=top_k,
    )
