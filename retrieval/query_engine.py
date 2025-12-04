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
from config.logging_config import get_logger, log_execution_time, LogTimer
from retrieval.retriever import (
    DocumentRetriever,
    RetrievalResult,
    create_retriever,
    DEFAULT_TOP_K,
)
from index.indexer import Indexer, create_indexer
from index.embeddings import configure_global_embeddings, create_embedding_service

logger = get_logger(__name__)


# Default LLM settings
DEFAULT_LLM_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_TEMPERATURE = 0.1


@dataclass
class QueryResult:
    """Result from a query operation."""
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
        sources = []
        for n in self.source_nodes:
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
    
    @property
    def cited_pages(self) -> List[int]:
        return sorted({s.get("page_number", 1) for s in self.sources})


class DocuLensQueryEngine:
    """Query engine for DocuLens multi-modal RAG."""
    
    def __init__(
        self, index: VectorStoreIndex, retriever: Optional[DocumentRetriever] = None,
        llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K, response_mode: ResponseMode = ResponseMode.COMPACT,
    ):
        self.index, self.llm_model, self.temperature = index, llm_model, temperature
        self.top_k, self.response_mode = top_k, response_mode
        self.retriever = retriever or DocumentRetriever(index=index, top_k=top_k)
        self.llm = OpenAI(model=llm_model, temperature=temperature, api_key=settings.OPENAI_API_KEY)
        self._query_engine = self._create_query_engine()
        logger.info(f"DocuLensQueryEngine initialized (model={llm_model}, top_k={top_k})")
    
    def _create_query_engine(self) -> RetrieverQueryEngine:
        return RetrieverQueryEngine(
            retriever=VectorIndexRetriever(index=self.index, similarity_top_k=self.top_k),
            response_synthesizer=get_response_synthesizer(llm=self.llm, response_mode=self.response_mode),
        )
    
    @log_execution_time()
    def query(self, query_str: str, top_k: Optional[int] = None) -> QueryResult:
        """Query the index and generate a response."""
        effective_top_k = top_k or self.top_k
        logger.info(f"Processing query: '{query_str[:80]}...'" if len(query_str) > 80 else f"Processing query: '{query_str}'")
        
        with LogTimer(logger, "Retrieval"):
            retrieval_result = self.retriever.retrieve(query_str, effective_top_k)
        
        with LogTimer(logger, "Response generation"):
            response = self._query_engine.query(query_str)
        
        return QueryResult(
            query=query_str, response=str(response),
            source_nodes=list(response.source_nodes) if response.source_nodes else [],
            retrieval_result=retrieval_result,
            metadata={"llm_model": self.llm_model, "top_k": effective_top_k},
        )
    
    def retrieve_only(self, query_str: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve relevant nodes without generating a response."""
        return self.retriever.retrieve(query_str, top_k or self.top_k)
    
    def query_multimodal(self, query_str: str, text_top_k: int = 3, image_top_k: int = 2) -> QueryResult:
        """Query with balanced multimodal retrieval."""
        retrieval_result = self.retriever.retrieve_multimodal(query_str, text_top_k, image_top_k)
        response = self._query_engine.query(query_str)
        return QueryResult(
            query=query_str, response=str(response),
            source_nodes=list(response.source_nodes) if response.source_nodes else [],
            retrieval_result=retrieval_result,
            metadata={"llm_model": self.llm_model, "text_top_k": text_top_k, "image_top_k": image_top_k, "mode": "multimodal"},
        )


def create_query_engine(
    indexer: Optional[Indexer] = None, index: Optional[VectorStoreIndex] = None,
    collection_name: Optional[str] = None, embedding_model: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_TEMPERATURE, top_k: int = DEFAULT_TOP_K,
) -> DocuLensQueryEngine:
    """Factory function to create a DocuLensQueryEngine."""
    logger.info(f"Creating DocuLensQueryEngine (model={llm_model}, top_k={top_k})")
    embedding_service = create_embedding_service(model_name=embedding_model)
    configure_global_embeddings(model_name=embedding_service.model_name)
    
    if index is None:
        if indexer is None:
            indexer = create_indexer(collection_name=collection_name, embedding_model=embedding_model)
        index = indexer.load_index()
    
    return DocuLensQueryEngine(index=index, llm_model=llm_model, temperature=temperature, top_k=top_k)


def load_query_engine(
    collection_name: Optional[str] = None, embedding_model: Optional[str] = None,
    llm_model: str = DEFAULT_LLM_MODEL, top_k: int = DEFAULT_TOP_K,
) -> DocuLensQueryEngine:
    """Convenience function to load a query engine from existing Qdrant index."""
    return create_query_engine(
        collection_name=collection_name, embedding_model=embedding_model, llm_model=llm_model, top_k=top_k
    )
