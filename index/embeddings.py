"""
Embedding configuration and utilities for DocuLens.

This module provides embedding model configuration and utilities
for converting TextNodes into vector embeddings using Sentence
Transformers (HuggingFace) models for local, cost-free embeddings.
"""

from typing import List, Optional

from llama_index.core import Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config.logging_config import get_logger, LogTimer

logger = get_logger(__name__)


# Default embedding model - high quality, balanced size
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSIONS = 384

# Model dimensions mapping for common sentence transformer models
MODEL_DIMENSIONS = {
    "bge-small-en-v1.5": 384,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
}


def get_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> HuggingFaceEmbedding:
    """
    Get a HuggingFace/Sentence Transformer embedding model instance.
    
    Args:
        model_name: Name of the embedding model. Defaults to BAAI/bge-small-en-v1.5.
        device: Device to run model on ('cpu', 'cuda', etc.). Auto-detected if None.
        
    Returns:
        Configured HuggingFaceEmbedding instance.
    """
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    logger.info(f"Loading embedding model: {model_name}")
    
    if device:
        logger.debug(f"Using device: {device}")
        return HuggingFaceEmbedding(model_name=model_name, device=device)
    return HuggingFaceEmbedding(model_name=model_name)


def configure_global_embeddings(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> HuggingFaceEmbedding:
    """
    Configure the global LlamaIndex embedding model.
    
    This sets the embedding model in LlamaIndex's global Settings,
    which is used by default for all indexing operations.
    
    Args:
        model_name: Name of the embedding model.
        device: Device to run model on.
        
    Returns:
        The configured embedding model.
    """
    logger.info(f"Configuring global embeddings: {model_name or DEFAULT_EMBEDDING_MODEL}")
    embed_model = get_embedding_model(model_name, device)
    Settings.embed_model = embed_model
    return embed_model


def embed_nodes(
    nodes: List[TextNode],
    embed_model: Optional[HuggingFaceEmbedding] = None,
    show_progress: bool = True,
) -> List[TextNode]:
    """
    Add embeddings to a list of TextNodes.
    
    This function embeds each node's text content and stores
    the embedding vector in the node.
    
    Args:
        nodes: List of TextNodes to embed.
        embed_model: Embedding model to use. Creates default if not provided.
        show_progress: Whether to show progress during embedding.
        
    Returns:
        List of TextNodes with embeddings added.
    """
    if not nodes:
        return nodes
    
    if embed_model is None:
        embed_model = get_embedding_model()
    
    # Extract texts for batch embedding
    texts = [node.text for node in nodes]
    
    # Get embeddings in batch
    logger.info(f"Embedding {len(texts)} nodes...")
    with LogTimer(logger, f"Batch embedding of {len(texts)} texts"):
        embeddings = embed_model.get_text_embedding_batch(
            texts,
            show_progress=show_progress,
        )
    
    # Assign embeddings to nodes
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
    
    return nodes


def get_embedding_dimensions(model_name: Optional[str] = None) -> int:
    """
    Get the embedding dimensions for a model.
    
    Args:
        model_name: Name of the embedding model.
        
    Returns:
        Number of dimensions for the embedding model.
    """
    model_name = model_name or DEFAULT_EMBEDDING_MODEL
    
    return MODEL_DIMENSIONS.get(model_name, DEFAULT_EMBEDDING_DIMENSIONS)


class EmbeddingService:
    """
    Service for managing document embeddings.
    
    This class provides a convenient interface for embedding
    TextNodes using Sentence Transformer models.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model.
            device: Device to run model on ('cpu', 'cuda', etc.).
        """
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        logger.info(f"Initializing EmbeddingService with model: {self.model_name}")
        self.embed_model = get_embedding_model(self.model_name, device)
        self.dimensions = get_embedding_dimensions(self.model_name)
        logger.debug(f"Embedding dimensions: {self.dimensions}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as list of floats.
        """
        return self.embed_model.get_text_embedding(text)
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed.
            show_progress: Whether to show progress.
            
        Returns:
            List of embedding vectors.
        """
        return self.embed_model.get_text_embedding_batch(
            texts,
            show_progress=show_progress,
        )
    
    def embed_nodes(
        self,
        nodes: List[TextNode],
        show_progress: bool = True,
    ) -> List[TextNode]:
        """
        Add embeddings to TextNodes.
        
        Args:
            nodes: List of TextNodes to embed.
            show_progress: Whether to show progress.
            
        Returns:
            TextNodes with embeddings added.
        """
        return embed_nodes(nodes, self.embed_model, show_progress)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string.
        
        Note: Some embedding models use different embeddings for
        queries vs documents. This method uses the query embedding.
        
        Args:
            query: Query text to embed.
            
        Returns:
            Query embedding vector.
        """
        return self.embed_model.get_query_embedding(query)


def create_embedding_service(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> EmbeddingService:
    """
    Factory function to create an EmbeddingService.
    
    Args:
        model_name: Name of the embedding model.
        device: Device to run model on.
        
    Returns:
        Configured EmbeddingService instance.
    """
    return EmbeddingService(model_name, device)
