"""
Qdrant vector store setup and management for DocuLens.

This module provides Qdrant client wrapper for connecting to Qdrant,
creating collections, and managing vector operations.
"""

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from config.settings import settings
from index.embeddings import get_embedding_dimensions, DEFAULT_EMBEDDING_MODEL


# Default collection settings
DEFAULT_COLLECTION_NAME = "doculens"
DEFAULT_DISTANCE_METRIC = qdrant_models.Distance.COSINE


class QdrantStore:
    """
    Wrapper for Qdrant vector store operations.
    
    This class provides methods for connecting to Qdrant,
    creating collections, and building LlamaIndex vector store.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize Qdrant store connection.
        
        Args:
            url: Qdrant server URL. Defaults to settings.
            api_key: Qdrant API key (for cloud). Defaults to settings.
            collection_name: Name of the collection. Defaults to settings.
            embedding_model: Embedding model name for dimensions lookup.
        """
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name or settings.QDRANT_COLLECTION or DEFAULT_COLLECTION_NAME
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.dimensions = get_embedding_dimensions(self.embedding_model)
        
        # Initialize Qdrant client
        self.client = self._create_client()
        
        # LlamaIndex vector store (created lazily)
        self._vector_store: Optional[QdrantVectorStore] = None
    
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client."""
        return QdrantClient(url=self.url)
    
    def collection_exists(self) -> bool:
        """Check if the collection already exists."""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)
    
    def create_collection(
        self,
        recreate: bool = False,
    ) -> None:
        """
        Create a Qdrant collection for storing embeddings.
        
        Args:
            recreate: If True, delete existing collection first.
        """
        if recreate and self.collection_exists():
            print(f"  Deleting existing collection: {self.collection_name}")
            self.client.delete_collection(self.collection_name)
        
        if not self.collection_exists():
            print(f"  Creating collection: {self.collection_name} (dimensions={self.dimensions})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.dimensions,
                    distance=DEFAULT_DISTANCE_METRIC,
                ),
            )
        else:
            print(f"  Collection already exists: {self.collection_name}")
    
    def get_vector_store(self) -> QdrantVectorStore:
        """
        Get LlamaIndex QdrantVectorStore instance.
        
        Returns:
            Configured QdrantVectorStore for use with LlamaIndex.
        """
        if self._vector_store is None:
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
            )
        return self._vector_store
    
    def get_storage_context(self) -> StorageContext:
        """
        Get LlamaIndex StorageContext with Qdrant.
        
        Returns:
            StorageContext configured with Qdrant vector store.
        """
        vector_store = self.get_vector_store()
        return StorageContext.from_defaults(vector_store=vector_store)  # type: ignore
    
    def get_collection_info(self) -> Optional[dict]:
        """
        Get information about the collection.
        
        Returns:
            Collection info dict or None if collection doesn't exist.
        """
        if not self.collection_exists():
            return None
        
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.points_count,  # points_count is the vector count
            "points_count": info.points_count,
            "status": info.status.value,
            "dimensions": self.dimensions,
        }
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if deleted, False if didn't exist.
        """
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)
            self._vector_store = None
            return True
        return False
    
    def count_vectors(self) -> int:
        """Get the number of vectors in the collection."""
        if not self.collection_exists():
            return 0
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0


def create_qdrant_store(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> QdrantStore:
    """
    Factory function to create a QdrantStore.
    
    Args:
        url: Qdrant server URL.
        api_key: Qdrant API key (for cloud).
        collection_name: Name of the collection.
        embedding_model: Embedding model name for dimensions lookup.
        
    Returns:
        Configured QdrantStore instance.
    """
    return QdrantStore(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )


def get_vector_store_index(
    qdrant_store: QdrantStore,
    nodes: Optional[List[TextNode]] = None,
    embed_model=None,
) -> VectorStoreIndex:
    """
    Create or load a VectorStoreIndex using Qdrant.
    
    If nodes are provided, they will be added to the index.
    If no nodes, loads existing index from the collection.
    
    Args:
        qdrant_store: QdrantStore instance.
        nodes: Optional list of nodes to index.
        embed_model: Embedding model to use.
        
    Returns:
        VectorStoreIndex connected to Qdrant.
    """
    storage_context = qdrant_store.get_storage_context()
    
    if nodes:
        # Create new index with nodes
        return VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    else:
        # Load existing index from Qdrant
        return VectorStoreIndex.from_vector_store(
            vector_store=qdrant_store.get_vector_store(),  # type: ignore
            embed_model=embed_model,
        )
