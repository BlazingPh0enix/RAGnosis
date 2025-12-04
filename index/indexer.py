"""
Main indexing orchestrator for DocuLens.

This module provides the Indexer class that coordinates chunking,
embedding, and indexing of parsed documents into Qdrant.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode

from config.settings import settings
from index.chunking import DocumentChunker, create_chunker
from index.embeddings import (
    EmbeddingService,
    create_embedding_service,
    configure_global_embeddings,
)
from index.qdrant_store import QdrantStore, create_qdrant_store


class Indexer:
    """
    Orchestrates the indexing process for parsed documents.
    
    This class coordinates:
    1. Loading parsed documents from disk
    2. Chunking content into nodes
    3. Embedding nodes
    4. Storing in Qdrant
    """
    
    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[DocumentChunker] = None,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize the indexer.
        
        Args:
            qdrant_store: QdrantStore instance. Creates default if not provided.
            embedding_service: EmbeddingService instance. Creates default if not provided.
            chunker: DocumentChunker instance. Creates default if not provided.
            data_dir: Base data directory. Defaults to settings.
        """
        self.qdrant_store = qdrant_store or create_qdrant_store()
        self.embedding_service = embedding_service or create_embedding_service()
        self.chunker = chunker or create_chunker()
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        
        # Configure global embeddings for LlamaIndex
        configure_global_embeddings(model_name=self.embedding_service.model_name)
        
        # Track indexed documents
        self._indexed_documents: List[str] = []
    
    @property
    def parsed_dir(self) -> Path:
        """Directory containing parsed document data."""
        return self.data_dir / "parsed"
    
    @property
    def summaries_dir(self) -> Path:
        """Directory containing image summaries."""
        return self.data_dir / "summaries"
    
    def load_parsed_document(self, document_name: str) -> dict:
        """
        Load a parsed document from disk.
        
        Args:
            document_name: Name of the document (without extension).
            
        Returns:
            Dictionary containing markdown content and metadata.
        """
        # Look for markdown file
        md_path = self.parsed_dir / f"{document_name}.md"
        json_path = self.parsed_dir / f"{document_name}.json"
        
        result = {
            "document_name": document_name,
            "markdown_content": "",
            "metadata": {},
        }
        
        # Load markdown content
        if md_path.exists():
            result["markdown_content"] = md_path.read_text(encoding="utf-8")
        
        # Load JSON metadata if available
        if json_path.exists():
            result["metadata"] = json.loads(json_path.read_text(encoding="utf-8"))
        
        return result
    
    def load_image_summaries(self, document_name: str) -> List[dict]:
        """
        Load image summaries for a document.
        
        Args:
            document_name: Name of the document.
            
        Returns:
            List of image summary dictionaries.
        """
        summaries_path = self.summaries_dir / f"{document_name}_summaries.json"
        
        if not summaries_path.exists():
            return []
        
        summaries_data = json.loads(summaries_path.read_text(encoding="utf-8"))
        
        # Handle both list and dict formats
        if isinstance(summaries_data, list):
            return summaries_data
        elif isinstance(summaries_data, dict) and "summaries" in summaries_data:
            return summaries_data["summaries"]
        
        return []
    
    def list_parsed_documents(self) -> List[str]:
        """
        List all available parsed documents.
        
        Returns:
            List of document names (without extension).
        """
        if not self.parsed_dir.exists():
            return []
        
        documents = set()
        for file_path in self.parsed_dir.glob("*.md"):
            documents.add(file_path.stem)
        
        return sorted(documents)
    
    def chunk_document(
        self,
        document_name: str,
        markdown_content: str,
        image_summaries: List[dict],
    ) -> List[TextNode]:
        """
        Chunk a document into TextNodes.
        
        Args:
            document_name: Name of the source document.
            markdown_content: Markdown content to chunk.
            image_summaries: List of image summary dictionaries.
            
        Returns:
            List of TextNodes.
        """
        return self.chunker.chunk_document(
            markdown_content=markdown_content,
            image_summaries=image_summaries,
            source_document=document_name,
        )
    
    def index_document(
        self,
        document_name: str,
        show_progress: bool = True,
    ) -> List[TextNode]:
        """
        Index a single parsed document.
        
        Args:
            document_name: Name of the document to index.
            show_progress: Whether to show progress.
            
        Returns:
            List of indexed TextNodes.
        """
        print(f"Indexing document: {document_name}")
        
        # Load parsed data
        parsed = self.load_parsed_document(document_name)
        image_summaries = self.load_image_summaries(document_name)
        
        if not parsed["markdown_content"]:
            print(f"  Warning: No markdown content found for {document_name}")
            return []
        
        # Chunk document
        print(f"  Chunking content...")
        nodes = self.chunk_document(
            document_name=document_name,
            markdown_content=parsed["markdown_content"],
            image_summaries=image_summaries,
        )
        print(f"  Created {len(nodes)} nodes")
        
        # Embed nodes
        print(f"  Embedding nodes...")
        nodes = self.embedding_service.embed_nodes(nodes, show_progress=show_progress)
        
        self._indexed_documents.append(document_name)
        return nodes
    
    def index_all_documents(
        self,
        document_names: Optional[List[str]] = None,
        recreate_collection: bool = False,
        show_progress: bool = True,
    ) -> Optional[VectorStoreIndex]:
        """
        Index all (or specified) parsed documents.
        
        Args:
            document_names: List of document names to index. If None, indexes all.
            recreate_collection: Whether to recreate the Qdrant collection.
            show_progress: Whether to show progress.
            
        Returns:
            VectorStoreIndex containing all indexed documents.
        """
        # Get documents to index
        if document_names is None:
            document_names = self.list_parsed_documents()
        
        if not document_names:
            print("No documents found to index.")
            return None
        
        print(f"Indexing {len(document_names)} documents...")
        
        # Create/recreate collection
        self.qdrant_store.create_collection(recreate=recreate_collection)
        
        # Index all documents
        all_nodes = []
        for doc_name in document_names:
            nodes = self.index_document(doc_name, show_progress=show_progress)
            all_nodes.extend(nodes)
        
        print(f"\nTotal nodes: {len(all_nodes)}")
        
        # Build vector store index
        print("Building vector store index...")
        storage_context = self.qdrant_store.get_storage_context()
        
        index = VectorStoreIndex(
            nodes=all_nodes,
            storage_context=storage_context,
            embed_model=self.embedding_service.embed_model,
            show_progress=show_progress,
        )
        
        print(f"Index built successfully. Vectors in Qdrant: {self.qdrant_store.count_vectors()}")
        
        return index
    
    def load_index(self) -> VectorStoreIndex:
        """
        Load existing index from Qdrant.
        
        Returns:
            VectorStoreIndex loaded from Qdrant.
        """
        if not self.qdrant_store.collection_exists():
            raise ValueError(
                f"Collection '{self.qdrant_store.collection_name}' does not exist. "
                "Run index_all_documents() first."
            )
        
        return VectorStoreIndex.from_vector_store(
            vector_store=self.qdrant_store.get_vector_store(),  # type: ignore
            embed_model=self.embedding_service.embed_model,
        )
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary with index statistics.
        """
        collection_info = self.qdrant_store.get_collection_info()
        
        return {
            "collection": collection_info,
            "indexed_documents": self._indexed_documents,
            "embedding_model": self.embedding_service.model_name,
            "embedding_dimensions": self.embedding_service.dimensions,
        }


def create_indexer(
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    data_dir: Optional[str] = None,
) -> Indexer:
    """
    Factory function to create an Indexer with custom configuration.
    
    Args:
        collection_name: Qdrant collection name.
        embedding_model: Embedding model to use.
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Overlap between chunks.
        data_dir: Base data directory.
        
    Returns:
        Configured Indexer instance.
    """
    qdrant_store = create_qdrant_store(
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    
    embedding_service = create_embedding_service(model_name=embedding_model)
    
    chunker = create_chunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return Indexer(
        qdrant_store=qdrant_store,
        embedding_service=embedding_service,
        chunker=chunker,
        data_dir=data_dir,
    )
