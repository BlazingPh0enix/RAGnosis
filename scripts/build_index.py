#!/usr/bin/env python
"""
CLI script to build the vector index from parsed documents.

This script:
1. Loads parsed documents from data/parsed/
2. Loads image summaries from data/summaries/
3. Chunks content into TextNodes
4. Embeds nodes using Sentence Transformers
5. Stores embeddings in Qdrant

Usage:
    python scripts/build_index.py                    # Index all documents
    python scripts/build_index.py doc1 doc2         # Index specific documents
    python scripts/build_index.py --recreate        # Recreate collection from scratch
    python scripts/build_index.py --list            # List available documents
    python scripts/build_index.py --stats           # Show index statistics
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from index.indexer import create_indexer, Indexer
from index.qdrant_store import create_qdrant_store


def list_documents(indexer: Indexer) -> None:
    """List all available parsed documents."""
    documents = indexer.list_parsed_documents()
    
    if not documents:
        print("No parsed documents found in data/parsed/")
        print(f"  Looking in: {indexer.parsed_dir}")
        return
    
    print(f"Found {len(documents)} parsed documents:")
    for doc in documents:
        # Check if summaries exist
        summaries = indexer.load_image_summaries(doc)
        summary_info = f" ({len(summaries)} image summaries)" if summaries else ""
        print(f"  - {doc}{summary_info}")


def show_stats(indexer: Indexer) -> None:
    """Show index statistics."""
    stats = indexer.get_index_stats()
    
    print("Index Statistics:")
    print(f"  Embedding Model: {stats['embedding_model']}")
    print(f"  Embedding Dimensions: {stats['embedding_dimensions']}")
    
    collection = stats.get("collection")
    if collection:
        print(f"\nQdrant Collection: {collection['name']}")
        print(f"  Status: {collection['status']}")
        print(f"  Vectors: {collection['vectors_count']}")
        print(f"  Points: {collection['points_count']}")
    else:
        print("\nNo Qdrant collection found.")
    
    if stats["indexed_documents"]:
        print(f"\nDocuments indexed this session: {len(stats['indexed_documents'])}")
        for doc in stats["indexed_documents"]:
            print(f"  - {doc}")


def build_index(
    document_names: Optional[list] = None,
    recreate: bool = False,
    collection_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> None:
    """Build the vector index."""
    print("=" * 60)
    print("DocuLens Index Builder")
    print("=" * 60)
    
    # Create indexer
    indexer = create_indexer(
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    print(f"\nConfiguration:")
    print(f"  Collection: {indexer.qdrant_store.collection_name}")
    print(f"  Embedding Model: {indexer.embedding_service.model_name}")
    print(f"  Dimensions: {indexer.embedding_service.dimensions}")
    print(f"  Chunk Size: {indexer.chunker.chunk_size}")
    print(f"  Chunk Overlap: {indexer.chunker.chunk_overlap}")
    print(f"  Data Directory: {indexer.data_dir}")
    print()
    
    # Check for documents
    available_docs = indexer.list_parsed_documents()
    if not available_docs:
        print("Error: No parsed documents found.")
        print(f"  Looking in: {indexer.parsed_dir}")
        print("\nRun the ingestion script first:")
        print("  python scripts/ingest_documents.py <pdf_path>")
        return
    
    # Filter to specified documents
    if document_names:
        invalid_docs = [d for d in document_names if d not in available_docs]
        if invalid_docs:
            print(f"Warning: Documents not found: {invalid_docs}")
        document_names = [d for d in document_names if d in available_docs]
        
        if not document_names:
            print("Error: No valid documents to index.")
            return
    else:
        document_names = available_docs
    
    print(f"Documents to index: {len(document_names)}")
    for doc in document_names:
        print(f"  - {doc}")
    print()
    
    # Build index
    try:
        index = indexer.index_all_documents(
            document_names=document_names,
            recreate_collection=recreate,
            show_progress=True,
        )
        
        if index:
            print("\n" + "=" * 60)
            print("Index built successfully!")
            show_stats(indexer)
        
    except Exception as e:
        print(f"\nError building index: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Build vector index from parsed documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "documents",
        nargs="*",
        help="Specific document names to index (without extension). "
             "If not provided, indexes all documents.",
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_docs",
        help="List available parsed documents and exit",
    )
    
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show index statistics and exit",
    )
    
    parser.add_argument(
        "--recreate", "-r",
        action="store_true",
        help="Recreate Qdrant collection (deletes existing data)",
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=None,
        help="Qdrant collection name (default: from settings)",
    )
    
    parser.add_argument(
        "--embedding-model", "-e",
        type=str,
        default=None,
        help="Sentence Transformer model name (default: all-MiniLM-L6-v2)",
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size for text splitting (default: 512)",
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)",
    )
    
    args = parser.parse_args()
    
    # Create indexer for list/stats operations
    indexer = create_indexer(
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    
    if args.list_docs:
        list_documents(indexer)
        return
    
    if args.stats:
        show_stats(indexer)
        return
    
    # Build index
    build_index(
        document_names=args.documents if args.documents else None,
        recreate=args.recreate,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


if __name__ == "__main__":
    main()
