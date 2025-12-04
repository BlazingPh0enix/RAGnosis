"""
Chunking strategies for text and table content.

This module provides chunking functionality for parsed documents,
converting markdown content into LlamaIndex TextNodes with proper
metadata for citation tracking.
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


# Default chunking parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Regex patterns for detecting content types
PAGE_MARKER_PATTERN = re.compile(r'\[Page\s+(\d+)\]', re.IGNORECASE)
TABLE_START_PATTERN = re.compile(r'<table[^>]*>', re.IGNORECASE)
TABLE_END_PATTERN = re.compile(r'</table>', re.IGNORECASE)
HTML_TABLE_PATTERN = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    page_number: int
    content_type: str  # "text", "table", "image_summary"
    source_document: str
    chunk_index: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for node metadata."""
        return {
            "page_number": self.page_number,
            "content_type": self.content_type,
            "source_document": self.source_document,
            "chunk_index": self.chunk_index,
        }


def extract_page_number(text: str, default: int = 1) -> int:
    """
    Extract page number from text containing [Page X] markers.
    
    Args:
        text: Text that may contain page markers.
        default: Default page number if no marker found.
        
    Returns:
        Extracted page number or default.
    """
    match = PAGE_MARKER_PATTERN.search(text)
    if match:
        return int(match.group(1))
    return default


def split_by_pages(markdown_content: str) -> List[Tuple[int, str]]:
    """
    Split markdown content by page markers.
    
    Args:
        markdown_content: Full markdown content with [Page X] markers.
        
    Returns:
        List of (page_number, content) tuples.
    """
    # Split by page markers while keeping the marker
    parts = PAGE_MARKER_PATTERN.split(markdown_content)
    
    pages = []
    current_page = 1
    
    # Parts alternate: [text_before, page_num, text_after, page_num, text_after, ...]
    for i, part in enumerate(parts):
        if i == 0:
            # Text before first page marker (if any)
            if part.strip():
                pages.append((current_page, part.strip()))
        elif i % 2 == 1:
            # This is a page number
            current_page = int(part)
        else:
            # This is content after a page marker
            if part.strip():
                pages.append((current_page, part.strip()))
    
    # If no page markers found, return entire content as page 1
    if not pages and markdown_content.strip():
        pages.append((1, markdown_content.strip()))
    
    return pages


def extract_tables(text: str) -> Tuple[List[str], str]:
    """
    Extract HTML tables from text, returning tables and remaining text.
    
    Args:
        text: Text containing potential HTML tables.
        
    Returns:
        Tuple of (list of table HTML strings, text with tables removed).
    """
    tables = HTML_TABLE_PATTERN.findall(text)
    remaining_text = HTML_TABLE_PATTERN.sub('', text)
    return tables, remaining_text.strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Chunk text using LlamaIndex SentenceSplitter.
    
    Args:
        text: Text to chunk.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks.
        
    Returns:
        List of text chunks.
    """
    if not text.strip():
        return []
    
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # SentenceSplitter.split_text returns list of strings
    chunks = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


def create_text_nodes(
    markdown_content: str,
    source_document: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[TextNode]:
    """
    Create TextNodes from markdown content with proper metadata.
    
    This function:
    1. Splits content by page markers
    2. Extracts tables and preserves them as single nodes
    3. Chunks remaining text using SentenceSplitter
    4. Adds metadata for citation tracking
    
    Args:
        markdown_content: Full markdown content from parsed document.
        source_document: Name of the source document.
        chunk_size: Maximum chunk size for text.
        chunk_overlap: Overlap between text chunks.
        
    Returns:
        List of TextNodes with metadata.
    """
    logger.debug(f"Creating text nodes for {source_document} (chunk_size={chunk_size}, overlap={chunk_overlap})")
    nodes = []
    node_index = 0
    
    # Split by pages
    pages = split_by_pages(markdown_content)
    
    for page_number, page_content in pages:
        # Extract tables from this page
        tables, remaining_text = extract_tables(page_content)
        
        # Create nodes for tables (preserve as single nodes)
        for table_html in tables:
            if table_html.strip():
                metadata = ChunkMetadata(
                    page_number=page_number,
                    content_type="table",
                    source_document=source_document,
                    chunk_index=node_index,
                )
                
                node = TextNode(
                    text=table_html,
                    metadata=metadata.to_dict(),
                    excluded_embed_metadata_keys=["chunk_index"],
                    excluded_llm_metadata_keys=["chunk_index"],
                )
                nodes.append(node)
                node_index += 1
        
        # Chunk remaining text
        if remaining_text:
            text_chunks = chunk_text(remaining_text, chunk_size, chunk_overlap)
            
            for chunk in text_chunks:
                metadata = ChunkMetadata(
                    page_number=page_number,
                    content_type="text",
                    source_document=source_document,
                    chunk_index=node_index,
                )
                
                node = TextNode(
                    text=chunk,
                    metadata=metadata.to_dict(),
                    excluded_embed_metadata_keys=["chunk_index"],
                    excluded_llm_metadata_keys=["chunk_index"],
                )
                nodes.append(node)
                node_index += 1
    
    logger.debug(f"{source_document}: Created {len(nodes)} text/table nodes")
    return nodes


def create_image_summary_nodes(
    image_summaries: List[dict],
    source_document: str,
) -> List[TextNode]:
    """
    Create TextNodes from image summaries.
    
    Image summaries are stored as text nodes so they can be searched
    alongside regular text content.
    
    Args:
        image_summaries: List of image summary dicts with keys:
            - summary: str (the description text)
            - page_number: int
            - image_name: str
        source_document: Name of the source document.
        
    Returns:
        List of TextNodes for image summaries.
    """
    logger.debug(f"{source_document}: Creating nodes for {len(image_summaries)} image summaries")
    nodes = []
    
    for i, summary in enumerate(image_summaries):
        summary_text = summary.get("summary", "")
        if not summary_text.strip():
            continue
        
        page_number = summary.get("page_number", 1)
        image_name = summary.get("image_name", f"image_{i}")
        
        # Create metadata
        metadata = {
            "page_number": page_number,
            "content_type": "image_summary",
            "source_document": source_document,
            "image_name": image_name,
            "chunk_index": i,
        }
        
        # Prefix the summary with context for better retrieval
        prefixed_text = f"[Image: {image_name}] {summary_text}"
        
        node = TextNode(
            text=prefixed_text,
            metadata=metadata,
            excluded_embed_metadata_keys=["chunk_index"],
            excluded_llm_metadata_keys=["chunk_index"],
        )
        nodes.append(node)
    
    return nodes


class DocumentChunker:
    """
    Chunks parsed documents into TextNodes for indexing.
    
    This class provides a convenient interface for converting
    ParsedDocument objects into lists of TextNodes with proper
    metadata for RAG retrieval and citation.
    """
    
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum chunk size for text content.
            chunk_overlap: Overlap between text chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(
        self,
        markdown_content: str,
        image_summaries: List[dict],
        source_document: str,
    ) -> List[TextNode]:
        """
        Chunk a parsed document into TextNodes.
        
        Args:
            markdown_content: Markdown content from the document.
            image_summaries: List of image summary dictionaries.
            source_document: Name of the source document.
            
        Returns:
            List of TextNodes ready for embedding and indexing.
        """
        logger.info(f"Chunking document: {source_document}")
        
        # Create text/table nodes
        text_nodes = create_text_nodes(
            markdown_content=markdown_content,
            source_document=source_document,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        
        # Create image summary nodes
        image_nodes = create_image_summary_nodes(
            image_summaries=image_summaries,
            source_document=source_document,
        )
        
        # Combine all nodes
        all_nodes = text_nodes + image_nodes
        
        logger.info(f"{source_document}: Created {len(all_nodes)} total nodes ({len(text_nodes)} text, {len(image_nodes)} image)")
        return all_nodes
    
    def chunk_from_parsed_document(self, parsed_doc) -> List[TextNode]:
        """
        Chunk a ParsedDocument object into TextNodes.
        
        Args:
            parsed_doc: ParsedDocument from the ingestion pipeline.
            
        Returns:
            List of TextNodes ready for embedding and indexing.
        """
        # Convert ImageSummary objects to dicts if needed
        image_summaries = []
        for summary in parsed_doc.image_summaries:
            if hasattr(summary, 'to_dict'):
                image_summaries.append(summary.to_dict())
            elif isinstance(summary, dict):
                image_summaries.append(summary)
            else:
                image_summaries.append({
                    "summary": str(summary),
                    "page_number": 1,
                    "image_name": "unknown",
                })
        
        return self.chunk_document(
            markdown_content=parsed_doc.markdown_content,
            image_summaries=image_summaries,
            source_document=parsed_doc.document_name,
        )


def create_chunker(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> DocumentChunker:
    """
    Factory function to create a DocumentChunker.
    
    Args:
        chunk_size: Maximum chunk size for text content.
        chunk_overlap: Overlap between text chunks.
        
    Returns:
        Configured DocumentChunker instance.
    """
    return DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
