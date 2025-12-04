"""
Ingestion pipeline orchestrator for document processing.

This module coordinates the complete document ingestion workflow:
1. Parse PDF using LlamaParse
2. Extract images from the document
3. Generate summaries for extracted images
4. Package everything into a structured ParsedDocument
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from llama_cloud_services import LlamaParse

from ingestion.llama_parse_client import create_parser
from ingestion.image_summarizer import ImageSummarizer, ImageSummary, create_summarizer
from config.settings import settings
from config.logging_config import get_logger, log_execution_time, LogTimer

logger = get_logger(__name__)


@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""
    name: str
    page_number: int
    width: int
    height: int
    file_path: str = ""
    
    def to_dict(self) -> dict:
        return {"name": self.name, "page_number": self.page_number, "width": self.width,
                "height": self.height, "file_path": self.file_path}


@dataclass
class ParsedDocument:
    """Complete parsed document with all extracted content."""
    source_path: str
    document_name: str
    markdown_content: str
    images: List[ExtractedImage] = field(default_factory=list)
    image_summaries: List[ImageSummary] = field(default_factory=list)
    page_count: int = 0
    is_cache_hit: bool = False
    auto_mode_triggered_pages: int = 0
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "source_path": self.source_path, "document_name": self.document_name,
            "markdown_content": self.markdown_content,
            "images": [img.to_dict() for img in self.images],
            "image_summaries": [s.to_dict() for s in self.image_summaries],
            "page_count": self.page_count, "is_cache_hit": self.is_cache_hit,
            "auto_mode_triggered_pages": self.auto_mode_triggered_pages,
            "processed_at": self.processed_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ParsedDocument":
        return cls(
            source_path=data["source_path"], document_name=data["document_name"],
            markdown_content=data["markdown_content"],
            images=[ExtractedImage(**img) for img in data.get("images", [])],
            image_summaries=[ImageSummary.from_dict(s) for s in data.get("image_summaries", [])],
            page_count=data.get("page_count", 0), is_cache_hit=data.get("is_cache_hit", False),
            auto_mode_triggered_pages=data.get("auto_mode_triggered_pages", 0),
            processed_at=data.get("processed_at", ""),
        )


class IngestionPipeline:
    """Orchestrates document ingestion: parsing, image extraction, and summarization."""
    
    def __init__(
        self, parser: Optional[LlamaParse] = None, summarizer: Optional[ImageSummarizer] = None,
        output_dir: Optional[str] = None, summarize_images: bool = True,
    ):
        self.parser = parser or create_parser()
        self.summarizer = summarizer if summarize_images else None
        self.summarize_images = summarize_images
        self.output_dir = Path(output_dir or settings.DATA_DIR)
        self.parsed_dir = self.output_dir / "parsed"
        self.images_dir = self.output_dir / "images"
        self.summaries_dir = self.output_dir / "summaries"
        self._ensure_directories()
        logger.info(f"IngestionPipeline initialized with output_dir: {self.output_dir}")
    
    def _ensure_directories(self) -> None:
        for d in [self.parsed_dir, self.images_dir, self.summaries_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    @log_execution_time()
    def process(self, pdf_path: str, save_outputs: bool = True, skip_existing: bool = False) -> ParsedDocument:
        """Process a PDF through the complete ingestion pipeline."""
        pdf_path_obj = Path(pdf_path).resolve()
        document_name = pdf_path_obj.stem
        
        logger.info(f"Starting document processing: {pdf_path_obj.name}")
        logger.debug(f"Full path: {pdf_path_obj}, save_outputs={save_outputs}, skip_existing={skip_existing}")
        
        # Check if already processed
        if skip_existing:
            existing = self._load_existing(document_name)
            if existing:
                logger.info(f"Skipping {document_name} - already processed (loaded from cache)")
                return existing
        
        # Step 1: Parse the document using LlamaParse
        # The parse() method returns a JobResult with all content
        logger.debug("Step 1: Parsing with LlamaParse...")
        with LogTimer(logger, f"LlamaParse parsing for {document_name}"):
            result = self.parser.parse(str(pdf_path_obj))
        
        # Handle both single JobResult and List[JobResult] (for partitioned files)
        # For single file parsing, we typically get a single JobResult
        if isinstance(result, list):
            job_result = result[0]  # Take first result for single file
        else:
            job_result = result
        
        # Access metadata from job_result
        is_cache_hit = job_result.job_metadata.job_is_cache_hit
        page_count = job_result.job_metadata.job_pages
        auto_mode_triggered = job_result.job_metadata.job_auto_mode_triggered_pages or 0
        
        if is_cache_hit:
            logger.info(f"{document_name}: LlamaParse cache hit - no credits used")
        else:
            logger.info(f"{document_name}: Parsed {page_count} pages (auto-mode triggered: {auto_mode_triggered})")
        
        # Step 2: Get markdown content
        markdown_content = job_result.get_markdown()
        
        # Step 3: Save images and get their metadata
        doc_images_dir = self.images_dir / document_name
        doc_images_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_images: List[ExtractedImage] = []
        image_data_list: List[tuple] = []  # (image_data, name, page_number) for summarization
        
        # Iterate through pages to get image info
        for page in job_result.pages:
            for img in page.images:
                try:
                    # Get image data using the JobResult method
                    image_data = job_result.get_image_data(img.name)
                    
                    # Save to disk
                    file_path = doc_images_dir / img.name
                    with open(file_path, "wb") as f:
                        f.write(image_data)
                    
                    extracted_images.append(ExtractedImage(
                        name=img.name,
                        page_number=page.page,
                        width=int(img.width or 0),
                        height=int(img.height or 0),
                        file_path=str(file_path),
                    ))
                    
                    # Store for summarization
                    image_data_list.append((image_data, img.name, page.page))
                    
                except Exception as e:
                    logger.warning(f"{document_name}: Failed to extract image {img.name}: {e}")
        
        logger.info(f"{document_name}: Extracted {len(extracted_images)} images")
        
        # Step 4: Summarize images
        image_summaries = []
        if self.summarize_images and self.summarizer and image_data_list:
            logger.debug(f"{document_name}: Starting image summarization for {len(image_data_list)} images")
            with LogTimer(logger, f"Image summarization for {document_name}"):
                for image_data, name, page_num in image_data_list:
                    try:
                        summary = self.summarizer.summarize_image(
                            image_data=image_data,
                            image_name=name,
                            page_number=page_num,
                            source_document=document_name,
                        )
                        image_summaries.append(summary)
                    except Exception as e:
                        logger.warning(f"{document_name}: Failed to summarize {name}: {e}")
            logger.info(f"{document_name}: Generated {len(image_summaries)} image summaries")
        
        # Step 5: Create the parsed document
        parsed_document = ParsedDocument(
            source_path=str(pdf_path_obj),
            document_name=document_name,
            markdown_content=markdown_content,
            images=extracted_images,
            image_summaries=image_summaries,
            page_count=page_count,
            is_cache_hit=is_cache_hit,
            auto_mode_triggered_pages=auto_mode_triggered,
        )
        
        # Step 6: Save outputs
        if save_outputs:
            self._save_outputs(parsed_document)
            logger.debug(f"{document_name}: Outputs saved to {self.output_dir}")
        
        logger.info(f"Completed processing document: {document_name}")
        return parsed_document
    
    def _save_outputs(self, document: ParsedDocument) -> None:
        """Save all document outputs to disk."""
        doc_name = document.document_name
        
        # Save markdown
        (self.parsed_dir / f"{doc_name}.md").write_text(document.markdown_content, encoding="utf-8")
        
        # Save image summaries
        if document.image_summaries:
            with open(self.summaries_dir / f"{doc_name}_summaries.json", "w") as f:
                json.dump([s.to_dict() for s in document.image_summaries], f, indent=2)
        
        # Save metadata
        metadata = {
            "source_path": document.source_path, "document_name": doc_name,
            "page_count": document.page_count, "is_cache_hit": document.is_cache_hit,
            "auto_mode_triggered_pages": document.auto_mode_triggered_pages,
            "processed_at": document.processed_at,
            "images": [img.to_dict() for img in document.images],
            "image_summary_count": len(document.image_summaries),
        }
        with open(self.parsed_dir / f"{doc_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _load_existing(self, document_name: str) -> Optional[ParsedDocument]:
        """Load a previously processed document if it exists."""
        md_path = self.parsed_dir / f"{document_name}.md"
        metadata_path = self.parsed_dir / f"{document_name}_metadata.json"
        
        if not (md_path.exists() and metadata_path.exists()):
            return None
        
        try:
            markdown = md_path.read_text(encoding="utf-8")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            
            # Load image summaries
            summaries = []
            summaries_path = self.summaries_dir / f"{document_name}_summaries.json"
            if summaries_path.exists():
                summaries = [ImageSummary.from_dict(s) for s in json.loads(summaries_path.read_text())]
            
            return ParsedDocument(
                source_path=metadata["source_path"], document_name=metadata["document_name"],
                markdown_content=markdown,
                images=[ExtractedImage(**img) for img in metadata.get("images", [])],
                image_summaries=summaries, page_count=metadata.get("page_count", 0),
                is_cache_hit=metadata.get("is_cache_hit", False),
                auto_mode_triggered_pages=metadata.get("auto_mode_triggered_pages", 0),
                processed_at=metadata.get("processed_at", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to load existing document {document_name}: {e}")
            return None
    
    def process_directory(
        self, directory: str, save_outputs: bool = True, skip_existing: bool = True,
    ) -> List[ParsedDocument]:
        """Process all PDF files in a directory."""
        dir_path = Path(directory)
        pdf_files = list(dir_path.glob("*.pdf")) + list(dir_path.glob("*.PDF"))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        documents = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_path.name}")
            try:
                documents.append(self.process(str(pdf_path), save_outputs, skip_existing))
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        
        logger.info(f"Completed: {len(documents)}/{len(pdf_files)} documents")
        return documents


def create_pipeline(summarize_images: bool = True, output_dir: Optional[str] = None) -> IngestionPipeline:
    """Factory function to create an IngestionPipeline with default settings."""
    logger.info(f"Creating IngestionPipeline (summarize_images={summarize_images})")
    return IngestionPipeline(
        parser=create_parser(),
        summarizer=create_summarizer() if summarize_images else None,
        output_dir=output_dir,
        summarize_images=summarize_images,
    )
