"""
Background document processor service.

Handles async document processing with status tracking.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
import tempfile
import shutil

# Apply nest_asyncio to allow nested event loops
import nest_asyncio
nest_asyncio.apply()

from api.schemas import ProcessingStatus, ProcessingStep, JobStatus
from config.settings import settings
from config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingJob:
    """Represents a document processing job."""
    job_id: str
    filename: str
    file_path: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    progress: int = 0
    current_step: str = "Initializing"
    error: Optional[str] = None
    collection_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Document stats
    page_count: Optional[int] = None
    image_count: Optional[int] = None
    chunk_count: Optional[int] = None
    
    # Processing steps
    steps: Dict[str, ProcessingStep] = field(default_factory=dict)
    
    # Cancellation flag
    cancelled: bool = False
    
    def __post_init__(self):
        self.steps = {
            "parsing": ProcessingStep(name="Parsing document", status="pending"),
            "extracting": ProcessingStep(name="Extracting images", status="pending"),
            "summarizing": ProcessingStep(name="Summarizing images", status="pending"),
            "embedding": ProcessingStep(name="Generating embeddings", status="pending"),
            "indexing": ProcessingStep(name="Building search index", status="pending"),
        }
    
    def update_step(self, step_name: str, status: str, progress: int = 0, message: Optional[str] = None):
        """Update a processing step."""
        if step_name in self.steps:
            self.steps[step_name].status = status
            self.steps[step_name].progress = progress
            self.steps[step_name].message = message
        self.updated_at = datetime.utcnow()
    
    def to_status(self) -> JobStatus:
        """Convert to JobStatus response model."""
        return JobStatus(
            job_id=self.job_id,
            filename=self.filename,
            status=self.status,
            progress=self.progress,
            current_step=self.current_step,
            steps=list(self.steps.values()),
            created_at=self.created_at,
            updated_at=self.updated_at,
            error=self.error,
            collection_name=self.collection_name,
            page_count=self.page_count,
            image_count=self.image_count,
            chunk_count=self.chunk_count,
        )


class DocumentProcessorService:
    """Service for managing document processing jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self.temp_dir = Path(tempfile.gettempdir()) / "doculens_uploads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DocumentProcessorService initialized, temp_dir: {self.temp_dir}")
    
    def create_job(self, filename: str, file_content: bytes) -> ProcessingJob:
        """Create a new processing job."""
        job_id = str(uuid.uuid4())[:8]
        
        # Save file to temp directory
        file_path = self.temp_dir / f"{job_id}_{filename}"
        file_path.write_bytes(file_content)
        
        # Create collection name from job_id
        collection_name = f"doculens_{job_id}"
        
        job = ProcessingJob(
            job_id=job_id,
            filename=filename,
            file_path=str(file_path),
            collection_name=collection_name,
        )
        self.jobs[job_id] = job
        
        logger.info(f"Created job {job_id} for file {filename}")
        return job
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Get a job by ID."""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job."""
        job = self.jobs.get(job_id)
        if job and job.status not in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
            job.cancelled = True
            job.status = ProcessingStatus.CANCELLED
            job.current_step = "Cancelled by user"
            job.updated_at = datetime.utcnow()
            logger.info(f"Job {job_id} cancelled")
            return True
        return False
    
    async def process_document(self, job_id: str):
        """Process a document asynchronously."""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        try:
            # Import here to avoid circular imports
            from ingestion.pipeline import IngestionPipeline, create_parser
            from ingestion.image_summarizer import create_summarizer
            from index.indexer import Indexer, create_indexer
            from index.qdrant_store import create_qdrant_store
            from index.embeddings import create_embedding_service, configure_global_embeddings
            from index.chunking import create_chunker
            
            logger.info(f"Starting processing for job {job_id}")
            
            # Step 1: Parse document
            if job.cancelled:
                return
            
            job.status = ProcessingStatus.PARSING
            job.current_step = "Parsing document with LlamaParse"
            job.progress = 5
            job.update_step("parsing", "in_progress", 0, "Initializing parser...")
            
            # Create pipeline
            pipeline = IngestionPipeline(
                parser=create_parser(),
                summarizer=create_summarizer(),
                output_dir=str(settings.DATA_DIR),
                summarize_images=True,
            )
            
            job.update_step("parsing", "in_progress", 20, "Sending to LlamaParse...")
            
            # Process document - run in executor to not block
            loop = asyncio.get_event_loop()
            parsed_doc = await loop.run_in_executor(
                None,
                lambda: pipeline.process(job.file_path, save_outputs=True, skip_existing=False)
            )
            
            job.page_count = parsed_doc.page_count
            job.image_count = len(parsed_doc.images)
            job.update_step("parsing", "completed", 100, f"Parsed {parsed_doc.page_count} pages")
            job.progress = 20
            
            logger.info(f"Job {job_id}: Parsed {parsed_doc.page_count} pages, {len(parsed_doc.images)} images")
            
            # Step 2: Images already extracted by pipeline
            if job.cancelled:
                return
            
            job.status = ProcessingStatus.EXTRACTING_IMAGES
            job.current_step = "Processing extracted images"
            job.update_step("extracting", "in_progress", 50, f"Extracted {len(parsed_doc.images)} images")
            await asyncio.sleep(0.1)  # Yield to event loop
            job.update_step("extracting", "completed", 100, f"{len(parsed_doc.images)} images ready")
            job.progress = 40
            
            # Step 3: Image summaries already generated by pipeline
            if job.cancelled:
                return
            
            job.status = ProcessingStatus.SUMMARIZING_IMAGES
            job.current_step = "Image summaries generated"
            job.update_step("summarizing", "in_progress", 50, f"Processing {len(parsed_doc.image_summaries)} summaries")
            await asyncio.sleep(0.1)
            job.update_step("summarizing", "completed", 100, f"{len(parsed_doc.image_summaries)} summaries ready")
            job.progress = 60
            
            logger.info(f"Job {job_id}: Generated {len(parsed_doc.image_summaries)} image summaries")
            
            # Step 4: Generate embeddings and index
            if job.cancelled:
                return
            
            job.status = ProcessingStatus.EMBEDDING
            job.current_step = "Generating embeddings"
            job.update_step("embedding", "in_progress", 0, "Creating embedding service...")
            
            # Create indexer with job-specific collection
            embedding_service = create_embedding_service()
            configure_global_embeddings(model_name=embedding_service.model_name)
            
            qdrant_store = create_qdrant_store(
                collection_name=job.collection_name,
                embedding_model=embedding_service.model_name,
            )
            
            chunker = create_chunker()
            
            indexer = Indexer(
                qdrant_store=qdrant_store,
                embedding_service=embedding_service,
                chunker=chunker,
                data_dir=str(settings.DATA_DIR),
            )
            
            job.update_step("embedding", "in_progress", 30, "Chunking document...")
            
            # Load image summaries
            image_summaries = indexer.load_image_summaries(parsed_doc.document_name)
            
            # Chunk document
            nodes = indexer.chunk_document(
                document_name=parsed_doc.document_name,
                markdown_content=parsed_doc.markdown_content,
                image_summaries=image_summaries,
            )
            job.chunk_count = len(nodes)
            
            job.update_step("embedding", "in_progress", 60, f"Embedding {len(nodes)} chunks...")
            
            # Embed nodes
            nodes = await loop.run_in_executor(
                None,
                lambda: embedding_service.embed_nodes(nodes, show_progress=False)
            )
            
            job.update_step("embedding", "completed", 100, f"Embedded {len(nodes)} chunks")
            job.progress = 80
            
            logger.info(f"Job {job_id}: Created {len(nodes)} embedded chunks")
            
            # Step 5: Index to Qdrant
            if job.cancelled:
                return
            
            job.status = ProcessingStatus.INDEXING
            job.current_step = "Building search index"
            job.update_step("indexing", "in_progress", 30, "Creating Qdrant collection...")
            
            # Create collection
            qdrant_store.create_collection(recreate=True)
            
            job.update_step("indexing", "in_progress", 60, "Indexing vectors...")
            
            # Build index
            from llama_index.core import VectorStoreIndex
            
            storage_context = qdrant_store.get_storage_context()
            
            index = await loop.run_in_executor(
                None,
                lambda: VectorStoreIndex(
                    nodes=nodes,
                    storage_context=storage_context,
                    embed_model=embedding_service.embed_model,
                    show_progress=False,
                )
            )
            
            vector_count = qdrant_store.count_vectors()
            job.update_step("indexing", "completed", 100, f"Indexed {vector_count} vectors")
            job.progress = 100
            
            logger.info(f"Job {job_id}: Indexed {vector_count} vectors to collection {job.collection_name}")
            
            # Complete
            job.status = ProcessingStatus.COMPLETED
            job.current_step = "Processing complete"
            job.updated_at = datetime.utcnow()
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            job.current_step = "Processing failed"
            
            # Mark current step as failed
            for step in job.steps.values():
                if step.status == "in_progress":
                    step.status = "failed"
                    step.message = str(e)
        
        finally:
            # Clean up temp file
            try:
                Path(job.file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed/failed jobs."""
        now = datetime.utcnow()
        to_remove = []
        
        for job_id, job in self.jobs.items():
            age = (now - job.created_at).total_seconds() / 3600
            if age > max_age_hours and job.status in [
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
                ProcessingStatus.CANCELLED,
            ]:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
            logger.info(f"Cleaned up old job {job_id}")


# Singleton instance
_processor_service: Optional[DocumentProcessorService] = None


def get_processor_service() -> DocumentProcessorService:
    """Get the singleton processor service."""
    global _processor_service
    if _processor_service is None:
        _processor_service = DocumentProcessorService()
    return _processor_service
