"""
Document management API routes.

Handles file upload, processing status, and cancellation.
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from api.schemas import (
    UploadResponse,
    JobStatus,
    ProcessingStatus,
    CollectionInfo,
    ErrorResponse,
)
from api.services.processor import get_processor_service
from api.dependencies import get_qdrant_client
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}},
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process"),
):
    """
    Upload a PDF document for processing.
    
    The document will be processed asynchronously:
    1. Parse with LlamaParse
    2. Extract and summarize images
    3. Generate embeddings
    4. Index to Qdrant
    
    Returns a job_id to track progress.
    """
    # Validate filename exists
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    filename = file.filename
    
    # Validate file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        logger.warning(f"Invalid file type: {file_ext}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    if len(content) > MAX_FILE_SIZE:
        logger.warning(f"File too large: {len(content)} bytes")
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB",
        )
    
    # Create processing job
    processor = get_processor_service()
    job = processor.create_job(filename, content)
    
    # Start background processing
    background_tasks.add_task(processor.process_document, job.job_id)
    
    logger.info(f"Started processing job {job.job_id} for {filename}")
    
    return UploadResponse(
        job_id=job.job_id,
        filename=filename,
        status=ProcessingStatus.PENDING,
        message="Document uploaded successfully. Processing started.",
    )


@router.get(
    "/{job_id}/status",
    response_model=JobStatus,
    responses={404: {"model": ErrorResponse}},
)
async def get_job_status(job_id: str):
    """
    Get the processing status of a document.
    
    Poll this endpoint to track progress.
    """
    processor = get_processor_service()
    job = processor.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job.to_status()


@router.post(
    "/{job_id}/cancel",
    response_model=JobStatus,
    responses={404: {"model": ErrorResponse}, 400: {"model": ErrorResponse}},
)
async def cancel_job(job_id: str):
    """
    Cancel a processing job.
    
    Only works for jobs that are still in progress.
    """
    processor = get_processor_service()
    job = processor.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}",
        )
    
    processor.cancel_job(job_id)
    logger.info(f"Cancelled job {job_id}")
    
    return job.to_status()


@router.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """
    List all available Qdrant collections.
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        
        result = []
        for col in collections:
            try:
                info = client.get_collection(col.name)
                result.append(CollectionInfo(
                    name=col.name,
                    vectors_count=info.points_count or 0,
                    status=str(info.status),
                ))
            except Exception:
                result.append(CollectionInfo(
                    name=col.name,
                    vectors_count=0,
                    status="unknown",
                ))
        
        return result
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/collections/{collection_name}",
    responses={404: {"model": ErrorResponse}},
)
async def delete_collection(collection_name: str):
    """
    Delete a Qdrant collection.
    """
    try:
        client = get_qdrant_client()
        
        # Check if collection exists
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{collection_name}' not found",
            )
        
        client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))
