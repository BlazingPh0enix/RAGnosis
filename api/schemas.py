"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PARSING = "parsing"
    EXTRACTING_IMAGES = "extracting_images"
    SUMMARIZING_IMAGES = "summarizing_images"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UploadResponse(BaseModel):
    """Response from document upload."""
    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Original filename")
    status: ProcessingStatus = Field(..., description="Current processing status")
    message: str = Field(..., description="Status message")


class ProcessingStep(BaseModel):
    """A single processing step."""
    name: str
    status: str  # pending, in_progress, completed, failed
    progress: int = 0  # 0-100
    message: Optional[str] = None


class JobStatus(BaseModel):
    """Detailed job status."""
    job_id: str
    filename: str
    status: ProcessingStatus
    progress: int = Field(..., ge=0, le=100, description="Overall progress 0-100")
    current_step: str
    steps: List[ProcessingStep]
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    collection_name: Optional[str] = None
    
    # Document stats (available after parsing)
    page_count: Optional[int] = None
    image_count: Optional[int] = None
    chunk_count: Optional[int] = None


class QueryRequest(BaseModel):
    """Request to query documents."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    collection_name: str = Field(..., description="Qdrant collection to query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")


class SourceInfo(BaseModel):
    """Information about a retrieved source."""
    page_number: int
    content_type: str  # text, image_summary
    source_document: str
    score: float
    text_preview: str
    image_name: Optional[str] = None


class QueryResponse(BaseModel):
    """Response from a query."""
    query: str
    response: str
    sources: List[SourceInfo]
    cited_pages: List[int]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CollectionInfo(BaseModel):
    """Information about a Qdrant collection."""
    name: str
    vectors_count: int
    status: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    qdrant_connected: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
