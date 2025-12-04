"""
DocuLens FastAPI Application

Main entry point for the REST API backend.
Run with: uvicorn api.main:app --reload --port 8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import documents, query
from api.dependencies import QdrantClientManager
from api.schemas import HealthResponse
from config.logging_config import get_logger, setup_logging
from config.settings import settings

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks.
    """
    # Startup
    logger.info("DocuLens API starting up...")
    logger.info(f"Qdrant URL: {settings.QDRANT_URL}")
    
    # Verify Qdrant connection
    try:
        client = QdrantClientManager.get_client()
        collections = client.get_collections()
        logger.info(f"Connected to Qdrant. Collections: {len(collections.collections)}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
    
    yield
    
    # Shutdown
    logger.info("DocuLens API shutting down...")
    QdrantClientManager.close()


# Create FastAPI app
app = FastAPI(
    title="DocuLens API",
    description="Multi-Modal RAG API for document processing and querying",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return JSONResponse(
        content={
            "message": "DocuLens API",
            "docs": "/docs",
            "health": "/health",
        }
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status and Qdrant connection state.
    """
    qdrant_connected = False
    
    try:
        client = QdrantClientManager.get_client()
        client.get_collections()
        qdrant_connected = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
    
    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        version="1.0.0",
        qdrant_connected=qdrant_connected,
        timestamp=datetime.utcnow(),
    )


@app.get("/api/settings", tags=["config"])
async def get_api_settings():
    """
    Get current API configuration (non-sensitive).
    """
    return {
        "qdrant_url": settings.QDRANT_URL,
        "default_collection": settings.QDRANT_COLLECTION,
        "embedding_model": "all-MiniLM-L6-v2",
        "llm_model": "gpt-5-nano-2025-08-07",
        "max_upload_size_mb": 50,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
