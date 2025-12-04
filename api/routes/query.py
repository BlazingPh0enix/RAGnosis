"""
Query API routes.

Handles document querying and response generation.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException

from api.schemas import QueryRequest, QueryResponse, SourceInfo, ErrorResponse
from api.dependencies import get_qdrant_client
from config.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def query_documents(request: QueryRequest):
    """
    Query indexed documents and get an AI-generated response.
    
    Retrieves relevant chunks from the specified collection
    and generates a response using the LLM.
    """
    # Import here to avoid startup delays
    from retrieval.query_engine import load_query_engine
    from index.qdrant_store import create_qdrant_store
    
    logger.info(f"Query: '{request.query[:50]}...' on collection '{request.collection_name}'")
    
    # Verify collection exists
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        
        if request.collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_name}' not found. "
                       f"Available: {', '.join(collections) if collections else 'none'}",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to check collections: {e}")
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
    
    # Load query engine and execute query
    try:
        engine = load_query_engine(
            collection_name=request.collection_name,
            top_k=request.top_k,
        )
        
        result = engine.query(request.query, top_k=request.top_k)
        
        # Format sources
        sources = [
            SourceInfo(
                page_number=s.get("page_number", 1),
                content_type=s.get("content_type", "text"),
                source_document=s.get("source_document", "unknown"),
                score=s.get("score", 0.0),
                text_preview=s.get("text_preview", ""),
                image_name=s.get("image_name"),
            )
            for s in result.sources
        ]
        
        logger.info(f"Query successful, {len(sources)} sources returned")
        
        return QueryResponse(
            query=request.query,
            response=result.response,
            sources=sources,
            cited_pages=result.cited_pages,
            metadata=result.metadata,
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/retrieve")
async def retrieve_only(request: QueryRequest):
    """
    Retrieve relevant document chunks without generating a response.
    
    Useful for debugging or when you want to see raw retrieval results.
    """
    from retrieval.query_engine import load_query_engine
    
    logger.info(f"Retrieve-only: '{request.query[:50]}...'")
    
    # Verify collection exists
    try:
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        
        if request.collection_name not in collections:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_name}' not found.",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
    
    try:
        engine = load_query_engine(
            collection_name=request.collection_name,
            top_k=request.top_k,
        )
        
        result = engine.retrieve_only(request.query, top_k=request.top_k)
        
        return {
            "query": request.query,
            "results": [
                {
                    "content": node.node.get_content()[:500],
                    "score": node.score,
                    "metadata": node.node.metadata,
                }
                for node in result.nodes
            ],
            "total_results": len(result.nodes),
        }
        
    except Exception as e:
        logger.error(f"Retrieve failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")
