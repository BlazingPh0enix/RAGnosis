"""
FastAPI dependency injection and shared resources.
"""

import sys
from pathlib import Path
from typing import Optional
from functools import lru_cache

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient

from config.settings import settings
from config.logging_config import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)


class QdrantClientManager:
    """Singleton manager for Qdrant client connection."""
    
    _instance: Optional[QdrantClient] = None
    
    @classmethod
    def get_client(cls) -> QdrantClient:
        """Get or create Qdrant client instance."""
        if cls._instance is None:
            logger.info(f"Creating Qdrant client connection to {settings.QDRANT_URL}")
            cls._instance = QdrantClient(url=settings.QDRANT_URL)
        return cls._instance
    
    @classmethod
    def close(cls):
        """Close the client connection."""
        if cls._instance is not None:
            logger.info("Closing Qdrant client connection")
            cls._instance.close()
            cls._instance = None


def get_qdrant_client() -> QdrantClient:
    """Dependency to get Qdrant client."""
    return QdrantClientManager.get_client()


@lru_cache()
def get_settings():
    """Get cached settings instance."""
    return settings
