import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env
load_dotenv()

@dataclass
class Settings:
    # API keys
    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Qdrant
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "doculens_collection")

    # Parsing options
    LLAMAPARSE_AUTO_MODE: bool = os.getenv("LLAMAPARSE_AUTO_MODE", "True").lower() in ("true", "1", "yes")

    # Embeddings
    TEXT_EMBEDDING_MODEL: str = os.getenv("TEXT_EMBEDDING_MODEL", "text-embedding-3-small")
    IMAGE_EMBEDDING_MODEL: str = os.getenv("IMAGE_EMBEDDING_MODEL", "clip-mini")

    # Other options
    PAGE_PREFIX: str = os.getenv("PAGE_PREFIX", "[Page {pageNumber}]")
    DATA_DIR: str = os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    
    # FastAPI Settings
    API_HOST: str = os.getenv("API_HOST", "localhost")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_URL: str = os.getenv("API_URL", "http://localhost:8000")
    
    # Upload Settings
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS: str = os.getenv("ALLOWED_EXTENSIONS", ".pdf")


def get_settings() -> Settings:
    settings = Settings()

    # Minimal validation
    if not settings.OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not configured. Some features may not work.")
    if not settings.LLAMA_CLOUD_API_KEY:
        print("Warning: LLAMA_CLOUD_API_KEY not configured. LlamaParse API will not work.")

    return settings


# Export a module-level instance for convenience
settings = get_settings()