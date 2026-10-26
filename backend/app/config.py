"""Application configuration management using Pydantic Settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base backend directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Project Metadata
    PROJECT_NAME: str = "DocuMind AI"
    VERSION: str = "0.3.0"
    DESCRIPTION: str = "AI-Powered Document Question Answering & Semantic RAG System"
    API_V1_STR: str = "/api/v1"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # CORS Configuration
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:5173,http://127.0.0.1:5173"

    # Document Ingestion & Chunking
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB default
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Storage Paths (relative to backend root)
    DATA_DIR: str = "data"
    CHROMA_PERSIST_DIR: str = "data/chroma"
    DOCUMENTS_METADATA_FILE: str = "data/documents_metadata.json"
    CHROMA_COLLECTION_NAME: str = "documind_documents"

    # Gemini API & RAG Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.8-flash"
    RAG_TOP_K: int = 4
    RAG_RELEVANCE_THRESHOLD: float = 0.85  # Max cosine distance (smaller distance = higher relevance)
    MAX_QUERY_LENGTH: int = 1000
    MAX_CONTEXT_CHARS: int = 12000

    @field_validator("RAG_RELEVANCE_THRESHOLD", mode="before")
    @classmethod
    def _validate_threshold(cls, v: Any) -> float:
        """Calibrate legacy 0.75 default to 0.85 without modifying local .env."""
        if v is None or v == 0.75 or v == "0.75":
            return 0.85
        return float(v)

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated CORS_ORIGINS string to a clean list of URLs."""
        if isinstance(self.CORS_ORIGINS, list):
            return self.CORS_ORIGINS
        if isinstance(self.CORS_ORIGINS, str):
            return [
                origin.strip()
                for origin in self.CORS_ORIGINS.split(",")
                if origin.strip()
            ]
        return ["*"]

    @property
    def data_dir_path(self) -> Path:
        """Absolute path to the data storage directory."""
        path = BASE_DIR / self.DATA_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_dir_path(self) -> Path:
        """Absolute path to ChromaDB persistence directory."""
        path = BASE_DIR / self.CHROMA_PERSIST_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def metadata_file_path(self) -> Path:
        """Absolute path to document metadata JSON store."""
        return BASE_DIR / self.DOCUMENTS_METADATA_FILE


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
