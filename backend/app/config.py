"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    VERSION: str = "0.1.0"
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


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
