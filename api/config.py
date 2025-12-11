"""
Configuration management.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API settings
    api_title: str = "Financial Document RAG API"
    api_version: str = "1.0.0"
    api_description: str = "Production-ready REST API for financial document Q&A"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    # CORS settings
    cors_origins: list = ["*"]

    # Logging
    log_level: str = "INFO"

    # RAG settings
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_k_results: int = 5
    default_min_score: float = 0.5

    # File upload
    max_upload_size: int = 50_000_000  # 50MB
    allowed_extensions: list = [".pdf"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
