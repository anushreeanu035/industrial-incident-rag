"""Configuration settings for Industrial Incident Investigator."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq API configuration
    GROQ_API_KEY: str = Field(default="", description="API key for Groq Cloud")
    GROQ_MODEL: str = Field(default="llama-3.3-70b-versatile", description="Default Groq LLM model")

    # Local embedding & reranker models
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Sentence Transformers embedding model")
    RERANKER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Cross-encoder reranking model")

    # Storage and paths
    QDRANT_PATH: str = Field(default="data/processed/qdrant_db", description="Local path for Qdrant vector database")
    QDRANT_COLLECTION_NAME: str = Field(default="incident_chunks", description="Collection name in Qdrant")
    BM25_INDEX_PATH: str = Field(default="data/processed/bm25_index.pkl", description="Path to persisted BM25 index")
    DOCUMENT_STORE_PATH: str = Field(default="data/processed/documents.json", description="Path to metadata store")
    BENCHMARK_DATASET_PATH: str = Field(default="data/benchmark/retrieval_benchmark.json", description="Benchmark dataset path")
    REPORTS_DIR: str = Field(default="reports", description="Directory where benchmark evaluation reports are written")

    # Chunking parameters
    CHUNK_SIZE: int = Field(default=400, description="Target chunk size in words/tokens")
    CHUNK_OVERLAP: int = Field(default=80, description="Chunk overlap")

    # Retrieval and fusion parameters
    DENSE_TOP_K: int = Field(default=10, ge=1, le=50, description="Top K results from dense retrieval")
    BM25_TOP_K: int = Field(default=10, ge=1, le=50, description="Top K results from BM25 retrieval")
    RRF_K: int = Field(default=60, ge=1, description="RRF constant k (standard is 60)")
    RERANK_TOP_K: int = Field(default=5, ge=1, le=20, description="Top K candidates passed to cross-encoder")
    FINAL_CONTEXT_K: int = Field(default=5, ge=1, le=20, description="Final context chunks for LLM generation")

    # Query rewriting
    QUERY_REWRITING_ENABLED: bool = Field(default=True, description="Enable query rewriting via LLM")
    NUM_QUERY_REWRITES: int = Field(default=3, ge=1, le=5, description="Number of query variants to generate")

    # Server & Client
    FASTAPI_HOST: str = Field(default="127.0.0.1", description="FastAPI host")
    FASTAPI_PORT: int = Field(default=8000, description="FastAPI port")
    BACKEND_URL: str = Field(default="http://127.0.0.1:8000", description="Backend URL used by frontend")

    # Logging & Environment
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    ENVIRONMENT: str = Field(default="development", description="Runtime environment")

    def resolve_path(self, relative_path: str) -> Path:
        """Resolves a relative path against the project root directory."""
        path = Path(relative_path)
        if path.is_absolute():
            return path
        return BASE_DIR / path


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings instance."""
    return Settings()
