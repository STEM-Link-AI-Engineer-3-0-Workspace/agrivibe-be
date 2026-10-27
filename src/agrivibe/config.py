from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from the environment or the `.env` file.

    The two API keys have no default, so the app fails at startup when either is
    missing instead of failing on the first request.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    pinecone_api_key: SecretStr

    chat_model: str = "openai:gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    pinecone_index: str = "fieldoracle"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    search_k: int = 4

    # Two-stage retrieval: fetch RERANK_CANDIDATES chunks, re-score them with a
    # cross-encoder and keep the best SEARCH_K. "none" skips the second stage.
    reranker: Literal["pinecone", "none"] = "pinecone"
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_candidates: int = 20

    cors_origins: list[str] = ["http://localhost:3000"]
