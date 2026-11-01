from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from the environment or the `.env` file.

    The two API keys and the database URL have no default, so the app fails at startup
    when one is missing instead of failing on the first request.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr
    pinecone_api_key: SecretStr

    # Postgres (a free Neon project is enough). Threads, farmer memory and each turn's evidence live here.
    database_url: SecretStr

    # The farmer this app acts for. Identity is decided here, on the server, never by the request:
    # see current_farmer in dependencies.py, the one place a real auth service would plug in.
    farmer_id: str = "frm_7f3a9c2e"
    farmer_name: str = "Nimal Perera"

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

    # The FieldSat MCP server from Level 6 Lesson 3. Port 8001, because this app uses 8000.
    fieldsat_url: str = "http://127.0.0.1:8001/mcp"

    cors_origins: list[str] = ["http://localhost:3000"]

    # Langfuse tracing. Optional: with no keys the app runs without it.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
