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

    # The FieldSat MCP server from Level 6 Lesson 3. Port 8001, because this app uses 8000.
    fieldsat_url: str = "http://127.0.0.1:8001/mcp"

    cors_origins: list[str] = ["http://localhost:3000"]

    # Langfuse tracing. Optional: with no keys the app runs without it.
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    # Separates traces in Langfuse: "development" on a laptop, "production" in the deployed stack.
    langfuse_environment: str = "development"

    # Lesson 3: guardrails (core/guardrails.py). GUARDRAILS=false switches them all off, for the before/after demo.
    guardrails: bool = True
    # Input moderation calls OpenAI's Moderations endpoint. MODERATION=false skips it, for example when the
    # API key is not allowed to call it (HTTP 403), rather than recording "moderation_unavailable" on every turn.
    moderation: bool = True
    moderation_model: str = "omni-moderation-latest"
    # Output check: a fertiliser dose above this, per application, is withheld. A demo value, not agronomy advice.
    max_dose_kg_per_ha: int = 150
    # Limits per chat turn, so a confused orchestrator cannot loop and burn tokens.
    max_model_calls: int = 12
    max_tool_calls: int = 10

    # Lesson 3: the LLM-as-judge used by evals/ and post-deployment evaluation. Pin it; change it on purpose.
    judge_model: str = "openai:gpt-5.6-luna"
