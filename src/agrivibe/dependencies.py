"""Wiring. Each function builds one object; FastAPI injects them into routes.

Everything is created once and reused (lru_cache). Tests replace these
functions with fakes through app.dependency_overrides.
"""

from functools import lru_cache

from langchain_pinecone import PineconeVectorStore

from agrivibe.config import Settings
from agrivibe.core.vectorstore import create_vector_store
from agrivibe.services.knowledge import KnowledgeService


@lru_cache
def get_settings() -> Settings:
    """Returns the application settings, loaded once."""
    return Settings()


@lru_cache
def get_vector_store() -> PineconeVectorStore:
    """Returns the shared Pinecone vector store."""
    return create_vector_store(get_settings())


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    """Returns the shared service that indexes and searches PDFs."""
    settings = get_settings()
    return KnowledgeService(get_vector_store(), settings.chunk_size, settings.chunk_overlap)
