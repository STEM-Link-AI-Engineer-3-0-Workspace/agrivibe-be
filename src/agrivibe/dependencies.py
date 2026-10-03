"""Wiring. Each function builds one object; FastAPI injects them into routes.

Everything is created once and reused (lru_cache). Tests replace these
functions with fakes through app.dependency_overrides.
"""

from functools import lru_cache

from langchain_pinecone import PineconeVectorStore
from langgraph.checkpoint.memory import InMemorySaver

from agrivibe.agents.rag import build_rag_agent
from agrivibe.config import Settings
from agrivibe.core.llm import create_chat_model
from agrivibe.db.vectorstore import create_vector_store
from agrivibe.services.answering import AnswerService
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


@lru_cache
def get_answer_service() -> AnswerService:
    """Returns the shared service that answers questions with the RAG agent."""
    settings = get_settings()
    # Threads live in process memory: they reset when the app restarts and are not shared between workers.
    rag_agent = build_rag_agent(
        create_chat_model(settings), get_vector_store(), k=settings.search_k, checkpointer=InMemorySaver()
    )
    return AnswerService(rag_agent)
