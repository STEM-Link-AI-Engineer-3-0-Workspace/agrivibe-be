"""Wiring. Each function builds one object; FastAPI injects them into routes.

Everything is created once and reused (lru_cache). Tests replace these
functions with fakes through app.dependency_overrides.
"""

from functools import lru_cache

from fastapi import Request
from langchain_pinecone import PineconeVectorStore
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agrivibe.agents.advisory import build_advisory_agent
from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldoracle import build_fieldoracle_agent
from agrivibe.agents.orchestrator import build_orchestrator
from agrivibe.agents.weather import build_weather_agent
from agrivibe.config import Settings
from agrivibe.core.llm import create_chat_model
from agrivibe.core.mcp import FieldSat
from agrivibe.core.reranker import Reranker
from agrivibe.core.tracing import create_tracer
from agrivibe.db.vectorstore import create_vector_store
from agrivibe.service.answering import AnswerService
from agrivibe.service.chatting import ChatService
from agrivibe.service.knowledge import KnowledgeService


@lru_cache
def get_settings() -> Settings:
    """Returns the application settings, loaded once."""
    # The required keys come from .env, which the type checker cannot see.
    return Settings()  # pyright: ignore[reportCallIssue]


@lru_cache
def get_vector_store() -> PineconeVectorStore:
    """Returns the shared Pinecone vector store."""
    return create_vector_store(get_settings())


@lru_cache
def get_reranker() -> Reranker | None:
    """Returns the shared cross-encoder named by `RERANKER`, or None when it is "none"."""
    settings = get_settings()
    if settings.reranker == "none":
        return None
    return Reranker(settings)


@lru_cache
def get_tracer() -> CallbackHandler | None:
    """Returns the shared Langfuse handler, or None when the Langfuse keys are not set."""
    return create_tracer(get_settings())


@lru_cache
def get_knowledge_service() -> KnowledgeService:
    """Returns the shared service that indexes and searches PDFs."""
    settings = get_settings()
    return KnowledgeService(get_vector_store(), settings.chunk_size, settings.chunk_overlap, get_reranker())


@lru_cache
def get_answer_service() -> AnswerService:
    """Returns the shared service that answers questions with the RAG agent."""
    settings = get_settings()
    # Threads live in process memory: they reset when the app restarts and are not shared between workers.
    rag_agent = build_fieldoracle_agent(
        create_chat_model(settings),
        get_vector_store(),
        k=settings.search_k,
        checkpointer=InMemorySaver(),
        reranker=get_reranker(),
    )
    return AnswerService(rag_agent, get_tracer())


def build_chat_service(settings: Settings, fieldsat: FieldSat) -> ChatService:
    """Builds the specialists and the orchestrator that calls them.

    Called once, from the lifespan in main.py: the FieldSat tools are only known after
    the app has asked the server for them.

    Args:
        settings: Application settings.
        fieldsat: The tools and instructions loaded from the FieldSat MCP server.

    Returns:
        The service behind POST /chat.
    """
    model = create_chat_model(settings)
    weather_agent = build_weather_agent(model, fieldsat)
    farm_context_agent = build_farm_context_agent(model, fieldsat)
    # A second FieldOracle, without a checkpointer: as a specialist it starts fresh on every call.
    # It searches the same way /ask does: vector search, then the cross-encoder.
    knowledge_agent = build_fieldoracle_agent(model, get_vector_store(), k=settings.search_k, reranker=get_reranker())
    advisory_agent = build_advisory_agent(model)
    # Both live in process memory, so they reset when the app restarts.
    # In production they would be database-backed (Postgres, Redis).
    orchestrator = build_orchestrator(
        model,
        fieldsat,
        weather_agent,
        farm_context_agent,
        knowledge_agent,
        advisory_agent,
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    return ChatService(orchestrator, get_tracer())


def get_chat_service(request: Request) -> ChatService:
    """Returns the chat service the lifespan built at startup."""
    return request.app.state.chat_service
