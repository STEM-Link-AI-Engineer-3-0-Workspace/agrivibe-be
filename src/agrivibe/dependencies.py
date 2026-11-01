"""Wiring. Each function builds one object; FastAPI injects them into routes.

Everything is created once and reused (lru_cache). Tests replace these
functions with fakes through app.dependency_overrides.
"""

from functools import lru_cache

from fastapi import Request
from langchain_pinecone import PineconeVectorStore
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

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
from agrivibe.models.farmer import Farmer
from agrivibe.service.chatting import ChatService
from agrivibe.service.knowledge import KnowledgeService
from agrivibe.service.threads import ThreadService


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


def current_farmer() -> Farmer:
    """Returns the farmer making the request. Every route that needs to know who is asking depends on this.

    This is the auth seam. Today it returns the farmer named in .env (FARMER_ID, FARMER_NAME).
    A real app resolves the caller here instead, from a session cookie or a token checked with an
    auth service, and no other code changes. Identity never comes from the request body.
    """
    settings = get_settings()
    return Farmer(id=settings.farmer_id, name=settings.farmer_name)


def build_chat_service(
    settings: Settings, fieldsat: FieldSat, checkpointer: BaseCheckpointSaver, store: BaseStore, threads: ThreadService
) -> ChatService:
    """Builds the specialists and the orchestrator that calls them.

    Called once, from the lifespan in main.py: the FieldSat tools are only known after
    the app has asked the server for them, and Postgres is opened there.

    Args:
        settings: Application settings.
        fieldsat: The tools and instructions loaded from the FieldSat MCP server.
        checkpointer: Saves each thread's state, in Postgres.
        store: Each farmer's long-term memory, thread index and turns, in Postgres.
        threads: Records each turn. Shares the store.

    Returns:
        The service behind POST /chat.
    """
    model = create_chat_model(settings)
    weather_agent = build_weather_agent(model, fieldsat)
    farm_context_agent = build_farm_context_agent(model, fieldsat)
    # FieldOracle has no checkpointer: as a specialist it starts fresh on every call.
    # It searches the same way POST /search does: vector search, then the cross-encoder.
    knowledge_agent = build_fieldoracle_agent(model, get_vector_store(), k=settings.search_k, reranker=get_reranker())
    advisory_agent = build_advisory_agent(model)
    orchestrator = build_orchestrator(
        model,
        fieldsat,
        weather_agent,
        farm_context_agent,
        knowledge_agent,
        advisory_agent,
        checkpointer=checkpointer,
        store=store,
    )
    return ChatService(orchestrator, threads, get_tracer())


def get_chat_service(request: Request) -> ChatService:
    """Returns the chat service the lifespan built at startup."""
    return request.app.state.chat_service


def get_thread_service(request: Request) -> ThreadService:
    """Returns the thread service the lifespan built at startup."""
    return request.app.state.thread_service
