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
from agrivibe.agents.advisory.prompts import SYSTEM_PROMPT as ADVISORY_PROMPT
from agrivibe.agents.farm_context.prompts import SYSTEM_PROMPT as FARM_PROMPT
from agrivibe.agents.fieldoracle.prompts import SYSTEM_PROMPT as KNOWLEDGE_PROMPT
from agrivibe.agents.orchestrator.prompts import SYSTEM_PROMPT as ORCHESTRATOR_PROMPT
from agrivibe.agents.weather.prompts import SYSTEM_PROMPT as WEATHER_PROMPT
from agrivibe.config import Settings
from agrivibe.core.guardrails import (
    advisory_guardrails,
    openai_moderation,
    orchestrator_guardrails,
    specialist_guardrails,
)
from agrivibe.core.llm import create_chat_model
from agrivibe.core.mcp import FieldSat
from agrivibe.core.reranker import Reranker
from agrivibe.core.tracing import Observability, create_observability, create_tracer, prompt_tracing
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
def get_observability() -> Observability | None:
    """Returns the shared Langfuse helper for per-turn traces and scores, or None without Langfuse keys."""
    return create_observability(get_settings())


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
        middleware=prompt_tracing(settings, "agrivibe-knowledge", KNOWLEDGE_PROMPT),
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
    # Lesson 3: the guardrails each agent needs (core/guardrails.py). GUARDRAILS=false leaves only the error counter.
    specialist_guards = specialist_guardrails(settings)
    weather_agent = build_weather_agent(
        model, fieldsat, [*specialist_guards, *prompt_tracing(settings, "agrivibe-weather", fieldsat.prompt(WEATHER_PROMPT))]
    )
    farm_context_agent = build_farm_context_agent(
        model, fieldsat, [*specialist_guards, *prompt_tracing(settings, "agrivibe-farm-context", fieldsat.prompt(FARM_PROMPT))]
    )
    # A second FieldOracle, without a checkpointer: as a specialist it starts fresh on every call.
    # It searches the same way /ask does: vector search, then the cross-encoder.
    knowledge_agent = build_fieldoracle_agent(
        model, get_vector_store(), k=settings.search_k, reranker=get_reranker(),
        middleware=prompt_tracing(settings, "agrivibe-knowledge", KNOWLEDGE_PROMPT),
    )
    advisory_agent = build_advisory_agent(
        model, [*advisory_guardrails(settings), *prompt_tracing(settings, "agrivibe-advisory", ADVISORY_PROMPT)]
    )
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
        guardrails=[*orchestrator_guardrails(
            settings,
            openai_moderation(settings.openai_api_key.get_secret_value(), settings.moderation_model)
            if settings.moderation else None,
        ), *prompt_tracing(settings, "agrivibe-orchestrator", fieldsat.prompt(ORCHESTRATOR_PROMPT))],
    )
    tags = [f"guardrails:{'on' if settings.guardrails else 'off'}"]
    return ChatService(orchestrator, observability=get_observability(), tags=tags)


def get_chat_service(request: Request) -> ChatService:
    """Returns the chat service the lifespan built at startup."""
    return request.app.state.chat_service
