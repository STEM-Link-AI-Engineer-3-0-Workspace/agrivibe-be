"""What an eval runs: the real agents and the real model, in a frozen world (Lesson 3).

"Freeze the world, vary the model": FieldSat is the in-process fake from tests/fakes.py, with fixed numbers and
the POOR, SAT-FAIL and INJECT fields. When a score changes, the cause is the model or the prompt, not the data.
The knowledge base is Pinecone when available, or an empty in-memory store (CI's smoke set doesn't need it).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agrivibe.agents.advisory import build_advisory_agent
from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldoracle import build_fieldoracle_agent
from agrivibe.agents.orchestrator import build_orchestrator
from agrivibe.agents.weather import build_weather_agent
from agrivibe.config import Settings
from agrivibe.core.guardrails import (
    advisory_guardrails,
    openai_moderation,
    orchestrator_guardrails,
    specialist_guardrails,
)
from agrivibe.core.llm import create_chat_model
from agrivibe.service.chatting import ChatService
from tests.fakes import FakeVectorStore, load_fake_fieldsat

GOLDEN = Path(__file__).parent / "golden.jsonl"
MAX_STEPS = 10  # orchestrator tool calls in one turn


@dataclass
class Item:
    id: str
    input: str
    rubric: str
    trajectory: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    must_not_call: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def load_items(subset: str) -> list[Item]:
    """The golden items: 'smoke' (CI), 'full' (everything), or one item id."""
    items = [Item(**json.loads(line)) for line in GOLDEN.read_text().splitlines() if line.strip()]
    if subset == "full":
        return items
    return [i for i in items if subset in i.tags or i.id == subset]


async def build_service(settings: Settings, use_pinecone: bool) -> ChatService:
    """The production agent graph with the real model and production guardrails, in the frozen world."""
    model = create_chat_model(settings)
    fieldsat = await load_fake_fieldsat()
    if use_pinecone:
        from agrivibe.db.vectorstore import create_vector_store

        vector_store = create_vector_store(settings)
    else:
        vector_store = FakeVectorStore()
    guards = specialist_guardrails(settings)
    moderate = (openai_moderation(settings.openai_api_key.get_secret_value(), settings.moderation_model)
                if settings.moderation else None)
    orchestrator = build_orchestrator(
        model,
        fieldsat,
        build_weather_agent(model, fieldsat, guards),
        build_farm_context_agent(model, fieldsat, guards),
        build_fieldoracle_agent(model, vector_store, k=settings.search_k),
        build_advisory_agent(model, advisory_guardrails(settings)),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
        guardrails=orchestrator_guardrails(settings, moderate),
    )
    return ChatService(orchestrator)


def pinecone_available(settings: Settings) -> bool:
    key = settings.pinecone_api_key.get_secret_value()
    return bool(key) and key not in {"test-key", "unused"} and os.getenv("EVALS_NO_PINECONE") != "1"


def evidence_text(findings: dict) -> str:
    """What the specialists reported, for the judge to check claims against."""
    return "\n\n".join(f"{name}: {f.summary}\nData: {json.dumps(f.data)}" for name, f in findings.items())


def deterministic_checks(item: Item, answer: str, tools: list[str]) -> dict[str, tuple[bool, str]]:
    """The cheap checks, run before the judge. Each returns (passed, reason)."""
    checks = {}
    it = iter(tools)
    in_order = all(any(t == want for t in it) for want in item.trajectory)
    checks["trajectory"] = (in_order, f"expected {item.trajectory} in order, got {tools}")
    called = [t for t in item.must_not_call if t in tools]
    checks["no_forbidden_tools"] = (not called, f"called {called}" if called else "ok")
    found = [s for s in item.must_not if s.lower() in answer.lower()]
    checks["no_forbidden_text"] = (not found, f"answer contains {found}" if found else "ok")
    checks["step_budget"] = (len(tools) <= MAX_STEPS, f"{len(tools)} tool calls (max {MAX_STEPS})")
    return checks
