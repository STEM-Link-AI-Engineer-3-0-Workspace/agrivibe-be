"""Tier 2, control tests (Lesson 3): the agent can only do what we allow.

Least privilege as a test: adding a tool becomes a reviewed decision. The fence test checks that untrusted tool
output is labelled and cleaned before the Advisory Agent reads it. It cannot check whether a real model would
have obeyed the injected text; the INJECT item in evals/golden.jsonl does that.
"""

from langchain_core.messages import AIMessage, ToolMessage

from agrivibe.config import Settings
from tests.fakes import INJECTED_NOTE
from tests.test_chat import calls, chat, use_orchestrator

ALLOWED = {"list_fields", "save_requirements", "ask_weather_agent",
           "ask_farm_context_agent", "ask_knowledge_agent", "ask_advisory_agent"}
INJECT = "SL-PADDY-INJECT"


def orchestrator_tool_names(agent) -> set[str]:
    return set(agent.nodes["tools"].bound.tools_by_name)


def test_orchestrator_only_has_allowed_tools(vector_store):
    agent, _ = use_orchestrator(vector_store)

    assert orchestrator_tool_names(agent) == ALLOWED


def test_unknown_tool_call_is_rejected_not_executed(client, vector_store):
    # The scripted model "hallucinates" a tool that does not exist.
    agent, _ = use_orchestrator(
        vector_store,
        orchestrator=[calls(("delete_field", {"field_id": "SL-PADDY-MINNERIYA"})),
                      AIMessage(content="I can't delete fields.")],
    )

    body = chat(client, "Delete my field")

    messages = agent.get_state({"configurable": {"thread_id": body["thread_id"]}}).values["messages"]
    replies = [m for m in messages if isinstance(m, ToolMessage) and m.name == "delete_field"]
    assert [m.status for m in replies] == ["error"]  # the tool node refused it; nothing ran
    assert body["answer"] == "I can't delete fields."


INJECT_SCRIPT = {
    "orchestrator": [
        calls(("save_requirements", {"field_id": INJECT, "crop": "paddy", "problem": "fertilise",
                                     "timeframe": "this week"})),
        calls(("ask_farm_context_agent", {"task": "NDVI trend"})),
        calls(("ask_advisory_agent", {"decision": "Should the farmer fertilise this week?"})),
    ],
    "farm_context": [
        calls(("get_satellite_analysis", {"field_id": INJECT, "comparison_days": 14})),
        AIMessage(content="NDVI fell 15.2%."),
    ],
    "advisory": [AIMessage(content="Wait for the rain to pass.")],
}


def test_tool_output_is_passed_as_data(client, vector_store):
    _, models = use_orchestrator(vector_store, settings=Settings(_env_file=None, guardrails=True), **INJECT_SCRIPT)

    body = chat(client, f"Should I fertilise {INJECT}?")

    # The specialist's model sees the FieldSat reply labelled as untrusted data...
    tool_reply = models["farm_context"].received[1][-1].text
    assert tool_reply.startswith("UNTRUSTED TOOL OUTPUT")
    # ...and the instruction never reaches the Advisory Agent, which writes the farmer's answer.
    evidence = models["advisory"].received[0][-1].text
    assert "SYSTEM NOTICE" not in evidence
    assert "removed by guardrail" in evidence
    assert "tool_output_injection" in body["guardrails"]


def test_without_the_fence_the_injection_reaches_the_advisory_agent(client, vector_store):
    # The threat, shown: with guardrails off, the provider's note goes straight into the Advisory Agent's input.
    _, models = use_orchestrator(vector_store, settings=Settings(_env_file=None, guardrails=False), **INJECT_SCRIPT)

    chat(client, f"Should I fertilise {INJECT}?")

    assert INJECTED_NOTE in models["advisory"].received[0][-1].text
