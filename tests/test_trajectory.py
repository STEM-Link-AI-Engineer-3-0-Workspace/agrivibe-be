"""Tier 2, trajectory tests (Lesson 3): with scripted model turns, check the PATH the orchestrator took.

The model's decisions are scripted, so these tests prove the graph wires the path correctly and that our
checks catch a bad path. They cannot prove a real model would choose this path: that is the real-model evals' job.
"""

from itertools import pairwise

from langchain_core.messages import AIMessage

from tests.test_chat import calls, chat, use_orchestrator

MINNERIYA = "SL-PADDY-MINNERIYA"
EXPECTED = ["list_fields", "save_requirements", "ask_weather_agent", "ask_advisory_agent"]
MAX_STEPS = 6  # orchestrator model calls for one decision question

FERTILISER_SCRIPT = {
    "orchestrator": [
        calls(("list_fields", {})),
        calls(("save_requirements", {"field_id": MINNERIYA, "crop": "paddy", "problem": "fertilise",
                                     "timeframe": "this week"})),
        calls(("ask_weather_agent", {"task": "Rain last week and next 48 hours"})),
        calls(("ask_advisory_agent", {"decision": "Should the farmer apply fertiliser this week?"})),
    ],
    "weather": [
        calls(("get_weather_context", {"field_id": MINNERIYA})),
        AIMessage(content="46 mm of rain expected in the next 48 hours."),
    ],
    "advisory": [AIMessage(content="Wait. Heavy rain is expected (weather), which would wash the fertiliser away.")],
}


def tool_path(body: dict) -> list[str]:
    """The orchestrator's tool calls for this turn, in order."""
    return body["tools_called"]


def no_repeats(path: list[str]) -> bool:
    """True when no tool was called twice in a row: the cheapest loop detector there is."""
    return all(a != b for a, b in pairwise(path))


def test_fertiliser_question_follows_the_expected_path(client, vector_store):
    _, models = use_orchestrator(vector_store, **FERTILISER_SCRIPT)

    body = chat(client, f"Should I fertilise {MINNERIYA} this week?")

    assert tool_path(body) == EXPECTED
    assert no_repeats(tool_path(body))
    assert models["orchestrator"].calls <= MAX_STEPS


def test_the_advisory_agent_is_last_and_its_answer_is_the_reply(client, vector_store):
    use_orchestrator(vector_store, **FERTILISER_SCRIPT)

    body = chat(client, f"Should I fertilise {MINNERIYA} this week?")

    assert tool_path(body)[-1] == "ask_advisory_agent"
    assert body["answer"].startswith("Wait. Heavy rain is expected")


def test_no_repeats_catches_a_loop(client, vector_store):
    # A confused orchestrator asks the Weather Agent three times in a row.
    use_orchestrator(
        vector_store,
        orchestrator=[
            calls(("save_requirements", {"field_id": MINNERIYA, "crop": "paddy", "problem": "rain",
                                         "timeframe": "now"})),
            *[calls(("ask_weather_agent", {"task": "Rain?"})) for _ in range(3)],
            AIMessage(content="It rained."),
        ],
        weather=[m for _ in range(3) for m in (calls(("get_weather_context", {"field_id": MINNERIYA})),
                                                AIMessage(content="Rain."))],
    )

    body = chat(client, "Has it rained?")

    assert tool_path(body).count("ask_weather_agent") == 3
    assert not no_repeats(tool_path(body))
