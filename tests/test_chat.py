import asyncio

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from agrivibe.agents.advisory import build_advisory_agent
from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldoracle import build_fieldoracle_agent
from agrivibe.agents.orchestrator import build_orchestrator
from agrivibe.agents.weather import build_weather_agent
from agrivibe.dependencies import get_chat_service
from agrivibe.main import app
from agrivibe.service.chatting import ChatService
from tests.conftest import upload
from tests.fakes import ScriptedChatModel, load_fake_fieldsat

MINNERIYA = "SL-PADDY-MINNERIYA"
REQUIREMENTS = {"field_id": MINNERIYA, "crop": "paddy", "problem": "is the field stressed", "timeframe": "now"}

def calls(*tool_calls):
    """One model turn that calls one or more tools. Several calls run in parallel."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"{name}-{i}"} for i, (name, args) in enumerate(tool_calls)],
    )


def use_orchestrator(vector_store, **replies):
    """Builds the real agents, each with a scripted model, and puts them behind POST /chat.

    Returns the orchestrator and the scripted models by name, so a test can read what
    each model was sent.
    """
    names = ["orchestrator", "weather", "farm_context", "knowledge", "advisory"]
    models = {name: ScriptedChatModel(replies=list(replies.get(name, []))) for name in names}
    fieldsat = asyncio.run(load_fake_fieldsat())
    agent = build_orchestrator(
        models["orchestrator"],
        fieldsat,
        build_weather_agent(models["weather"], fieldsat),
        build_farm_context_agent(models["farm_context"], fieldsat),
        build_fieldoracle_agent(models["knowledge"], vector_store),
        build_advisory_agent(models["advisory"]),
        checkpointer=InMemorySaver(),
        store=InMemoryStore(),
    )
    service = ChatService(agent)
    app.dependency_overrides[get_chat_service] = lambda: service
    return agent, models


def chat(client, message, user_id="farmer-001", thread_id=None):
    return client.post("/chat", json={"message": message, "user_id": user_id, "thread_id": thread_id}).json()


FIELD_STATE = {
    "orchestrator": [
        calls(("save_requirements", REQUIREMENTS)),
        calls(("ask_farm_context_agent", {"task": "NDVI trend over 14 days"}),
              ("ask_weather_agent", {"task": "Rain last week and next 48 hours"})),
        AIMessage(content="Your field got less green in the last two weeks, and more heavy rain is coming."),
    ],
    "farm_context": [
        calls(("get_satellite_analysis", {"field_id": MINNERIYA, "comparison_days": 14})),
        AIMessage(content="NDVI fell 15.2% in 14 days."),
    ],
    "weather": [
        calls(("get_weather_context", {"field_id": MINNERIYA})),
        AIMessage(content="46 mm of rain expected in 48 hours."),
    ],
}


def test_field_state_question_calls_two_specialists_in_parallel(client, vector_store):
    use_orchestrator(vector_store, **FIELD_STATE)

    body = chat(client, "Is my Minneriya field stressed?")

    assert body["tools_called"] == ["save_requirements", "ask_farm_context_agent", "ask_weather_agent"]
    assert body["answer"].startswith("Your field got less green")


def test_knowledge_question_returns_fieldoracle_citations(client, vector_store):
    upload(client, "fertiliser_subsidy_circular_2026.pdf", "policy")
    first = next(iter(vector_store.store))
    use_orchestrator(
        vector_store,
        orchestrator=[
            calls(("ask_knowledge_agent", {"question": "When do fertiliser subsidy claims close?"})),
            AIMessage(content=f"Claims close on 15 November 2026. [{first}]"),
        ],
        knowledge=[
            calls(("search_knowledge", {"query": "subsidy claim deadline", "category": "policy"})),
            calls(("CitedAnswer", {"found": True, "sentences": [
                {"sentence": "Claims close on 15 November 2026.", "citations": [first]},
            ]})),
        ],
    )

    body = chat(client, "When do subsidy claims close?")

    assert body["tools_called"] == ["ask_knowledge_agent"]
    assert body["findings"] == {
        "knowledge": {
            "summary": f"Claims close on 15 November 2026. [{first}]",
            "data": {"found": True, "citations": [first]},
        }
    }


def test_a_failing_specialist_does_not_end_the_chat(client, vector_store):
    use_orchestrator(
        vector_store,
        orchestrator=[
            calls(("save_requirements", REQUIREMENTS)),
            calls(("ask_weather_agent", {"task": "Rain last week"})),
            AIMessage(content="The weather data is unavailable right now."),
        ],
        weather=[],  # the scripted model has no replies, so the Weather Agent raises
    )

    body = chat(client, "Has it rained on my Minneriya field?")

    assert body["answer"] == "The weather data is unavailable right now."
    assert body["findings"]["weather"]["summary"].startswith("This agent failed")
    assert body["findings"]["weather"]["data"] == {}


def test_follow_up_keeps_the_conversation(client, vector_store):
    agent, _ = use_orchestrator(
        vector_store,
        orchestrator=[
            calls(("list_fields", {})),
            AIMessage(content="Which field? Minneriya Paddy Fields is the one I know."),
            AIMessage(content="Thanks, I will use SL-PADDY-MINNERIYA."),
        ],
    )

    first = chat(client, "Is my field stressed?")
    second = chat(client, "Minneriya", thread_id=first["thread_id"])

    assert first["tools_called"] == ["list_fields"]
    assert second["tools_called"] == []
    messages = agent.get_state({"configurable": {"thread_id": first["thread_id"]}}).values["messages"]
    assert [m.text for m in messages if m.type == "human"] == ["Is my field stressed?", "Minneriya"]


def test_a_specialist_needs_a_saved_field(client, vector_store):
    _, models = use_orchestrator(
        vector_store,
        orchestrator=[
            calls(("ask_weather_agent", {"task": "Rain last week"})),
            AIMessage(content="Which field do you mean?"),
        ],
    )

    body = chat(client, "Has it rained?")

    # The field comes from the saved requirements, so without one the specialist never runs.
    assert body["findings"]["weather"]["summary"].startswith("No field is saved yet")
    assert models["weather"].calls == 0


# Central memory


def test_specialists_read_the_requirements_and_write_findings(client, vector_store):
    _, models = use_orchestrator(vector_store, **FIELD_STATE)

    body = chat(client, "Is my Minneriya field stressed?")

    assert body["requirements"] == REQUIREMENTS
    # Each finding is the specialist's summary, plus FieldSat's reply as data.
    assert body["findings"]["weather"]["summary"] == "46 mm of rain expected in 48 hours."
    assert body["findings"]["weather"]["data"]["rainfall_last_7_days_mm"] == 112.0
    assert body["findings"]["farm_context"]["summary"] == "NDVI fell 15.2% in 14 days."
    assert body["findings"]["farm_context"]["data"]["field_id"] == MINNERIYA
    # The Weather Agent saw the saved requirements above the task, not the farmer's chat.
    task = models["weather"].received[0][-1].text
    assert task.startswith(f"Field: {MINNERIYA} (paddy)")
    assert task.endswith("Rain last week and next 48 hours")


def test_a_new_message_clears_findings_but_keeps_requirements(client, vector_store):
    replies = dict(FIELD_STATE)
    replies["orchestrator"] = FIELD_STATE["orchestrator"] + [
        calls(("ask_knowledge_agent", {"question": "When is the first top dressing due?"})),
        AIMessage(content="The knowledge base does not say."),
    ]
    replies["knowledge"] = [
        calls(("search_knowledge", {"query": "first top dressing"})),
        calls(("CitedAnswer", {"found": False, "sentences": []})),
    ]
    use_orchestrator(vector_store, **replies)

    first = chat(client, "Is my Minneriya field stressed?")
    second = chat(client, "When is the first top dressing due?", thread_id=first["thread_id"])

    assert second["findings"] == {
        "knowledge": {"summary": "The knowledge base does not cover this.", "data": {"found": False, "citations": []}}
    }
    assert second["requirements"] == REQUIREMENTS


def test_the_farm_is_remembered_in_a_new_conversation(client, vector_store):
    replies = dict(FIELD_STATE)
    replies["orchestrator"] = FIELD_STATE["orchestrator"] + [
        AIMessage(content="Hello again."),
        AIMessage(content="Hello."),
    ]
    _, models = use_orchestrator(vector_store, **replies)

    chat(client, "Is my Minneriya field stressed?", user_id="farmer-001")
    chat(client, "Hi", user_id="farmer-001")  # a new thread, same farmer
    chat(client, "Hi", user_id="farmer-002")  # a new thread, another farmer

    same_farmer = models["orchestrator"].received[-2][0].text  # the system prompt of each call
    other_farmer = models["orchestrator"].received[-1][0].text
    assert f"This farmer's saved field is {MINNERIYA} (paddy)" in same_farmer
    assert "This farmer's saved field" not in other_farmer


# The Advisory Agent


def test_a_decision_ends_with_the_advisory_agent(client, vector_store):
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")
    passage = next(iter(vector_store.store))
    _, models = use_orchestrator(
        vector_store,
        orchestrator=[
            calls(("save_requirements", {**REQUIREMENTS, "problem": "spray fungicide", "timeframe": "this week"})),
            calls(("ask_farm_context_agent", {"task": "NDVI trend over 14 days"}),
                  ("ask_weather_agent", {"task": "Rain last week and next 48 hours"}),
                  ("ask_knowledge_agent", {"question": "When should paddy be sprayed?"})),
            calls(("ask_advisory_agent", {"decision": "Should the farmer spray fungicide this week?"})),
        ],
        farm_context=FIELD_STATE["farm_context"],
        weather=FIELD_STATE["weather"],
        knowledge=[
            calls(("search_knowledge", {"query": "spraying paddy rain"})),
            calls(("CitedAnswer", {"found": True, "sentences": [
                {"sentence": "Do not spray before rain.", "citations": [passage]},
            ]})),
        ],
        advisory=[AIMessage(content=f"Wait. Heavy rain is expected (weather) and spray washes off [{passage}].")],
    )

    body = chat(client, "Should I spray my Minneriya field this week?")

    assert body["tools_called"][-1] == "ask_advisory_agent"
    # return_direct: the Advisory Agent's answer is the reply, with no extra orchestrator turn.
    assert body["answer"] == f"Wait. Heavy rain is expected (weather) and spray washes off [{passage}]."
    assert models["orchestrator"].calls == 3
    # The Advisory Agent was sent the decision and all three findings from central memory.
    sent = models["advisory"].received[0][-1].text
    assert "Decision to make: Should the farmer spray fungicide this week?" in sent
    # Each finding arrives as its summary, then its data as JSON.
    assert "farm_context:\nNDVI fell 15.2% in 14 days.\nData: {" in sent
    assert '"ndvi_change_percentage": -15.2' in sent
    assert "weather:\n46 mm of rain expected in 48 hours.\nData: {" in sent
    assert '"heavy_rain_expected": true' in sent
    assert f"knowledge:\nDo not spray before rain. [{passage}]" in sent
