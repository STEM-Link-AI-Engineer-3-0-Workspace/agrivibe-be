"""Every guardrail gets a deterministic test (Lesson 3). A guardrail nobody tests is a guess."""

from langchain_core.messages import AIMessage

from agrivibe.config import Settings
from agrivibe.core.guardrails import REFUSAL, redact_phones
from tests.test_chat import calls, chat, use_orchestrator

MINNERIYA = "SL-PADDY-MINNERIYA"
ON = Settings(_env_file=None, guardrails=True)


async def never_flagged(text: str) -> bool:
    return False


async def always_flagged(text: str) -> bool:
    return True


async def moderation_down(text: str) -> bool:
    raise ConnectionError("moderation API unreachable")


def test_phone_number_is_redacted_before_the_model_sees_it(client, vector_store):
    _, models = use_orchestrator(vector_store, settings=ON, moderate=never_flagged,
                                 orchestrator=[AIMessage(content="Which field?")])

    chat(client, "My number is 077 123 4567, call me about my field")

    sent = models["orchestrator"].received[0][-1].text
    assert "077 123 4567" not in sent
    assert "REDACTED" in sent


def test_traces_never_store_a_phone_number():
    # The same rule is Langfuse's mask (core/tracing.py), applied to everything sent to Langfuse.
    data = {"input": "call +94 77 123 4567", "messages": [{"content": "or 0771234567"}]}

    masked = redact_phones(data)

    assert masked == {"input": "call [REDACTED_PHONE]", "messages": [{"content": "or [REDACTED_PHONE]"}]}


def test_flagged_input_is_refused_before_any_model_call(client, vector_store):
    _, models = use_orchestrator(vector_store, settings=ON, moderate=always_flagged,
                                 orchestrator=[AIMessage(content="never sent")])

    body = chat(client, "something abusive")

    assert body["answer"] == REFUSAL
    assert models["orchestrator"].calls == 0
    assert body["guardrails"] == ["input_flagged"]


def test_moderation_outage_fails_open_and_is_recorded(client, vector_store):
    use_orchestrator(vector_store, settings=ON, moderate=moderation_down,
                     orchestrator=[AIMessage(content="Hello, which field?")])

    body = chat(client, "Hello")

    assert body["answer"] == "Hello, which field?"
    assert body["guardrails"] == ["moderation_unavailable"]


def test_a_looping_orchestrator_is_stopped_by_the_call_limit(client, vector_store):
    limited = Settings(_env_file=None, guardrails=True, max_model_calls=4)
    _, models = use_orchestrator(vector_store, settings=limited, moderate=never_flagged,
                                 orchestrator=[calls(("list_fields", {})) for _ in range(20)])

    chat(client, "Which fields do I have?")

    assert models["orchestrator"].calls == 4


def advisory_says(text: str) -> dict:
    return {
        "orchestrator": [
            calls(("save_requirements", {"field_id": MINNERIYA, "crop": "paddy", "problem": "fertilise",
                                         "timeframe": "now"})),
            calls(("ask_advisory_agent", {"decision": "How much urea?"})),
        ],
        "advisory": [AIMessage(content=text)],
    }


def test_an_out_of_range_dose_is_withheld(client, vector_store):
    use_orchestrator(vector_store, settings=ON, moderate=never_flagged,
                     **advisory_says("Apply 300 kg/ha of urea today."))

    body = chat(client, "How much urea should I apply?")

    assert body["answer"].startswith("I can't recommend that dose: 300 kg/ha")
    assert body["guardrails"] == ["dose_out_of_range"]


def test_a_normal_dose_passes(client, vector_store):
    use_orchestrator(vector_store, settings=ON, moderate=never_flagged,
                     **advisory_says("Apply 50 kg/ha of urea after the rain."))

    body = chat(client, "How much urea should I apply?")

    assert body["answer"] == "Apply 50 kg/ha of urea after the rain."
    assert body["guardrails"] == []


def test_guardrails_off_means_no_guardrails(client, vector_store):
    use_orchestrator(vector_store, settings=Settings(_env_file=None, guardrails=False),
                     **advisory_says("Apply 300 kg/ha of urea today."))

    body = chat(client, "How much urea should I apply?")

    assert body["answer"] == "Apply 300 kg/ha of urea today."
