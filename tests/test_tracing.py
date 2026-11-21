from langchain_core.callbacks import BaseCallbackHandler

from agrivibe.config import Settings
from agrivibe.core.tracing import create_tracer
from agrivibe.dependencies import get_chat_service
from agrivibe.main import app
from agrivibe.service.chatting import ChatService
from tests.test_chat import FIELD_STATE, chat, use_orchestrator


class RecordingTracer(BaseCallbackHandler):
    """Stands in for Langfuse's handler: records every run it is told about."""

    def __init__(self):
        self.runs = []

    def on_chain_start(self, serialized, inputs, *, metadata=None, **kwargs):
        self.runs.append((kwargs.get("name"), metadata or {}))

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        self.runs.append(("chat model", metadata or {}))


def test_no_langfuse_keys_means_no_tracer():
    # _env_file=None: ignore a local .env that has the keys.
    assert create_tracer(Settings(_env_file=None)) is None  # pyright: ignore[reportCallIssue]


def test_one_chat_is_one_trace_with_the_farmer_and_the_thread(client, vector_store):
    agent, _ = use_orchestrator(vector_store, **FIELD_STATE)
    tracer = RecordingTracer()
    service = ChatService(agent, tracer)
    app.dependency_overrides[get_chat_service] = lambda: service

    body = chat(client, "Is my Minneriya field stressed?", user_id="farmer-007")

    names = [name for name, _ in tracer.runs]
    assert names[0] == "chat"
    # The specialists run inside their tools, so their model calls reach the same tracer.
    assert names.count("chat model") == 7
    for _, metadata in tracer.runs:
        assert metadata["langfuse_session_id"] == body["thread_id"]
        assert metadata["langfuse_user_id"] == "farmer-007"


def test_prompt_link_is_rendered_before_each_model_with_the_same_parent():
    """Exercise a real agent graph: Langfuse links a template's parent to its next generation."""
    import asyncio
    from types import SimpleNamespace

    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage

    from agrivibe.core.tracing import PromptTracing
    from tests.fakes import ScriptedChatModel

    prompt = SimpleNamespace(prompt='Source prompt with literal JSON: {"field": "id"}')

    class Links(BaseCallbackHandler):
        def __init__(self):
            self.prompt_parents = set()
            self.linked = 0

        def on_chain_start(self, serialized, inputs, *, parent_run_id=None, metadata=None, **kwargs):
            if (metadata or {}).get("langfuse_prompt") is prompt:
                self.prompt_parents.add(parent_run_id)

        def on_chat_model_start(self, serialized, messages, *, parent_run_id=None, **kwargs):
            assert parent_run_id in self.prompt_parents
            self.linked += 1

    for asynchronous in (False, True):
        model = ScriptedChatModel(replies=[AIMessage(content="hello")])
        agent = create_agent(model, system_prompt="Source plus dynamic farmer memory", middleware=[PromptTracing(prompt)])
        tracer = Links()
        args = ({"messages": [{"role": "user", "content": "hello"}]}, {"callbacks": [tracer]})
        if asynchronous:
            asyncio.run(agent.ainvoke(*args))
        else:
            agent.invoke(*args)
        assert tracer.linked == 1
        assert model.received[0][0].content == "Source plus dynamic farmer memory"
