from collections import Counter
import logging

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langfuse.api import NotFoundError

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agrivibe.config import Settings
from agrivibe.core.guardrails import redact_phones


def create_langfuse(settings: Settings) -> Langfuse | None:
    """Creates the Langfuse client, or None when the Langfuse keys are not set.

    The keys are passed here because pydantic-settings reads `.env` without putting its values in
    the environment, where Langfuse would look for them.

    Two Lesson 3 settings: `environment` separates development traces from production ones, and
    `mask` runs on everything sent to Langfuse, so a phone number a farmer types is never stored.
    """
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        environment=settings.langfuse_environment,
        mask=lambda *, data, **_: redact_phones(data),
    )


def create_tracer(settings: Settings) -> CallbackHandler | None:
    """Creates the Langfuse callback handler, or None when the Langfuse keys are not set.

    This follows Langfuse's LangChain guide: https://langfuse.com/integrations/frameworks/langchain.

    Args:
        settings: Application settings with the Langfuse keys and URL.

    Returns:
        A handler to pass in a run's `callbacks`. Every model call, tool call and
        specialist run inside that run is sent to Langfuse as one trace.
    """
    if create_langfuse(settings) is None:  # registers the client the handler sends to
        return None
    return CallbackHandler(public_key=settings.langfuse_public_key)


class Observability:
    """One Langfuse trace per chat turn, with a known id, plus the scores we attach to it (Lesson 3).

    Knowing the trace id lets the API return it, so the frontend can send 👍/👎 for that exact trace,
    and lets the service attach guardrail and tool-error signals as scores you can filter and chart.
    """

    def __init__(self, client: Langfuse, public_key: str):
        self.client = client
        self.public_key = public_key

    def new_trace(self) -> tuple[str, CallbackHandler]:
        """A fresh trace id and a handler that writes this run into that trace."""
        trace_id = self.client.create_trace_id()
        return trace_id, CallbackHandler(public_key=self.public_key, trace_context={"trace_id": trace_id})

    def record_events(self, trace_id: str, events: list[tuple[str, str]]) -> None:
        """Turns the turn's guardrail events into scores: one categorical score per guardrail that fired,
        and a numeric `tool_errors` count on every trace (zero included, so the rate can be computed)."""
        counts = Counter(name for name, _ in events)
        self.client.create_score(trace_id=trace_id, name="tool_errors", value=float(counts.pop("tool_error", 0)))
        for name, detail in events:
            if name in counts:
                self.client.create_score(
                    trace_id=trace_id, name="guardrail", value=name, data_type="CATEGORICAL", comment=detail
                )

    def feedback(self, trace_id: str, positive: bool, comment: str | None = None) -> None:
        """Stores a farmer's 👍/👎 on a trace as the `user_feedback` score."""
        self.client.create_score(
            trace_id=trace_id, name="user_feedback", value=1.0 if positive else 0.0, data_type="BOOLEAN",
            comment=comment,
        )

    def trace_url(self, trace_id: str) -> str | None:
        return self.client.get_trace_url(trace_id=trace_id)


def create_observability(settings: Settings) -> Observability | None:
    """The Observability helper, or None when the Langfuse keys are not set."""
    client = create_langfuse(settings)
    if client is None or settings.langfuse_public_key is None:
        return None
    return Observability(client, settings.langfuse_public_key)


def run_config(
    name: str,
    thread_id: str,
    tracer: BaseCallbackHandler | None,
    user_id: str | None = None,
    tags: list[str] | None = None,
) -> RunnableConfig:
    """The config for one agent run: the thread to continue and, when tracing is on, Langfuse.

    Args:
        name: The trace's name in Langfuse, for example "chat".
        thread_id: The conversation. It is also the Langfuse session, so a chat's turns are grouped.
        tracer: A Langfuse handler, or None.
        user_id: The farmer, shown as the trace's user in Langfuse.
        tags: Trace tags, for example ["guardrails:on"], to filter by in Langfuse.

    Returns:
        The config to pass to `invoke` or `ainvoke`.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "run_name": name}
    if tracer is not None:
        config["callbacks"] = [tracer]
        config["metadata"] = {"langfuse_session_id": thread_id}
        if user_id:
            config["metadata"]["langfuse_user_id"] = user_id
        if tags:
            config["metadata"]["langfuse_tags"] = tags
    return config


class PromptTracing(AgentMiddleware):
    """Render a versioned source prompt beside each model call so Langfuse links it.

    The agent still uses its original system message, including dynamic farmer memory.
    A concrete SystemMessage keeps JSON braces in source prompts literal.
    """

    def __init__(self, prompt):
        self.template = ChatPromptTemplate(
            [SystemMessage(content=prompt.prompt)], metadata={"langfuse_prompt": prompt}
        )

    def wrap_model_call(self, request, handler):
        self.template.invoke({})
        return handler(request)

    async def awrap_model_call(self, request, handler):
        await self.template.ainvoke({})
        return await handler(request)


def prompt_tracing(settings: Settings, name: str, source: str) -> list[AgentMiddleware]:
    """Version the source-controlled prompt only when its content changes.

    This records what this checkout uses; editing a deployment label does not change
    application behaviour. Tracing remains optional if Langfuse is unavailable.
    """
    client = create_langfuse(settings)
    if client is None:
        return []
    try:
        try:
            prompt = client.get_prompt(name, label="latest", type="text")
        except NotFoundError:
            prompt = None
        if prompt is None or prompt.prompt != source:
            prompt = client.create_prompt(
                name=name, prompt=source, type="text",
                labels=[settings.langfuse_environment],
                commit_message="Synced from AgriVibe source at startup",
            )
        return [PromptTracing(prompt)]
    except Exception:
        logging.getLogger(__name__).warning("Could not register Langfuse prompt %s", name, exc_info=True)
        return []
