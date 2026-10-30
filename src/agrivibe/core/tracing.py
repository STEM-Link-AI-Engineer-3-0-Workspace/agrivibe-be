from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from agrivibe.config import Settings


def create_tracer(settings: Settings) -> CallbackHandler | None:
    """Creates the Langfuse callback handler, or None when the Langfuse keys are not set.

    This follows Langfuse's LangChain guide: https://langfuse.com/integrations/frameworks/langchain.
    The keys are passed to the client here because pydantic-settings reads `.env`
    without putting its values in the environment, where Langfuse would look for them.

    Args:
        settings: Application settings with the Langfuse keys and URL.

    Returns:
        A handler to pass in a run's `callbacks`. Every model call, tool call and
        specialist run inside that run is sent to Langfuse as one trace.
    """
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    # Registers the client that the handler sends to. It flushes on its own when the app exits.
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )
    return CallbackHandler(public_key=settings.langfuse_public_key)


def run_config(
    name: str, thread_id: str, tracer: BaseCallbackHandler | None, user_id: str | None = None
) -> RunnableConfig:
    """The config for one agent run: the thread to continue and, when tracing is on, Langfuse.

    Args:
        name: The trace's name in Langfuse, for example "chat".
        thread_id: The conversation. It is also the Langfuse session, so a chat's turns are grouped.
        tracer: The handler from `create_tracer`, or None.
        user_id: The farmer, shown as the trace's user in Langfuse.

    Returns:
        The config to pass to `invoke` or `ainvoke`.
    """
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "run_name": name}
    if tracer is not None:
        config["callbacks"] = [tracer]
        config["metadata"] = {"langfuse_session_id": thread_id}
        if user_id:
            config["metadata"]["langfuse_user_id"] = user_id
    return config
