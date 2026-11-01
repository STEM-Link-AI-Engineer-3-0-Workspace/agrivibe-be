import time
from datetime import UTC, datetime
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agrivibe.agents.orchestrator.memory import FarmerContext
from agrivibe.core.tracing import run_config
from agrivibe.models.chat import ChatResponse, Turn
from agrivibe.service.threads import ThreadService


def current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drops the earlier turns of a thread, keeping the latest message onwards.

    Args:
        messages: The thread's full message history after a run.

    Returns:
        The messages from the last user message to the end.
    """
    starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    return messages[starts[-1]:] if starts else messages


def tools_called(messages: list[BaseMessage]) -> list[str]:
    """Lists the tools the orchestrator called, in order.

    Args:
        messages: The orchestrator's messages for one turn.

    Returns:
        The tool names. A specialist's own tool calls are not here: they stay inside
        the specialist's run.
    """
    return [call["name"] for message in messages if isinstance(message, AIMessage) for call in message.tool_calls]


class ChatService:
    """Runs the farmer's messages through the orchestrator and records each turn."""

    def __init__(self, orchestrator, threads: ThreadService, tracer: BaseCallbackHandler | None = None):
        """Initializes the service.

        Args:
            orchestrator: The agent built by `build_orchestrator`.
            threads: Records each turn and checks that a thread is the farmer's.
            tracer: The Langfuse handler from `create_tracer`, or None to run without tracing.
        """
        self.orchestrator = orchestrator
        self.threads = threads
        self.tracer = tracer

    async def chat(self, message: str, farmer_id: str, thread_id: str | None = None) -> ChatResponse:
        """Sends one message to the orchestrator, records the turn and returns it.

        Args:
            message: What the farmer wrote.
            farmer_id: Who wrote it, as decided by the server. Keys the farmer's threads and memory.
            thread_id: One of the farmer's conversations to continue. A new one is started when empty.

        Returns:
            The reply, the tools the orchestrator called for it, the saved requirements and
            the specialists' findings.

        Raises:
            ThreadNotFound: thread_id is not one of this farmer's conversations.
        """
        if thread_id:
            await self.threads.summary(farmer_id, thread_id)  # someone else's thread is never continued
        else:
            thread_id = str(uuid4())

        started_at, started = datetime.now(UTC), time.monotonic()
        result = await self.orchestrator.ainvoke(
            # The empty findings clear the last question's evidence (see merge_findings).
            {"messages": [{"role": "user", "content": message}], "findings": {}},
            run_config("chat", thread_id, self.tracer, farmer_id),
            context=FarmerContext(user_id=farmer_id),
        )
        turn = Turn(
            message=message,
            answer=result["messages"][-1].text,
            tools_called=tools_called(current_turn(result["messages"])),
            requirements=result.get("requirements"),
            findings=result["findings"],
            started_at=started_at,
            seconds=round(time.monotonic() - started, 1),
        )
        await self.threads.record_turn(farmer_id, thread_id, turn)
        return ChatResponse(**turn.model_dump(), thread_id=thread_id)
