from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage

from agrivibe.agents.orchestrator.memory import FarmerContext
from agrivibe.models.chat import ChatResponse
from agrivibe.service.answering import current_turn


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
    """Runs the farmer's messages through the orchestrator."""

    def __init__(self, orchestrator):
        """Initializes the service.

        Args:
            orchestrator: The agent built by `build_orchestrator`.
        """
        self.orchestrator = orchestrator

    async def chat(self, message: str, user_id: str, thread_id: str | None = None) -> ChatResponse:
        """Sends one message to the orchestrator and returns its reply.

        Args:
            message: What the farmer wrote.
            user_id: Who wrote it. Keys the farmer's long-term memory.
            thread_id: The conversation to continue. A new one is started when empty.

        Returns:
            The reply, the tools the orchestrator called for it, and the saved requirements
            and findings.
        """
        thread_id = thread_id or str(uuid4())
        result = await self.orchestrator.ainvoke(
            # The empty findings clear the last question's evidence (see merge_findings).
            {"messages": [{"role": "user", "content": message}], "findings": {}},
            {"configurable": {"thread_id": thread_id}},
            context=FarmerContext(user_id=user_id),
        )
        return ChatResponse(
            message=message,
            user_id=user_id,
            thread_id=thread_id,
            answer=result["messages"][-1].text,
            tools_called=tools_called(current_turn(result["messages"])),
            requirements=result.get("requirements"),
            findings=result["findings"],
        )
