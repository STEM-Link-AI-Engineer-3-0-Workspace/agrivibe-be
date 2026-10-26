"""What the Weather and Farm Context agents share: the field they work on, and the report they return.

The model never decides which field it reads, and never copies a number. The field comes
from the runtime context, and the numbers come from FieldSat's own reply. The model only
writes the summary.
"""

from dataclasses import dataclass
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel


@dataclass
class FieldContext:
    """The field to report on. Pass it with every run: `ainvoke(..., context=FieldContext(field_id))`."""

    field_id: str


class FieldReport(BaseModel):
    """A specialist's answer, found under `structured_response`."""

    field_id: str  # from the context
    summary: str  # the agent's last message, in its own words
    data: dict[str, Any] | None  # FieldSat's reply, exactly as the server sent it. None if the tool failed.


class FieldSatReport(AgentMiddleware[AgentState, FieldContext]):
    """Pins every FieldSat call to the context's field, then builds the FieldReport."""

    async def awrap_tool_call(self, request, handler):
        """Runs the tool on the context's field, whatever field_id the model wrote."""
        context = cast(FieldContext, request.runtime.context)  # the type checker cannot see it from here
        args = {**request.tool_call["args"], "field_id": context.field_id}
        return await handler(request.override(tool_call={**request.tool_call, "args": args}))

    def after_agent(self, state: AgentState, runtime: Runtime[FieldContext]) -> dict:
        """Builds the FieldReport once the agent has finished."""
        replies = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.artifact]
        report = FieldReport(
            field_id=runtime.context.field_id,
            summary=state["messages"][-1].text,
            data=replies[-1].artifact["structured_content"] if replies else None,
        )
        return {"structured_response": report}
