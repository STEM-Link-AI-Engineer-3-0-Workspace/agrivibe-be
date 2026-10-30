"""Bodies for POST /chat."""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """A message from the farmer to AgriVibe."""

    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(
        min_length=1,
        max_length=100,
        description="Who is asking. A real app takes this from the login; here the client sends it.",
        examples=["farmer-001"],
    )
    thread_id: str | None = Field(
        default=None,
        max_length=200,
        description="The conversation to continue. Leave empty to start a new one; the response returns its id.",
    )


class Finding(BaseModel):
    """What one specialist reported for this message."""

    summary: str = Field(description="The specialist's report in a few sentences. Knowledge cites passage ids in brackets.")
    data: dict[str, Any] = Field(
        description=(
            "weather and farm_context: FieldSat's reply as the server sent it, such as rainfall_last_7_days_mm "
            "or ndvi_change_percentage. knowledge: found and citations. Empty if the specialist or its tool failed."
        )
    )


class ChatResponse(BaseModel):
    """AgriVibe's reply, the tools the orchestrator called, and what central memory holds."""

    message: str
    user_id: str
    thread_id: str = Field(description="The conversation this reply belongs to. Send it back to continue.")
    answer: str
    tools_called: list[str] = Field(
        description="Every tool the orchestrator called for this message, in order. The specialists are the ask_* tools."
    )
    requirements: dict[str, str] | None = Field(
        description="What the orchestrator saved about the farmer's need: field_id, crop, problem, timeframe."
    )
    findings: dict[str, Finding] = Field(
        description="What each specialist reported for this message, by name: weather, farm_context, knowledge."
    )
