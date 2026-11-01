"""Bodies for POST /chat. A Turn is also what GET /threads/{thread_id} returns for each past message."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agrivibe.models.knowledge import Source


class ChatRequest(BaseModel):
    """A message from the farmer to AgriVibe.

    There is no user_id: the server decides who is asking (see current_farmer in dependencies.py).
    """

    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(
        default=None,
        max_length=200,
        description="One of your conversations to continue. Leave empty to start a new one; the response returns its id.",
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
    sources: list[Source] = Field(
        default_factory=list,
        description="knowledge only: every passage the answer cites, with its file, page, text and rerank score.",
    )


class Turn(BaseModel):
    """One message from the farmer and everything AgriVibe did to answer it."""

    message: str
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
    started_at: datetime
    seconds: float = Field(description="How long the turn took, end to end.")


class ChatResponse(Turn):
    """AgriVibe's reply to one message, with the evidence behind it."""

    thread_id: str = Field(description="The conversation this reply belongs to. Send it back to continue.")
