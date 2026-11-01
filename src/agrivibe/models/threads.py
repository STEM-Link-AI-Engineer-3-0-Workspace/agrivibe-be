"""Bodies for GET /threads and GET /threads/{thread_id}."""

from datetime import datetime

from pydantic import BaseModel, Field

from agrivibe.models.chat import Turn


class ThreadSummary(BaseModel):
    """One of the farmer's conversations, as a list shows it."""

    thread_id: str
    title: str = Field(description="The conversation's first message, shortened.")
    created_at: datetime
    updated_at: datetime
    turns: int = Field(description="How many messages the farmer has sent in it.")


class Thread(ThreadSummary):
    """A whole conversation: every message, its answer and the evidence behind it, oldest first."""

    history: list[Turn]
