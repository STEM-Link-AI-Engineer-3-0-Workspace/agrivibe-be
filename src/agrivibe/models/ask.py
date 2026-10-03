"""Bodies for POST /ask."""

from pydantic import BaseModel, Field

from agrivibe.models.knowledge import Source


class AskRequest(BaseModel):
    """A question for the RAG agent."""

    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = Field(
        default=None,
        max_length=200,
        description="The conversation to continue. Leave empty to start a new one; the response returns its id.",
    )


class AnswerSentence(BaseModel):
    """One sentence of the answer and the ids of the passages it cites."""

    sentence: str
    citations: list[str]


class AskResponse(BaseModel):
    """The agent's answer, the passages it cites and the searches it ran."""

    question: str
    thread_id: str = Field(description="The conversation this answer belongs to. Send it back to ask a follow-up.")
    found: bool
    answer: str = Field(description="The sentences joined, each followed by its citations: `Sentence [id-1, id-2]`.")
    sentences: list[AnswerSentence]
    sources: list[Source]
    searches: list[str] = Field(description="The search queries the agent ran, in order.")
