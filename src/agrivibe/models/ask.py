"""Bodies for POST /ask."""

from pydantic import BaseModel, Field

from agrivibe.models.knowledge import Source


class AskRequest(BaseModel):
    """A question for the RAG agent."""

    question: str = Field(min_length=1, max_length=2000)


class AnswerSentence(BaseModel):
    """One sentence of the answer, its citation and whether the citation checked out."""

    sentence: str
    citation: str
    verified: bool = Field(description="False if the citation is not a passage the agent actually retrieved.")


class AskResponse(BaseModel):
    """The agent's answer, the passages it cites and the searches it ran."""

    question: str
    found: bool
    answer: str
    sentences: list[AnswerSentence]
    sources: list[Source]
    searches: list[str] = Field(description="The search queries the agent ran, in order.")
