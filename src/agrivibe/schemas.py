"""Request and response bodies. These are the contract a frontend codes against."""

from pydantic import BaseModel, Field

from agrivibe.core.vectorstore import Category


class IndexResponse(BaseModel):
    """The result of indexing one PDF."""

    doc_id: str
    filename: str
    category: Category
    pages: int
    chunks: int


class SearchRequest(BaseModel):
    """A plain vector search, optionally limited to one category."""

    query: str = Field(min_length=1)
    category: Category | None = None
    k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    """A retrieved chunk and where it came from."""

    id: str
    source: str
    page: int
    category: Category
    text: str


class SearchHit(Source):
    """A search result: a chunk plus its similarity score."""

    score: float


class SearchResponse(BaseModel):
    """The hits of a search, best match first."""

    hits: list[SearchHit]


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
