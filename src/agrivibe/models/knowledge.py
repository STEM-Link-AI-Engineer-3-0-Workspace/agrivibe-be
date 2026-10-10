"""Bodies for POST /documents and POST /search."""

from pydantic import BaseModel, Field

from agrivibe.db.vectorstore import Category


class IndexResponse(BaseModel):
    """The result of indexing one PDF."""

    doc_id: str
    filename: str
    category: Category
    pages: int
    chunks: int


class SearchRequest(BaseModel):
    """A search, optionally limited to one category and optionally reranked."""

    query: str = Field(min_length=1)
    category: Category | None = None
    k: int = Field(default=4, ge=1, le=20)
    rerank: bool = Field(
        default=True, description="Re-score the vector hits with the cross-encoder. False keeps vector order."
    )


class Source(BaseModel):
    """A retrieved chunk and where it came from."""

    id: str
    source: str
    page: int
    category: Category
    text: str
    rerank_score: float | None = Field(
        default=None, description="The cross-encoder's score for this chunk. None when reranking is off."
    )


class SearchHit(Source):
    """A search result: a chunk plus its similarity score and, if reranked, its rerank score."""

    score: float = Field(description="The vector similarity score from the first stage.")


class SearchResponse(BaseModel):
    """The hits of a search, best match first."""

    hits: list[SearchHit]
