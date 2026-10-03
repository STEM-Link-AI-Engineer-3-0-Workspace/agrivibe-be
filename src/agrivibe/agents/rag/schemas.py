from pydantic import BaseModel, Field


class CitedSentence(BaseModel):
    """One sentence of the answer and the passage that supports it."""

    sentence: str = Field(description="One sentence of the answer, written from a retrieved passage.")
    citation: str = Field(
        description="The id of the passage that supports this sentence, copied exactly from the search results."
    )


class CitedAnswer(BaseModel):
    """The final answer, written only from retrieved passages."""

    found: bool = Field(
        description="True only if the retrieved passages answer the question. False if the knowledge base does not cover it."
    )
    sentences: list[CitedSentence] = Field(
        description="The answer, one sentence per item. Empty when found is false."
    )
