from typing import Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from agrivibe.config import Settings

# The three document classes FieldOracle serves. Every chunk stores one of these
# in its metadata, so a search can be limited to a single class.
Category = Literal["cultivation_guide", "policy", "farm_record"]


def create_vector_store(settings: Settings) -> PineconeVectorStore:
    """Connects to the Pinecone index, creating it on first run.

    The index dimension must match the embedding model, so changing
    `EMBEDDING_MODEL` to one with a different size needs a new index.

    Args:
        settings: Application settings with the API keys and index details.

    Returns:
        A vector store that embeds text with OpenAI and stores it in Pinecone.
    """
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )

    pc = Pinecone(api_key=settings.pinecone_api_key.get_secret_value())
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )

    return PineconeVectorStore(index=pc.Index(settings.pinecone_index), embedding=embeddings)


def search_chunks(
    vector_store: VectorStore,
    query: str,
    k: int,
    category: Category | None = None,
) -> list[tuple[Document, float]]:
    """Returns the chunks closest to a query.

    Args:
        vector_store: The store to search.
        query: The text to search for.
        k: The number of chunks to return.
        category: If set, only chunks tagged with this category are searched.

    Returns:
        Up to k (document, score) pairs, best match first.
    """
    search_filter = {"category": category} if category else None
    return vector_store.similarity_search_with_score(query, k=k, filter=search_filter)
