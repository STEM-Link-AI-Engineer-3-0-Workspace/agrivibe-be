from typing import Literal

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from agrivibe.config import Settings
from agrivibe.core.reranker import Reranker, rerank

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
    reranker: Reranker | None = None,
) -> list[tuple[Document, float]]:
    """Returns the chunks closest to a query.

    Without a reranker this is one vector search for k chunks. With one, it is
    two stages: the vector search fetches a longer shortlist, then the
    cross-encoder re-scores it and the best k are kept.

    Args:
        vector_store: The store to search.
        query: The text to search for.
        k: The number of chunks to return.
        category: If set, only chunks tagged with this category are searched.
        reranker: If set, re-scores the shortlist. Each returned chunk then
            carries a `rerank_score` in its metadata.

    Returns:
        Up to k (document, similarity score) pairs, best match first.
    """
    search_filter = {"category": category} if category else None
    fetch_k = max(reranker.candidates, k) if reranker else k
    hits = vector_store.similarity_search_with_score(query, k=fetch_k, filter=search_filter)
    return rerank(reranker, query, hits, k)
