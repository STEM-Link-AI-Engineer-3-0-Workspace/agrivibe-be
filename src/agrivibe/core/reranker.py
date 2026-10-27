import logging
import time

from langchain_core.documents import Document
from langchain_pinecone import PineconeRerank

from agrivibe.config import Settings

logger = logging.getLogger(__name__)


class Reranker:
    """A cross-encoder that scores how well each chunk answers a query.

    An embedding search compares two vectors made separately. A cross-encoder
    reads the query and the chunk together, so it judges relevance better, but
    it runs once per chunk. That is why it only re-scores a short list.
    """

    kind = "pinecone"

    def __init__(self, settings: Settings):
        """Connects to Pinecone's hosted rerank model.

        Args:
            settings: Application settings with the Pinecone key, the model
                name and the number of candidates to re-score.
        """
        self.candidates = settings.rerank_candidates
        # Hosted by Pinecone, so it reuses the Pinecone key and downloads nothing.
        self.model = PineconeRerank(model=settings.rerank_model, pinecone_api_key=settings.pinecone_api_key)

    def score(self, query: str, docs: list[Document]) -> list[float]:
        """Scores each chunk against the query.

        Args:
            query: The search query.
            docs: The chunks to score.

        Returns:
            One score per chunk, in the same order as `docs`. Higher is more
            relevant. Empty if the reranker failed.
        """
        # The record id goes along so the results map back to our chunks, never metadata chunk_id.
        passages = [{"id": doc.id, "text": doc.page_content} for doc in docs]
        results = self.model.rerank(passages, query, top_n=len(passages))
        scores = {result["index"]: result["score"] for result in results}
        return [scores[i] for i in range(len(docs))] if results else []


def rerank(
    reranker: Reranker | None, query: str, hits: list[tuple[Document, float]], k: int
) -> list[tuple[Document, float]]:
    """Keeps the k hits the cross-encoder scores highest.

    Args:
        reranker: The cross-encoder. If None, the hits keep their vector order.
        query: The search query.
        hits: (document, similarity score) pairs from the vector search.
        k: The number of hits to keep.

    Returns:
        Up to k hits, best rerank score first. Each document's metadata gets a
        `rerank_score`; the similarity score is kept alongside it.
    """
    if reranker is None or not hits:
        return hits[:k]

    start = time.perf_counter()
    scores = reranker.score(query, [doc for doc, _score in hits])
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info("Reranked %d chunks with %s in %.0f ms", len(hits), reranker.kind, elapsed_ms)

    if not scores:
        logger.warning("Rerank failed, keeping the vector search order")
        return hits[:k]

    reranked = []
    for (doc, similarity), rerank_score in zip(hits, scores):
        # Copy the metadata so the score does not leak into the store's own copy of the chunk.
        metadata = {**doc.metadata, "rerank_score": rerank_score}
        doc = Document(id=doc.id, page_content=doc.page_content, metadata=metadata)
        reranked.append((doc, similarity))
    reranked.sort(key=lambda hit: hit[0].metadata["rerank_score"], reverse=True)
    return reranked[:k]
