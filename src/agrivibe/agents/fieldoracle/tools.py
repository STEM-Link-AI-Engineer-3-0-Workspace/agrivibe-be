from typing import Any

from langchain.tools import BaseTool, tool
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.vectorstores import VectorStore

from agrivibe.core.reranker import Reranker
from agrivibe.db.vectorstore import Category, search_chunks


def format_passages(docs: list[Document]) -> str:
    """Formats retrieved chunks as text for the model to read.

    Args:
        docs: The retrieved chunks.

    Returns:
        One block per chunk, headed by `[id] source, page N` so the model
        has an exact id to cite.
    """
    if not docs:
        return "No passages found."
    blocks = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{doc.id}] {meta['source']}, page {meta['page']}"
        blocks.append(header + "\n" + doc.page_content)
    return "\n\n".join(blocks)


def cited_sources(messages: list[BaseMessage], citations: list[str]) -> list[dict[str, Any]]:
    """Returns the passages an answer cites, as the search tool retrieved them.

    The search tool returns the documents as its artifact, so the cited passages are read
    from the agent's messages rather than searched again.

    Args:
        messages: The agent's message history for one run.
        citations: The record ids (`<doc_id>-<n>`) the answer cites, in order.

    Returns:
        One dict per cited passage: id, source, page, category, text and rerank_score.
    """
    retrieved = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_knowledge":
            for doc in message.artifact or []:
                retrieved[doc.id] = doc
    return [
        {
            "id": doc.id,
            "source": doc.metadata["source"],
            "page": doc.metadata["page"],
            "category": doc.metadata["category"],
            "text": doc.page_content,
            "rerank_score": doc.metadata.get("rerank_score"),
        }
        for citation in citations
        if (doc := retrieved.get(citation)) is not None
    ]


def make_search_tool(vector_store: VectorStore, k: int, reranker: Reranker | None = None) -> BaseTool:
    """Wraps the vector store search as a tool the agent can call.

    Args:
        vector_store: The store to search.
        k: The number of passages each search returns.
        reranker: Re-scores the vector hits before the best k are kept. None turns reranking off.

    Returns:
        The `search_knowledge` tool.
    """
    @tool(response_format="content_and_artifact", parse_docstring=True)
    def search_knowledge(query: str, category: Category | None = None) -> tuple[str, list[Document]]:
        """Search the FieldOracle knowledge base for passages about a question.

        Args:
            query: What to search for, written as a short search query, not the farmer's full message.
            category: Limit the search to one document class. cultivation_guide covers how to grow a
                crop. policy covers government circulars, subsidies and mandates. farm_record covers
                what happened on a specific registered field. Leave empty to search everything.

        Returns:
            The passages as text for the model, and the documents themselves as the artifact.
        """
        docs = [doc for doc, _score in search_chunks(vector_store, query, k, category, reranker)]
        # The model reads the text. The documents ride along as the artifact, in reranked
        # order, so the service can return the cited passages without searching again.
        return format_passages(docs), docs

    return search_knowledge
