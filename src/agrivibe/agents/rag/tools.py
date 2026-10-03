from langchain.tools import BaseTool, tool
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from agrivibe.core.vectorstore import Category, search_chunks


def format_passages(docs: list[Document]) -> str:
    """Formats retrieved chunks as text for the model to read.

    Args:
        docs: The retrieved chunks.

    Returns:
        One block per chunk, headed by `[chunk_id] source, page N` so the model
        has an exact id to cite.
    """
    if not docs:
        return "No passages found."
    blocks = []
    for doc in docs:
        meta = doc.metadata
        header = f"[{meta['chunk_id']}] {meta['source']}, page {meta['page']}"
        blocks.append(header + "\n" + doc.page_content)
    return "\n\n".join(blocks)


def make_search_tool(vector_store: VectorStore, k: int) -> BaseTool:
    """Wraps the vector store search as a tool the agent can call.

    Args:
        vector_store: The store to search.
        k: The number of passages each search returns.

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
        docs = [doc for doc, _score in search_chunks(vector_store, query, k, category)]
        # The model reads the text. The documents ride along as the artifact,
        # so the service can check every citation against what was retrieved.
        return format_passages(docs), docs

    return search_knowledge
