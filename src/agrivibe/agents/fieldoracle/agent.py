from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langgraph.types import Checkpointer

from agrivibe.agents.fieldoracle.prompts import SYSTEM_PROMPT
from agrivibe.agents.fieldoracle.schemas import CitedAnswer
from agrivibe.agents.fieldoracle.tools import make_search_tool
from agrivibe.core.reranker import Reranker


def build_fieldoracle_agent(
    model: BaseChatModel,
    vector_store: VectorStore,
    k: int = 4,
    checkpointer: Checkpointer = None,
    reranker: Reranker | None = None,
):
    """Builds the RAG agent: a search tool plus a cited, structured answer.

    Args:
        model: The chat model that decides what to search and writes the answer.
        vector_store: The store the search tool reads from.
        k: The number of passages each search returns.
        checkpointer: Saves each thread's messages between runs, so a follow-up
            question sees the earlier turns. Runs must then pass a `thread_id`.
        reranker: Re-scores each search's hits with a cross-encoder. None turns reranking off.

    Returns:
        A compiled agent graph. Its result holds the full message history under
        `messages` and a `CitedAnswer` under `structured_response`.
    """
    return create_agent(
        model,
        tools=[make_search_tool(vector_store, k, reranker)],
        system_prompt=SYSTEM_PROMPT,
        response_format=CitedAnswer,
        checkpointer=checkpointer,
        name="FieldOracle",
    )
