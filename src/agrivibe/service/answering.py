from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agrivibe.agents.fieldoracle import CitedAnswer
from agrivibe.core.tracing import run_config
from agrivibe.models.ask import AnswerSentence, AskResponse
from agrivibe.models.knowledge import Source

NOT_FOUND = "The indexed documents do not cover this question."


def current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drops the earlier turns of a thread, keeping the latest question onwards.

    Args:
        messages: The thread's full message history after a run.

    Returns:
        The messages from the last user message to the end.
    """
    starts = [i for i, message in enumerate(messages) if isinstance(message, HumanMessage)]
    return messages[starts[-1]:] if starts else messages


def retrieved_chunks(messages: list[BaseMessage]) -> dict[str, Document]:
    """Collects every chunk the search tool returned during one agent run.

    Args:
        messages: The agent's message history for the run.

    Returns:
        The retrieved chunks, keyed by record id (`<doc_id>-<n>`). Metadata
        chunk_id is only the chunk's position and repeats across documents.
    """
    chunks = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_knowledge":
            for doc in message.artifact or []:
                chunks[doc.id] = doc
    return chunks


def search_queries(messages: list[BaseMessage]) -> list[str]:
    """Lists the queries the agent chose to search for.

    Args:
        messages: The agent's message history for the run.

    Returns:
        The search queries, in the order the agent ran them.
    """
    return [
        call["args"]["query"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call["name"] == "search_knowledge"
    ]


def to_source(doc: Document) -> Source:
    """Converts a retrieved chunk to the `Source` shape the API returns."""
    meta = doc.metadata
    return Source(
        id=doc.id or "",
        source=meta["source"],
        page=meta["page"],
        category=meta["category"],
        text=doc.page_content,
        rerank_score=meta.get("rerank_score"),
    )


class AnswerService:
    """Answers questions with the RAG agent."""

    def __init__(self, rag_agent, tracer: BaseCallbackHandler | None = None):
        """Initializes the service.

        Args:
            rag_agent: The agent built by `build_fieldoracle_agent`.
            tracer: The Langfuse handler from `create_tracer`, or None to run without tracing.
        """
        self.rag_agent = rag_agent
        self.tracer = tracer

    def ask(self, question: str, thread_id: str | None = None) -> AskResponse:
        """Asks the agent a question and builds a cited response.

        Args:
            question: The user's question.
            thread_id: The conversation to continue. A new one is started when empty.

        Returns:
            The answer as text with citations in brackets, the same sentences as a
            list, and the retrieved passages the answer cites.
        """
        thread_id = thread_id or str(uuid4())
        result = self.rag_agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            run_config("ask", thread_id, self.tracer),
        )
        answer: CitedAnswer = result["structured_response"]
        # The checkpointer returns the whole thread. Sources and searches come from this turn only.
        messages = current_turn(result["messages"])

        found = answer.found and bool(answer.sentences)
        sentences = answer.sentences if found else []
        cited = {chunk_id for s in sentences for chunk_id in s.citations}

        return AskResponse(
            question=question,
            thread_id=thread_id,
            found=found,
            answer=" ".join(f"{s.sentence} [{', '.join(s.citations)}]" for s in sentences) if found else NOT_FOUND,
            sentences=[AnswerSentence(sentence=s.sentence, citations=s.citations) for s in sentences],
            sources=[to_source(doc) for chunk_id, doc in retrieved_chunks(messages).items() if chunk_id in cited],
            searches=search_queries(messages),
        )
