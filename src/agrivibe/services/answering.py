from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from agrivibe.agents.rag import CitedAnswer
from agrivibe.models.ask import AnswerSentence, AskResponse
from agrivibe.models.knowledge import Source

NOT_FOUND = "The indexed documents do not cover this question."


def retrieved_chunks(messages: list[BaseMessage]) -> dict[str, Document]:
    """Collects every chunk the search tool returned during one agent run.

    Args:
        messages: The agent's message history for the run.

    Returns:
        The retrieved chunks, keyed by chunk id.
    """
    chunks = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_knowledge":
            for doc in message.artifact or []:
                chunks[doc.metadata["chunk_id"]] = doc
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
    return Source(id=meta["chunk_id"], source=meta["source"], page=meta["page"], category=meta["category"], text=doc.page_content)


class AnswerService:
    """Answers questions with the RAG agent."""

    def __init__(self, rag_agent):
        """Initializes the service.

        Args:
            rag_agent: The agent built by `build_rag_agent`.
        """
        self.rag_agent = rag_agent

    def ask(self, question: str) -> AskResponse:
        """Asks the agent a question and builds a cited response.

        Args:
            question: The user's question.

        Returns:
            The answer as text with citations in brackets, the same sentences as a
            list, and the retrieved passages the answer cites.
        """
        result = self.rag_agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer: CitedAnswer = result["structured_response"]
        messages = result["messages"]

        found = answer.found and bool(answer.sentences)
        sentences = answer.sentences if found else []
        cited = {chunk_id for s in sentences for chunk_id in s.citations}

        return AskResponse(
            question=question,
            found=found,
            answer=" ".join(f"{s.sentence} [{', '.join(s.citations)}]" for s in sentences) if found else NOT_FOUND,
            sentences=[AnswerSentence(sentence=s.sentence, citations=s.citations) for s in sentences],
            sources=[to_source(doc) for chunk_id, doc in retrieved_chunks(messages).items() if chunk_id in cited],
            searches=search_queries(messages),
        )
