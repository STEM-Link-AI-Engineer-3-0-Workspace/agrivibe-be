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
    """Answers questions with the RAG agent and checks its citations."""

    def __init__(self, rag_agent):
        """Initializes the service.

        Args:
            rag_agent: The agent built by `build_rag_agent`.
        """
        self.rag_agent = rag_agent

    def ask(self, question: str) -> AskResponse:
        """Asks the agent a question and builds a checked, cited response.

        Args:
            question: The user's question.

        Returns:
            The answer as sentence and citation pairs. A citation to a passage
            that was not retrieved in this run is kept but marked
            `verified: false`, and only verified citations appear in `sources`.
        """
        result = self.rag_agent.invoke({"messages": [{"role": "user", "content": question}]})
        answer: CitedAnswer = result["structured_response"]
        retrieved = retrieved_chunks(result["messages"])

        # Grounding check: a citation only counts if that passage was retrieved in this run.
        sentences = [
            AnswerSentence(sentence=s.sentence, citation=s.citation, verified=s.citation in retrieved)
            for s in answer.sentences
        ]

        cited_ids = dict.fromkeys(s.citation for s in sentences if s.verified)
        sources = [to_source(retrieved[chunk_id]) for chunk_id in cited_ids]

        found = answer.found and bool(sentences)
        return AskResponse(
            question=question,
            found=found,
            answer=" ".join(s.sentence for s in sentences) if found else NOT_FOUND,
            sentences=sentences if found else [],
            sources=sources if found else [],
            searches=search_queries(result["messages"]),
        )
