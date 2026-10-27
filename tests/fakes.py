"""Stand-ins for OpenAI and Pinecone, so the tests run offline and for free."""

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores import InMemoryVectorStore


class FakeVectorStore(InMemoryVectorStore):
    """In-memory store that accepts Pinecone-style {"key": value} filters."""

    def __init__(self):
        super().__init__(DeterministicFakeEmbedding(size=64))

    def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        if isinstance(filter, dict):
            wanted = filter

            def matches(doc):
                return all(doc.metadata.get(key) == value for key, value in wanted.items())

            filter = matches

        return super().similarity_search_with_score(query, k=k, filter=filter, **kwargs)


class ScriptedChatModel(BaseChatModel):
    """Replies with a fixed list of messages, one per call. Tool calls included."""

    replies: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        reply = self.replies[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


class FakeReranker:
    """Stands in for the cross-encoder. A chunk scores one point per query word it contains."""

    kind = "fake"

    def __init__(self, candidates=20):
        self.candidates = candidates
        self.calls = 0

    def score(self, query, docs):
        self.calls += 1
        words = set(query.lower().split())
        return [float(sum(word in doc.page_content.lower() for word in words)) for doc in docs]
