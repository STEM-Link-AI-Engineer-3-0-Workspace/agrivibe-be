from langchain_core.messages import AIMessage

from agrivibe.agents.rag import build_rag_agent
from agrivibe.dependencies import get_answer_service
from agrivibe.main import app
from agrivibe.services.answering import NOT_FOUND, AnswerService
from tests.conftest import upload
from tests.fakes import ScriptedChatModel


def search_call(query, category=None):
    return AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": query, "category": category}, "id": query}])


def answer_call(found, sentences):
    return AIMessage(content="", tool_calls=[{"name": "CitedAnswer", "args": {"found": found, "sentences": sentences}, "id": "final"}])


def use_agent(vector_store, replies):
    """Run the real agent graph, with a scripted model in place of OpenAI."""
    agent = build_rag_agent(ScriptedChatModel(replies=replies), vector_store, k=10)
    app.dependency_overrides[get_answer_service] = lambda: AnswerService(agent)


def test_ask_returns_cited_sentences(client, vector_store):
    upload(client, "fertiliser_subsidy_circular_2026.pdf", "policy")
    first, second = list(vector_store.store)[:2]
    use_agent(vector_store, [
        search_call("subsidy claim deadline", "policy"),
        answer_call(True, [
            {"sentence": "Claims close on 15 November 2026.", "citations": [first, second]},
            {"sentence": "Apply at the Agrarian Service Centre.", "citations": [second]},
        ]),
    ])

    body = client.post("/ask", json={"question": "When do subsidy claims close?"}).json()

    assert body["found"] is True
    assert body["answer"] == (
        f"Claims close on 15 November 2026. [{first}, {second}] Apply at the Agrarian Service Centre. [{second}]"
    )
    assert body["searches"] == ["subsidy claim deadline"]
    assert {source["id"] for source in body["sources"]} == {first, second}


def test_ask_not_found(client, vector_store):
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")
    use_agent(vector_store, [search_call("tea pruning cycle"), answer_call(False, [])])

    body = client.post("/ask", json={"question": "How often should I prune tea?"}).json()

    assert body["found"] is False
    assert body["answer"] == NOT_FOUND
    assert body["sources"] == []
