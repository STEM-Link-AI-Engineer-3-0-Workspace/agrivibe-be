from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agrivibe.agents.fieldoracle import build_fieldoracle_agent
from agrivibe.dependencies import get_answer_service
from agrivibe.main import app
from agrivibe.service.answering import NOT_FOUND, AnswerService
from tests.conftest import upload
from tests.fakes import FakeReranker, ScriptedChatModel


def search_call(query, category=None):
    return AIMessage(content="", tool_calls=[{"name": "search_knowledge", "args": {"query": query, "category": category}, "id": query}])


def answer_call(found, sentences):
    return AIMessage(content="", tool_calls=[{"name": "CitedAnswer", "args": {"found": found, "sentences": sentences}, "id": "final"}])


def use_agent(vector_store, replies, k=10, reranker=None):
    """Run the real agent graph, with a scripted model in place of OpenAI."""
    agent = build_fieldoracle_agent(
        ScriptedChatModel(replies=replies), vector_store, k=k, checkpointer=InMemorySaver(), reranker=reranker
    )
    service = AnswerService(agent)
    app.dependency_overrides[get_answer_service] = lambda: service
    return agent


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


def test_follow_up_continues_the_thread(client, vector_store):
    upload(client, "fertiliser_subsidy_circular_2026.pdf", "policy")
    first, second = list(vector_store.store)[:2]
    agent = use_agent(vector_store, [
        search_call("subsidy claim deadline", "policy"),
        answer_call(True, [{"sentence": "Claims close on 15 November 2026.", "citations": [first]}]),
        search_call("where to apply for subsidy", "policy"),
        answer_call(True, [{"sentence": "Apply at the Agrarian Service Centre.", "citations": [second]}]),
    ])

    opening = client.post("/ask", json={"question": "When do subsidy claims close?"}).json()
    follow_up = client.post("/ask", json={"question": "Where do I apply?", "thread_id": opening["thread_id"]}).json()

    assert follow_up["thread_id"] == opening["thread_id"]
    assert follow_up["searches"] == ["where to apply for subsidy"]
    assert {source["id"] for source in follow_up["sources"]} == {second}
    history = agent.get_state({"configurable": {"thread_id": opening["thread_id"]}}).values["messages"]
    assert [m.content for m in history if isinstance(m, HumanMessage)] == ["When do subsidy claims close?", "Where do I apply?"]


def test_new_question_starts_a_new_thread(client, vector_store):
    use_agent(vector_store, [
        search_call("tea pruning cycle"), answer_call(False, []),
        search_call("tea pruning cycle"), answer_call(False, []),
    ])

    one = client.post("/ask", json={"question": "How often should I prune tea?"}).json()
    two = client.post("/ask", json={"question": "How often should I prune tea?"}).json()

    assert one["thread_id"] != two["thread_id"]


def test_reranked_citations_point_to_retrieved_record_ids(client, vector_store):
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")
    upload(client, "fertiliser_subsidy_circular_2026.pdf", "policy")
    reranker = FakeReranker()
    # The same two-stage search the tool runs, to know which record ids it will return.
    candidates = [doc for doc, _score in vector_store.similarity_search_with_score("subsidy claim deadline", k=20)]
    scores = dict(zip([doc.id for doc in candidates], reranker.score("subsidy claim deadline", candidates)))
    top = sorted(scores, key=lambda record_id: scores[record_id], reverse=True)[:2]
    agent = use_agent(vector_store, [
        search_call("subsidy claim deadline"),
        answer_call(True, [{"sentence": "Claims close on 15 November 2026.", "citations": top}]),
    ], k=2, reranker=reranker)

    body = client.post("/ask", json={"question": "When do subsidy claims close?"}).json()

    tool_message = next(
        m for m in agent.get_state({"configurable": {"thread_id": body["thread_id"]}}).values["messages"]
        if m.type == "tool"
    )
    # The model reads passages headed by record ids, best rerank score first.
    assert tool_message.content.index(f"[{top[0]}]") < tool_message.content.index(f"[{top[1]}]")
    # Sources come from the tool's artifact, so they keep its reranked order and carry the rerank score.
    assert [source["id"] for source in body["sources"]] == top
    assert [source["rerank_score"] for source in body["sources"]] == [scores[i] for i in top]
