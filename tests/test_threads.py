import psycopg
from langchain_core.messages import AIMessage

from tests.conftest import TEST_DATABASE_URL
from tests.test_chat import FIELD_STATE, REQUIREMENTS, as_farmer, calls, chat, use_orchestrator

FOLLOW_UP = {
    **FIELD_STATE,
    "orchestrator": FIELD_STATE["orchestrator"] + [AIMessage(content="It rained 112 mm this week.")],
}


def test_a_new_farmer_has_no_conversations(client, vector_store):
    use_orchestrator(vector_store)

    assert client.get("/threads").json() == []


def test_a_conversation_keeps_every_turn_with_its_evidence(client, vector_store):
    use_orchestrator(vector_store, **FOLLOW_UP)

    first = chat(client, "Is my Minneriya field stressed?")
    chat(client, "How much rain was that?", thread_id=first["thread_id"])

    [summary] = client.get("/threads").json()
    assert summary["thread_id"] == first["thread_id"]
    assert summary["title"] == "Is my Minneriya field stressed?"
    assert summary["turns"] == 2

    thread = client.get(f"/threads/{first['thread_id']}").json()
    opening, follow_up = thread["history"]
    assert [opening["message"], follow_up["message"]] == ["Is my Minneriya field stressed?", "How much rain was that?"]
    assert follow_up["answer"] == "It rained 112 mm this week."
    # Findings are cleared on every message, so the first turn's evidence survives only here.
    assert opening["tools_called"] == ["save_requirements", "ask_farm_context_agent", "ask_weather_agent"]
    assert opening["findings"]["weather"]["data"]["rainfall_last_7_days_mm"] == 112.0
    assert opening["requirements"] == REQUIREMENTS
    assert follow_up["findings"] == {}
    assert opening["seconds"] >= 0 and opening["started_at"]


def test_conversations_are_listed_most_recent_first(client, vector_store):
    use_orchestrator(vector_store, orchestrator=[
        AIMessage(content="Hello."), AIMessage(content="Hello again."), AIMessage(content="Still here."),
    ])

    older = chat(client, "Hi")
    newer = chat(client, "Hi again")
    chat(client, "Back to the first one", thread_id=older["thread_id"])

    assert [t["thread_id"] for t in client.get("/threads").json()] == [older["thread_id"], newer["thread_id"]]


def test_a_long_first_message_is_shortened_for_the_title(client, vector_store):
    use_orchestrator(vector_store, orchestrator=[AIMessage(content="Noted.")])

    chat(client, "Should I " + "really " * 30 + "spray?")

    [summary] = client.get("/threads").json()
    assert len(summary["title"]) <= 80 and summary["title"].endswith("…")


def test_another_farmers_conversation_is_not_found(client, vector_store):
    _, models = use_orchestrator(vector_store, orchestrator=[AIMessage(content="Hello.")])
    mine = chat(client, "Hi")["thread_id"]

    as_farmer("frm_someone_else")

    assert client.get("/threads").json() == []
    assert client.get(f"/threads/{mine}").status_code == 404
    assert client.delete(f"/threads/{mine}").status_code == 404
    # Continuing someone else's thread is refused before the orchestrator runs.
    response = client.post("/chat", json={"message": "Hi", "thread_id": mine})
    assert response.status_code == 404
    assert models["orchestrator"].calls == 1


def test_a_made_up_thread_id_is_not_found(client, vector_store):
    use_orchestrator(vector_store)

    assert client.post("/chat", json={"message": "Hi", "thread_id": "not-a-thread"}).status_code == 404
    assert client.get("/threads/not-a-thread").status_code == 404


def test_deleting_a_conversation_removes_its_history_and_state(client, vector_store):
    agent, _ = use_orchestrator(vector_store, orchestrator=[AIMessage(content="Hello.")])
    thread_id = chat(client, "Hi")["thread_id"]

    assert client.delete(f"/threads/{thread_id}").status_code == 204

    assert client.get("/threads").json() == []
    assert client.get(f"/threads/{thread_id}").status_code == 404
    assert agent.get_state({"configurable": {"thread_id": thread_id}}).values == {}



def test_turns_are_numbered_in_order_and_deleted_with_their_thread(client, vector_store):
    use_orchestrator(vector_store, orchestrator=[AIMessage(content="Hello."), AIMessage(content="Still here.")])
    thread_id = chat(client, "Hi")["thread_id"]
    chat(client, "Are you there?", thread_id=thread_id)

    def positions():
        with psycopg.connect(TEST_DATABASE_URL) as conn:
            rows = conn.execute("SELECT position FROM turns WHERE thread_id = %s ORDER BY position", (thread_id,))
            return [position for (position,) in rows]

    assert positions() == [0, 1]
    client.delete(f"/threads/{thread_id}")
    assert positions() == []  # ON DELETE CASCADE: the turns went with their thread

def test_the_server_decides_who_is_asking(client, vector_store):
    use_orchestrator(vector_store, orchestrator=[AIMessage(content="Hello.")])
    as_farmer("frm_server_side")

    # A client that still sends user_id is ignored: identity is never taken from the body.
    body = client.post("/chat", json={"message": "Hi", "user_id": "frm_spoofed"}).json()

    assert "user_id" not in body
    assert client.get("/me").json() == {"id": "frm_server_side", "name": "Test Farmer"}
    assert [t["thread_id"] for t in client.get("/threads").json()] == [body["thread_id"]]
    as_farmer("frm_spoofed")
    assert client.get("/threads").json() == []


def test_a_specialist_task_still_names_the_saved_field(client, vector_store):
    _, models = use_orchestrator(vector_store, orchestrator=[
        calls(("save_requirements", REQUIREMENTS)),
        calls(("ask_weather_agent", {"task": "Rain last week"})),
        AIMessage(content="112 mm."),
    ], weather=FIELD_STATE["weather"])

    thread_id = chat(client, "Has it rained on my Minneriya field?")["thread_id"]

    assert client.get(f"/threads/{thread_id}").json()["history"][0]["requirements"]["field_id"] == REQUIREMENTS["field_id"]
    assert models["weather"].received[0][-1].text.startswith("Field: SL-PADDY-MINNERIYA")
