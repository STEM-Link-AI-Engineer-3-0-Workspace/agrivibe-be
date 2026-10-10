from agrivibe.dependencies import get_knowledge_service
from agrivibe.main import app
from agrivibe.service.knowledge import KnowledgeService
from tests.conftest import upload
from tests.fakes import FakeReranker

QUERY = "fertiliser subsidy claim deadline"


def use_reranker(vector_store, reranker):
    """Serve /search from a KnowledgeService that reranks with `reranker` (None turns it off)."""
    app.dependency_overrides[get_knowledge_service] = lambda: KnowledgeService(vector_store, 1000, 150, reranker)


def upload_samples(client):
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")
    upload(client, "fertiliser_subsidy_circular_2026.pdf", "policy")
    upload(client, "minneriya_field_records_2026.pdf", "farm_record")


def vector_order(vector_store, k):
    return [doc.id for doc, _score in vector_store.similarity_search_with_score(QUERY, k=k)]


def test_search_returns_reranked_order_and_exactly_k_hits(client, vector_store):
    upload_samples(client)
    reranker = FakeReranker(candidates=20)
    use_reranker(vector_store, reranker)

    hits = client.post("/search", json={"query": QUERY, "k": 3}).json()["hits"]

    # Stage one fetched 20 candidates; stage two kept the 3 that score highest.
    candidates = [doc for doc, _score in vector_store.similarity_search_with_score(QUERY, k=20)]
    best = sorted(reranker.score(QUERY, candidates), reverse=True)[:3]
    assert len(hits) == 3
    assert [hit["rerank_score"] for hit in hits] == best
    assert all(hit["id"] in {doc.id for doc in candidates} for hit in hits)
    assert [hit["id"] for hit in hits] != vector_order(vector_store, 3)
    # The similarity score from stage one is still reported next to the rerank score.
    assert all(isinstance(hit["score"], float) for hit in hits)


def test_rerank_false_keeps_vector_order(client, vector_store):
    upload_samples(client)
    reranker = FakeReranker()
    use_reranker(vector_store, reranker)

    hits = client.post("/search", json={"query": QUERY, "k": 4, "rerank": False}).json()["hits"]

    assert [hit["id"] for hit in hits] == vector_order(vector_store, 4)
    assert all(hit["rerank_score"] is None for hit in hits)
    assert reranker.calls == 0


def test_reranker_none_behaves_as_before(client, vector_store):
    upload_samples(client)
    use_reranker(vector_store, None)

    hits = client.post("/search", json={"query": QUERY, "k": 4}).json()["hits"]

    assert [hit["id"] for hit in hits] == vector_order(vector_store, 4)
    assert all(hit["rerank_score"] is None for hit in hits)


def test_hits_are_identified_by_record_id_not_chunk_id(client, vector_store):
    upload_samples(client)
    use_reranker(vector_store, FakeReranker())

    hits = client.post("/search", json={"query": QUERY, "k": 20}).json()["hits"]

    # chunk_id restarts at 0 in every PDF, so only the record id is unique.
    chunk_ids = [vector_store.store[hit["id"]]["metadata"]["chunk_id"] for hit in hits]
    assert len(set(chunk_ids)) < len(hits)
    assert len({hit["id"] for hit in hits}) == len(hits)
