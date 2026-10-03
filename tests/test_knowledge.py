from tests.conftest import upload


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_index_pdf(client, vector_store):
    response = upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")

    assert response.status_code == 201
    body = response.json()
    assert body["pages"] == 2
    assert body["chunks"] == len(vector_store.store)


def test_reindexing_the_same_pdf_does_not_duplicate(client, vector_store):
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")
    first = len(vector_store.store)
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")

    assert len(vector_store.store) == first


def test_rejects_non_pdf(client):
    response = client.post(
        "/documents", files={"file": ("notes.txt", b"hello", "text/plain")}, data={"category": "policy"}
    )
    assert response.status_code == 415


def test_search_filters_by_category(client):
    upload(client, "paddy_cultivation_guide.pdf", "cultivation_guide")
    upload(client, "fertiliser_subsidy_circular_2026.pdf", "policy")

    response = client.post("/search", json={"query": "subsidy deadline", "category": "policy", "k": 10})

    hits = response.json()["hits"]
    assert hits
    assert {hit["category"] for hit in hits} == {"policy"}
