import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Settings need both keys to exist. The fakes below mean neither is ever used.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")

from agrivibe.dependencies import get_knowledge_service
from agrivibe.main import app
from agrivibe.services.knowledge import KnowledgeService
from tests.fakes import FakeVectorStore

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture
def vector_store():
    return FakeVectorStore()


@pytest.fixture
def client(vector_store):
    app.dependency_overrides[get_knowledge_service] = lambda: KnowledgeService(vector_store, 1000, 150)
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, filename, category):
    with open(SAMPLES / filename, "rb") as f:
        return client.post("/documents", files={"file": (filename, f, "application/pdf")}, data={"category": category})
