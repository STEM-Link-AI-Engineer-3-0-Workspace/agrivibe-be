import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Settings need both keys and a database URL to exist. The fakes below mean none is ever used,
# and the app's real lifespan (FieldSat, DATABASE_URL) is swapped for test_lifespan below.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")

# Threads live in our own tables, so the chat and thread tests need a real Postgres. Point this at
# an empty database you don't mind wiping, e.g. postgresql://localhost/agrivibe_test. Without it,
# those tests are skipped; the others need no database.
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

from agrivibe.core.database import open_database
from agrivibe.dependencies import get_knowledge_service
from agrivibe.main import app
from agrivibe.service.knowledge import KnowledgeService
from tests.fakes import FakeVectorStore

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.fixture
def vector_store():
    return FakeVectorStore()


@asynccontextmanager
async def test_lifespan(app):
    """The app's lifespan in tests: no FieldSat, and an emptied test database when there is one.

    It runs inside the TestClient's event loop, which every request then shares. The pool has to
    open in that loop: an async pool opened in another one hangs on its first query.
    """
    if not TEST_DATABASE_URL:
        app.state.pool = None
        yield
        return
    async with open_database(TEST_DATABASE_URL) as db:
        async with db.pool.connection() as conn:
            await conn.execute("TRUNCATE farmers CASCADE")  # threads and turns go with them
        app.state.pool = db.pool
        yield


@pytest.fixture
def client(vector_store, monkeypatch):
    app.dependency_overrides[get_knowledge_service] = lambda: KnowledgeService(vector_store, 1000, 150)
    monkeypatch.setattr(app.router, "lifespan_context", test_lifespan)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def upload(client, filename, category):
    with open(SAMPLES / filename, "rb") as f:
        return client.post("/documents", files={"file": (filename, f, "application/pdf")}, data={"category": category})
