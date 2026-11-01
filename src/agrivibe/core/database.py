"""Postgres: where the orchestrator's checkpointer and store keep their data.

The checkpointer saves each thread's state (messages, requirements, findings). The store keeps
what outlives a thread: each farmer's saved field, their thread index and every turn's evidence.
Both use LangGraph's own tables, created by `setup()` on first start, so there is no schema here.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

CONNECT_TIMEOUT_SECONDS = 20  # a Neon database that scaled to zero takes a moment to wake


def where(url: str) -> str:
    """The host and database of a connection string, without the password, for error messages."""
    parts = urlsplit(url)
    return f"{parts.hostname}{parts.path}"


@asynccontextmanager
async def open_memory(url: str) -> AsyncIterator[tuple[AsyncPostgresSaver, AsyncPostgresStore]]:
    """Connects to Postgres and yields the checkpointer and the store, sharing one connection pool.

    Args:
        url: A Postgres connection string. With Neon, use the direct one (no -pooler in the host):
            the checkpointer uses prepared statements, which a transaction pooler can break.

    Yields:
        The checkpointer and the store, with their tables created.

    Raises:
        RuntimeError: If the database cannot be reached.
    """
    pool = AsyncConnectionPool(
        url,
        min_size=1,
        max_size=10,
        open=False,
        # The settings LangGraph's own from_conn_string uses.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        # Neon closes idle connections. Checking each one before use replaces a dead one quietly.
        check=AsyncConnectionPool.check_connection,
    )
    try:
        await pool.open(wait=True, timeout=CONNECT_TIMEOUT_SECONDS)
        checkpointer, store = AsyncPostgresSaver(pool), AsyncPostgresStore(pool)
        await checkpointer.setup()
        await store.setup()
    except Exception as error:
        await pool.close()
        raise RuntimeError(
            f"Could not connect to Postgres at {where(url)}. Check DATABASE_URL in .env: "
            "a free Neon project's direct connection string is enough."
        ) from error
    try:
        yield checkpointer, store
    finally:
        await pool.close()
