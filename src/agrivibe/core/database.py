"""Postgres: AgriVibe's own tables, and where the orchestrator's checkpointer and store keep theirs.

- Our tables (schema.sql): farmers, threads and turns, which ThreadService reads and writes with SQL.
- LangGraph's checkpointer: each thread's agent state (messages, requirements, findings), so a
  conversation can continue.
- LangGraph's store: what the orchestrator remembers about a farmer, such as their field.

All three share one connection pool. LangGraph creates its own tables with `setup()`.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

CONNECT_TIMEOUT_SECONDS = 20  # a Neon database that scaled to zero takes a moment to wake
SCHEMA = Path(__file__).with_name("schema.sql").read_text()


class Database(NamedTuple):
    """What open_database yields."""

    pool: AsyncConnectionPool
    checkpointer: AsyncPostgresSaver
    store: AsyncPostgresStore


def where(url: str) -> str:
    """The host and database of a connection string, without the password, for error messages."""
    parts = urlsplit(url)
    return f"{parts.hostname}{parts.path}"


@asynccontextmanager
async def open_database(url: str) -> AsyncIterator[Database]:
    """Connects to Postgres, creates any missing tables, and yields the pool, checkpointer and store.

    Args:
        url: A Postgres connection string. With Neon, use the direct one (no -pooler in the host):
            the checkpointer uses prepared statements, which a transaction pooler can break.

    Yields:
        The connection pool for our tables, and LangGraph's checkpointer and store on the same pool.

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
        async with pool.connection() as conn:
            # Several statements in one string can't be a prepared statement, so this one isn't.
            await conn.execute(SCHEMA, prepare=False)
    except Exception as error:
        await pool.close()
        raise RuntimeError(
            f"Could not connect to Postgres at {where(url)}. Check DATABASE_URL in .env: "
            "a free Neon project's direct connection string is enough."
        ) from error
    try:
        yield Database(pool, checkpointer, store)
    finally:
        await pool.close()
