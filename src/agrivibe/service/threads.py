"""The farmer's conversations, in AgriVibe's own tables (core/schema.sql):

    farmers ──< threads ──< turns

A thread belongs to one farmer and a turn to one thread. Postgres keeps those links: a thread
can't name a farmer who doesn't exist, and deleting a thread deletes its turns (ON DELETE CASCADE).
Each turn keeps the evidence the farmer saw, in its findings column. The orchestrator clears its
findings on every new message, so this is the only record of past evidence.

The orchestrator's own state lives in LangGraph's tables, beside these: the checkpointer keeps each
thread's messages so a conversation can continue, and the store keeps what it remembers about the farmer.
"""

from __future__ import annotations  # list[...] below means the builtin, not ThreadService.list

from typing import Any
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from agrivibe.models.chat import Turn
from agrivibe.models.farmer import Farmer
from agrivibe.models.threads import Thread, ThreadSummary

TITLE_LENGTH = 80

# One farmer's thread as a list shows it. The turn count is counted, not stored.
SUMMARY = """
    SELECT t.id::text AS thread_id, t.title, t.created_at, t.updated_at, count(u.id)::int AS turns
    FROM threads t LEFT JOIN turns u ON u.thread_id = t.id
"""


class ThreadNotFound(LookupError):
    """The thread does not exist, or it is not this farmer's. The API answers 404 either way."""


def title_for(message: str) -> str:
    """The first message, cut to fit a list."""
    message = " ".join(message.split())
    return message if len(message) <= TITLE_LENGTH else message[: TITLE_LENGTH - 1].rstrip() + "…"


def thread_key(thread_id: str) -> UUID:
    """The thread id as the uuid the tables store. A string that isn't one names no thread."""
    try:
        return UUID(thread_id)
    except ValueError:
        raise ThreadNotFound(thread_id) from None


class ThreadService:
    """Lists, reads, records and deletes the farmer's conversations."""

    def __init__(self, pool: AsyncConnectionPool, checkpointer: BaseCheckpointSaver):
        """Initializes the service.

        Args:
            pool: Connections to Postgres, where our tables are.
            checkpointer: The orchestrator's checkpointer. Deleting a thread clears its state here too.
        """
        self.pool = pool
        self.checkpointer = checkpointer

    async def add_farmer(self, farmer: Farmer) -> None:
        """Creates the farmer, or updates their name.

        Startup calls this for the farmer in .env, the way an admin seeds an account. With real
        sign-up, registering would call it instead.
        """
        await self._run(
            "INSERT INTO farmers (id, name) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name",
            (farmer.id, farmer.name),
        )

    async def list(self, farmer_id: str) -> list[ThreadSummary]:
        """Returns the farmer's conversations, the most recently used first."""
        rows = await self._all(
            SUMMARY + "WHERE t.farmer_id = %s GROUP BY t.id ORDER BY t.updated_at DESC", (farmer_id,)
        )
        return [ThreadSummary(**row) for row in rows]

    async def summary(self, farmer_id: str, thread_id: str) -> ThreadSummary:
        """Returns one of the farmer's conversations without its history.

        Raises:
            ThreadNotFound: It does not exist or belongs to someone else.
        """
        row = await self._one(
            SUMMARY + "WHERE t.id = %s AND t.farmer_id = %s GROUP BY t.id", (thread_key(thread_id), farmer_id)
        )
        if row is None:
            raise ThreadNotFound(thread_id)
        return ThreadSummary(**row)

    async def get(self, farmer_id: str, thread_id: str) -> Thread:
        """Returns one of the farmer's conversations with every turn, oldest first.

        Raises:
            ThreadNotFound: It does not exist or belongs to someone else.
        """
        summary = await self.summary(farmer_id, thread_id)
        rows = await self._all(
            """SELECT message, answer, tools_called, requirements, findings, started_at, seconds
               FROM turns WHERE thread_id = %s ORDER BY position""",
            (thread_key(thread_id),),
        )
        return Thread(**summary.model_dump(), history=[Turn(**row) for row in rows])

    async def record_turn(self, farmer_id: str, thread_id: str, turn: Turn) -> None:
        """Saves one turn, creating its thread on the first one.

        Both inserts run in one transaction: the turn and its thread are saved together or not at
        all. The thread's row stays locked until the commit, so two messages to the same thread
        at once take the next positions one after the other instead of claiming the same one.

        Raises:
            ThreadNotFound: The thread exists but belongs to someone else.
        """
        key, saved = thread_key(thread_id), turn.model_dump(mode="json")
        async with self.pool.connection() as conn, conn.transaction():
            cursor = await conn.execute(
                """INSERT INTO threads (id, farmer_id, title, created_at) VALUES (%s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET updated_at = now() WHERE threads.farmer_id = EXCLUDED.farmer_id
                   RETURNING id""",
                (key, farmer_id, title_for(turn.message), turn.started_at),
            )
            if await cursor.fetchone() is None:
                raise ThreadNotFound(thread_id)
            await conn.execute(
                """INSERT INTO turns (thread_id, position, message, answer, tools_called, requirements, findings, started_at, seconds)
                   SELECT %(thread)s, coalesce(max(position) + 1, 0), %(message)s, %(answer)s, %(tools)s,
                          %(requirements)s, %(findings)s, %(started_at)s, %(seconds)s
                   FROM turns WHERE thread_id = %(thread)s""",
                {
                    "thread": key,
                    "message": turn.message,
                    "answer": turn.answer,
                    "tools": turn.tools_called,
                    "requirements": Jsonb(saved["requirements"]) if turn.requirements is not None else None,
                    "findings": Jsonb(saved["findings"]),
                    "started_at": turn.started_at,
                    "seconds": turn.seconds,
                },
            )

    async def delete(self, farmer_id: str, thread_id: str) -> None:
        """Deletes one of the farmer's conversations: the thread, its turns (by cascade) and the orchestrator's state.

        Raises:
            ThreadNotFound: It does not exist or belongs to someone else.
        """
        row = await self._one(
            "DELETE FROM threads WHERE id = %s AND farmer_id = %s RETURNING id", (thread_key(thread_id), farmer_id)
        )
        if row is None:
            raise ThreadNotFound(thread_id)
        await self.checkpointer.adelete_thread(thread_id)

    async def _run(self, query: str, params: tuple) -> None:
        async with self.pool.connection() as conn:
            await conn.execute(query, params)

    async def _one(self, query: str, params: tuple) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            return await (await conn.execute(query, params)).fetchone()

    async def _all(self, query: str, params: tuple) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            return await (await conn.execute(query, params)).fetchall()
