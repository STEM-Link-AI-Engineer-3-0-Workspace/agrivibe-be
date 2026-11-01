"""The farmer's conversations: which threads are theirs, and what was said and found in each.

The orchestrator's checkpointer already keeps each thread's state, but only as the agent's own
messages, keyed by thread_id alone. This service adds what a product needs on top, in the same
LangGraph store the orchestrator uses for farmer memory:

- ("threads", farmer_id) → one item per conversation: its title, timestamps and turn count.
  It is the owner index: a thread that is not here is not the farmer's.
- ("turns", thread_id) → one item per message: the Turn the farmer saw, evidence included.
  Findings are cleared on every new message, so this is the only record of past evidence.
"""

from datetime import UTC, datetime

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from agrivibe.models.chat import Turn
from agrivibe.models.threads import Thread, ThreadSummary

TITLE_LENGTH = 80
LIMIT = 1000  # the store returns 10 items per search unless told otherwise


class ThreadNotFound(LookupError):
    """The thread does not exist, or it is not this farmer's. The API answers 404 either way."""


def title_for(message: str) -> str:
    """The first message, cut to fit a list."""
    message = " ".join(message.split())
    return message if len(message) <= TITLE_LENGTH else message[: TITLE_LENGTH - 1].rstrip() + "…"


class ThreadService:
    """Lists, reads, records and deletes the farmer's conversations."""

    def __init__(self, store: BaseStore, checkpointer: BaseCheckpointSaver):
        """Initializes the service.

        Args:
            store: The orchestrator's store. Holds the thread index and every turn.
            checkpointer: The orchestrator's checkpointer. Deleting a thread clears its state here too.
        """
        self.store = store
        self.checkpointer = checkpointer

    async def list(self, farmer_id: str) -> list[ThreadSummary]:
        """Returns the farmer's conversations, the most recently used first."""
        items = await self.store.asearch(("threads", farmer_id), limit=LIMIT)
        threads = [ThreadSummary(thread_id=item.key, **item.value) for item in items]
        return sorted(threads, key=lambda thread: thread.updated_at, reverse=True)

    async def summary(self, farmer_id: str, thread_id: str) -> ThreadSummary:
        """Returns one of the farmer's conversations without its history.

        Raises:
            ThreadNotFound: It does not exist or belongs to someone else.
        """
        item = await self.store.aget(("threads", farmer_id), thread_id)
        if item is None:
            raise ThreadNotFound(thread_id)
        return ThreadSummary(thread_id=thread_id, **item.value)

    async def get(self, farmer_id: str, thread_id: str) -> Thread:
        """Returns one of the farmer's conversations with every turn, oldest first.

        Raises:
            ThreadNotFound: It does not exist or belongs to someone else.
        """
        summary = await self.summary(farmer_id, thread_id)
        items = await self.store.asearch(("turns", thread_id), limit=LIMIT)
        history = [Turn(**item.value) for item in sorted(items, key=lambda item: item.key)]
        return Thread(**summary.model_dump(), history=history)

    async def record_turn(self, farmer_id: str, thread_id: str, turn: Turn) -> None:
        """Saves one turn and updates the thread's entry in the farmer's index, creating it on the first turn."""
        item = await self.store.aget(("threads", farmer_id), thread_id)
        entry = item.value if item else {"title": title_for(turn.message), "created_at": turn.started_at.isoformat(), "turns": 0}
        number = entry["turns"]
        # Zero-padded keys sort in the order the turns happened.
        await self.store.aput(("turns", thread_id), f"{number:05d}", turn.model_dump(mode="json"))
        await self.store.aput(
            ("threads", farmer_id),
            thread_id,
            {**entry, "turns": number + 1, "updated_at": datetime.now(UTC).isoformat()},
        )

    async def delete(self, farmer_id: str, thread_id: str) -> None:
        """Deletes one of the farmer's conversations: its index entry, its turns and the orchestrator's state.

        Raises:
            ThreadNotFound: It does not exist or belongs to someone else.
        """
        await self.summary(farmer_id, thread_id)
        for item in await self.store.asearch(("turns", thread_id), limit=LIMIT):
            await self.store.adelete(("turns", thread_id), item.key)
        await self.checkpointer.adelete_thread(thread_id)
        await self.store.adelete(("threads", farmer_id), thread_id)
