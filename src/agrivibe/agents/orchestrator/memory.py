"""Central memory: what the orchestrator and the specialists share, and for how long.

- Context (`FarmerContext`): who is asking. The app passes it with every run. Agents read it
  and never change it.
- State (`AgriVibeState`): one conversation, saved per thread_id by the checkpointer.
  `messages` belong to the orchestrator alone. `requirements` and `findings` are the shared
  workflow state: specialist tools read the requirements and write their findings.
- Store: what outlives a conversation. Each farmer's field, under ("farmers", user_id).
"""

from dataclasses import dataclass
from typing import Annotated, Any, NotRequired, TypedDict

from langchain.agents import AgentState
from langchain.agents.middleware import ModelRequest, dynamic_prompt


@dataclass
class FarmerContext:
    """Who is asking. Passed with every run."""

    user_id: str


class Requirements(TypedDict):
    """What the orchestrator gathered from the farmer before calling a specialist."""

    field_id: str
    crop: str
    problem: str
    timeframe: str


class Finding(TypedDict):
    """One specialist's report: a summary to read, and the data it is based on."""

    summary: str
    data: dict[str, Any]  # FieldSat's reply, or FieldOracle's found and citations; empty if it failed


def merge_findings(current: dict[str, Finding], update: dict[str, Finding]) -> dict[str, Finding]:
    """Combines the findings that specialists write.

    Specialists called in parallel finish in the same step, each under its own key, so
    their updates are merged instead of one replacing the other. An empty update clears
    the findings: the service sends one with every new message, so the evidence gathered
    for one question never leaks into the next.
    """
    if not update:
        return {}
    return current | update


class AgriVibeState(AgentState):
    """The orchestrator's state: its messages, plus the shared workflow state."""

    requirements: NotRequired[Requirements]
    findings: Annotated[dict[str, Finding], merge_findings]


@dynamic_prompt
def remember_the_farm(request: ModelRequest[FarmerContext]) -> str:
    """Adds the farmer's saved field, from the store, to the orchestrator's system prompt.

    This runs before every model call, so a new conversation already knows the field the
    farmer talked about last time.
    """
    prompt = request.system_prompt or ""
    store = request.runtime.store
    # Without a store (the orchestrator was built without one) there is nothing to remember.
    saved = store.get(("farmers", request.runtime.context.user_id), "farm") if store else None
    if saved is None:
        return prompt
    farm = saved.value
    return prompt + f"\n\nThis farmer's saved field is {farm['field_id']} ({farm['crop']}). Use it when they do not name a field."
