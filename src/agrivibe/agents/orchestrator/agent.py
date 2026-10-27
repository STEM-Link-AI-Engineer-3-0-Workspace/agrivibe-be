from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from agrivibe.agents.orchestrator.memory import AgriVibeState, FarmerContext, remember_the_farm
from agrivibe.agents.orchestrator.prompts import SYSTEM_PROMPT
from agrivibe.agents.orchestrator.tools import make_specialist_tools, save_requirements
from agrivibe.core.mcp import FieldSat, pick_tools


def build_orchestrator(
    model: BaseChatModel,
    fieldsat: FieldSat,
    weather_agent,
    farm_context_agent,
    knowledge_agent,
    advisory_agent,
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
):
    """Builds the orchestrator: the one agent the farmer talks to.

    It works out what the farmer needs and calls the specialists as tools. Only the
    orchestrator keeps the conversation; every specialist starts fresh on each call.

    Args:
        model: The chat model that decides which specialists to call and writes the reply.
        fieldsat: The FieldSat server's tools and instructions. The orchestrator keeps list_fields,
            to find the farmer's field, and its prompt ends with the server's instructions.
        weather_agent: Built by build_weather_agent.
        farm_context_agent: Built by build_farm_context_agent.
        knowledge_agent: Built by build_fieldoracle_agent, without a checkpointer.
        advisory_agent: Built by build_advisory_agent.
        checkpointer: Saves each thread's state (messages, requirements, findings), so the
            farmer can ask follow-up questions.
        store: Long-term memory shared by every thread. Holds each farmer's saved field.

    Returns:
        A compiled agent graph. Run it with `ainvoke`, a `thread_id` and a `FarmerContext`.
    """
    tools = [
        *pick_tools(fieldsat.tools, "list_fields"),
        save_requirements,
        *make_specialist_tools(weather_agent, farm_context_agent, knowledge_agent, advisory_agent),
    ]
    return create_agent(
        model,
        tools=tools,
        system_prompt=fieldsat.prompt(SYSTEM_PROMPT),
        middleware=[remember_the_farm],
        state_schema=AgriVibeState,
        context_schema=FarmerContext,
        checkpointer=checkpointer,
        store=store,
        name="orchestrator",
    )
