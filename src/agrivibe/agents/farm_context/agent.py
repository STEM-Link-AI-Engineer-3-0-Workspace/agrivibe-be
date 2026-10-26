from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.farm_context.prompts import SYSTEM_PROMPT
from agrivibe.agents.fieldsat import FieldContext, FieldSatReport
from agrivibe.core.mcp import pick_tools


def build_farm_context_agent(model: BaseChatModel, fieldsat_tools: list[BaseTool]):
    """Builds the Farm Context Agent: satellite field health for one field, from FieldSat.

    Args:
        model: The chat model that reads the satellite analysis and writes the summary.
        fieldsat_tools: Every tool the FieldSat server offers. This agent keeps get_satellite_analysis.

    Returns:
        A compiled agent graph. Run it with `ainvoke` and `context=FieldContext(field_id)`,
        because MCP tools are async. The result holds a `FieldReport` under `structured_response`.
    """
    return create_agent(
        model,
        tools=pick_tools(fieldsat_tools, "get_satellite_analysis"),
        system_prompt=SYSTEM_PROMPT,
        middleware=[FieldSatReport()],
        context_schema=FieldContext,
        name="farm_context_agent",
    )
