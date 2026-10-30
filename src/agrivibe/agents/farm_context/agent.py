from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.farm_context.prompts import SYSTEM_PROMPT
from agrivibe.agents.fieldsat import FieldContext, FieldSatReport
from agrivibe.core.mcp import FieldSat, pick_tools


def build_farm_context_agent(model: BaseChatModel, fieldsat: FieldSat):
    """Builds the Farm Context Agent: satellite field health for one field, from FieldSat.

    Args:
        model: The chat model that reads the satellite analysis and writes the summary.
        fieldsat: The FieldSat server's tools and instructions. This agent keeps get_satellite_analysis,
            and its prompt ends with the server's instructions.

    Returns:
        A compiled agent graph. Run it with `ainvoke` and `context=FieldContext(field_id)`,
        because MCP tools are async. The result holds a `FieldReport` under `structured_response`.
    """
    return create_agent(
        model,
        tools=pick_tools(fieldsat.tools, "get_satellite_analysis"),
        system_prompt=fieldsat.prompt(SYSTEM_PROMPT),
        middleware=[FieldSatReport()],
        context_schema=FieldContext,
        name="farm_context_agent",
    )
