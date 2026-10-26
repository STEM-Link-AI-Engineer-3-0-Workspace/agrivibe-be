from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.farm_context.prompts import SYSTEM_PROMPT
from agrivibe.agents.farm_context.schemas import FieldHealthReport
from agrivibe.core.mcp import pick_tools


def build_farm_context_agent(model: BaseChatModel, fieldsat_tools: list[BaseTool]):
    """Builds the Farm Context Agent: satellite field health for one field, from FieldSat.

    Args:
        model: The chat model that picks the field and writes the report.
        fieldsat_tools: Every tool the FieldSat server offers. This agent keeps two.

    Returns:
        A compiled agent graph. Its result holds a `FieldHealthReport` under
        `structured_response`. Run it with `ainvoke`, because MCP tools are async.
    """
    return create_agent(
        model,
        tools=pick_tools(fieldsat_tools, "list_fields", "get_satellite_analysis"),
        system_prompt=SYSTEM_PROMPT,
        response_format=FieldHealthReport,
        name="farm_context_agent",
    )
