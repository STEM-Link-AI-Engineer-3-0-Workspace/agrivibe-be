from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.weather.prompts import SYSTEM_PROMPT
from agrivibe.core.mcp import pick_tools


def build_weather_agent(model: BaseChatModel, fieldsat_tools: list[BaseTool]):
    """Builds the Weather Agent: recent and forecast rain for one field, from FieldSat.

    Args:
        model: The chat model that picks the field and writes the report.
        fieldsat_tools: Every tool the FieldSat server offers. This agent keeps two.

    Returns:
        A compiled agent graph. Its last message is the weather report. Run it with
        `ainvoke`: MCP tools are async, so `invoke` cannot call them.
    """
    return create_agent(
        model,
        tools=pick_tools(fieldsat_tools, "list_fields", "get_weather_context"),
        system_prompt=SYSTEM_PROMPT,
        name="weather_agent",
    )
