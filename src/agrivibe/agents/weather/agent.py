from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.fieldsat import FieldContext, FieldSatReport
from agrivibe.agents.weather.prompts import SYSTEM_PROMPT
from agrivibe.core.mcp import pick_tools


def build_weather_agent(model: BaseChatModel, fieldsat_tools: list[BaseTool]):
    """Builds the Weather Agent: recent and forecast rain for one field, from FieldSat.

    Args:
        model: The chat model that reads the weather and writes the summary.
        fieldsat_tools: Every tool the FieldSat server offers. This agent keeps get_weather_context.

    Returns:
        A compiled agent graph. Run it with `ainvoke` and `context=FieldContext(field_id)`;
        MCP tools are async, so `invoke` cannot call them. The result holds a `FieldReport`
        under `structured_response`.
    """
    return create_agent(
        model,
        tools=pick_tools(fieldsat_tools, "get_weather_context"),
        system_prompt=SYSTEM_PROMPT,
        middleware=[FieldSatReport()],
        context_schema=FieldContext,
        name="weather_agent",
    )
