from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from agrivibe.agents.fieldsat import FieldContext, FieldSatReport
from agrivibe.agents.weather.prompts import SYSTEM_PROMPT
from agrivibe.core.mcp import FieldSat, pick_tools


def build_weather_agent(model: BaseChatModel, fieldsat: FieldSat):
    """Builds the Weather Agent: recent and forecast rain for one field, from FieldSat.

    Args:
        model: The chat model that reads the weather and writes the summary.
        fieldsat: The FieldSat server's tools and instructions. This agent keeps get_weather_context,
            and its prompt ends with the server's instructions.

    Returns:
        A compiled agent graph. Run it with `ainvoke` and `context=FieldContext(field_id)`;
        MCP tools are async, so `invoke` cannot call them. The result holds a `FieldReport`
        under `structured_response`.
    """
    return create_agent(
        model,
        tools=pick_tools(fieldsat.tools, "get_weather_context"),
        system_prompt=fieldsat.prompt(SYSTEM_PROMPT),
        middleware=[FieldSatReport()],
        context_schema=FieldContext,
        name="weather_agent",
    )
