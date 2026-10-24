"""The FieldSat MCP server's tools, loaded as LangChain tools.

FieldSat (built in Level 6 Lesson 3) runs as its own process. This app never imports
its code: it asks the server which tools it has, over MCP, when the app starts.
"""

import warnings

from langchain.tools import BaseTool

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # langchain.mcp is marked beta; the warning is expected
    from langchain.mcp import MCPAdapter

START_FIELDSAT = "uv run fieldsat --demo --transport http --port 8001"


async def load_fieldsat_tools(url: str) -> list[BaseTool]:
    """Connects to the FieldSat MCP server and turns its tools into LangChain tools.

    The tools remember where the server is, so they still work after this function
    returns: each call opens a session, runs the tool on the server and closes it.

    Args:
        url: The server's Streamable HTTP endpoint, for example http://127.0.0.1:8001/mcp.

    Returns:
        One LangChain tool per FieldSat tool: list_fields, get_satellite_analysis and
        get_weather_context.

    Raises:
        RuntimeError: If the server cannot be reached.
    """
    try:
        async with MCPAdapter(url) as adapter:
            return await adapter.list_tools()
    except Exception as error:
        raise RuntimeError(
            f"Could not reach the FieldSat MCP server at {url}. "
            f"Start it from your Level 6 Lesson 3 repo with: {START_FIELDSAT}"
        ) from error


def pick_tools(tools: list[BaseTool], *names: str) -> list[BaseTool]:
    """Returns the tools with these names, in that order.

    Each agent gets only the tools it needs. A name the server does not offer raises
    a KeyError, so a renamed tool fails at startup rather than in the middle of a chat.

    Args:
        tools: Every tool the server offers.
        names: The tools to keep.

    Returns:
        The named tools.
    """
    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in names]
