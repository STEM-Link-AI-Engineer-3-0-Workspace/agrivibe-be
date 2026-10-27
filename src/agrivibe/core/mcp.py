"""The FieldSat MCP server's tools, loaded as LangChain tools, and its instructions for using them.

FieldSat (built in Level 6 Lesson 3) runs as its own process. This app never imports
its code: it asks the server which tools it has, over MCP, when the app starts.

How to use the tools (which one to call first, what each argument means) is the server's
business, because only the server knows when its tools change. So the agents' own prompts
never describe the tools: each tool's description and argument schema come from the server,
and the server's instructions are added to the prompt of every agent that uses FieldSat.
"""

import warnings
from dataclasses import dataclass
from typing import cast

from fastmcp import Client, FastMCP
from langchain.tools import BaseTool

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # langchain.mcp is marked beta; the warning is expected
    from langchain.mcp import MCPAdapter

START_FIELDSAT = "uv run fieldsat --demo --transport http --port 8001"


@dataclass
class FieldSat:
    """What the FieldSat server sent when the app connected."""

    tools: list[BaseTool]  # one LangChain tool per FieldSat tool, with the server's descriptions
    instructions: str  # the server's own guidance on its tools. Empty if it sent none.

    def prompt(self, system_prompt: str) -> str:
        """Returns an agent's system prompt with the server's instructions after it."""
        if not self.instructions:
            return system_prompt
        return (
            f"{system_prompt}\n\n"
            "The FieldSat server's instructions for its tools (you may have only some of them):\n"
            f"{self.instructions}"
        )


async def load_fieldsat(url: str | FastMCP) -> FieldSat:
    """Connects to the FieldSat MCP server and reads its tools and its instructions.

    The tools remember where the server is, so they still work after this function
    returns: each call opens a session, runs the tool on the server and closes it.

    Args:
        url: The server's Streamable HTTP endpoint, for example http://127.0.0.1:8001/mcp.
            Tests pass a FastMCP server instead, which is reached in memory.

    Returns:
        The server's tools (list_fields, get_satellite_analysis and get_weather_context)
        and the instructions it sent when the session started.

    Raises:
        RuntimeError: If the server cannot be reached.
    """
    try:
        async with MCPAdapter(url) as adapter:
            tools = await adapter.list_tools()
            # Sent once, when the session starts, so it is read while still connected.
            client = cast(Client, adapter.client)  # one server gives one client, never a ClientGroup
            return FieldSat(tools=tools, instructions=client.instructions or "")
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
