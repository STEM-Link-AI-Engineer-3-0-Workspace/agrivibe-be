import asyncio

import pytest
from langchain_core.messages import ToolMessage

from agrivibe.core.mcp import load_fieldsat_tools, pick_tools
from tests.fakes import load_fake_fieldsat_tools


def test_server_tools_become_langchain_tools():
    tools = asyncio.run(load_fake_fieldsat_tools())

    assert [tool.name for tool in tools] == ["list_fields", "get_satellite_analysis", "get_weather_context"]

    [weather] = pick_tools(tools, "get_weather_context")
    call = {"type": "tool_call", "id": "1", "name": "get_weather_context", "args": {"field_id": "SL-PADDY-MINNERIYA"}}
    message = asyncio.run(weather.ainvoke(call))

    assert isinstance(message, ToolMessage)
    assert message.artifact["structured_content"]["rainfall_last_7_days_mm"] == 112.0


def test_pick_tools_rejects_a_missing_tool():
    tools = asyncio.run(load_fake_fieldsat_tools())

    with pytest.raises(KeyError):
        pick_tools(tools, "get_soil_moisture")


def test_unreachable_server_says_how_to_start_it():
    with pytest.raises(RuntimeError, match="uv run fieldsat"):
        asyncio.run(load_fieldsat_tools("http://127.0.0.1:9/mcp"))
