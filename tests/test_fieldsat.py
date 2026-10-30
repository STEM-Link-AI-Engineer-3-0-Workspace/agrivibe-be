import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldsat import FieldContext
from agrivibe.core.mcp import load_fieldsat, pick_tools
from tests.fakes import FAKE_FIELDSAT_INSTRUCTIONS, ScriptedChatModel, load_fake_fieldsat


def test_server_tools_become_langchain_tools():
    fieldsat = asyncio.run(load_fake_fieldsat())

    assert [tool.name for tool in fieldsat.tools] == ["list_fields", "get_satellite_analysis", "get_weather_context"]

    [weather] = pick_tools(fieldsat.tools, "get_weather_context")
    call = {"type": "tool_call", "id": "1", "name": "get_weather_context", "args": {"field_id": "SL-PADDY-MINNERIYA"}}
    message = asyncio.run(weather.ainvoke(call))

    assert isinstance(message, ToolMessage)
    assert message.artifact["structured_content"]["rainfall_last_7_days_mm"] == 112.0


def test_the_server_instructions_reach_the_agent_prompt():
    fieldsat = asyncio.run(load_fake_fieldsat())
    assert fieldsat.instructions == FAKE_FIELDSAT_INSTRUCTIONS

    model = ScriptedChatModel(replies=[AIMessage(content="No reading.")])
    agent = build_farm_context_agent(model, fieldsat)
    asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Report"}]}, context=FieldContext("SL-PADDY-MINNERIYA"))
    )

    system_prompt = model.received[0][0].text
    assert system_prompt.endswith(FAKE_FIELDSAT_INSTRUCTIONS)


def test_pick_tools_rejects_a_missing_tool():
    fieldsat = asyncio.run(load_fake_fieldsat())

    with pytest.raises(KeyError):
        pick_tools(fieldsat.tools, "get_soil_moisture")


def test_unreachable_server_says_how_to_start_it():
    with pytest.raises(RuntimeError, match="uv run fieldsat"):
        asyncio.run(load_fieldsat("http://127.0.0.1:9/mcp"))
