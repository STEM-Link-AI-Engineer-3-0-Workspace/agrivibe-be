import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.weather import build_weather_agent
from tests.fakes import ScriptedChatModel, load_fake_fieldsat_tools


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def tool_names(agent):
    """The names of the tools an agent was built with."""
    tool_node = agent.nodes["tools"].bound
    assert isinstance(tool_node, ToolNode)
    return set(tool_node.tools_by_name)


def run(agent, question):
    return asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": question}]}))


def test_weather_agent_reads_fieldsat():
    tools = asyncio.run(load_fake_fieldsat_tools())
    model = ScriptedChatModel(replies=[
        tool_call("get_weather_context", {"field_id": "SL-PADDY-MINNERIYA"}),
        AIMessage(content="SL-PADDY-MINNERIYA: 112 mm of rain in the last 7 days, 46 mm expected in 48 hours."),
    ])

    result = run(build_weather_agent(model, tools), "How much rain has Minneriya had?")

    [reply] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert reply.name == "get_weather_context"
    assert reply.artifact["structured_content"]["rainfall_last_7_days_mm"] == 112.0
    assert result["messages"][-1].text.startswith("SL-PADDY-MINNERIYA")


def test_weather_agent_only_gets_weather_tools():
    tools = asyncio.run(load_fake_fieldsat_tools())
    agent = build_weather_agent(ScriptedChatModel(replies=[]), tools)

    assert tool_names(agent) == {"list_fields", "get_weather_context"}


def test_farm_context_agent_finds_the_field_then_reads_the_satellite():
    tools = asyncio.run(load_fake_fieldsat_tools())
    model = ScriptedChatModel(replies=[
        tool_call("list_fields", {}),
        tool_call("get_satellite_analysis", {"field_id": "SL-PADDY-MINNERIYA", "comparison_days": 14}),
        AIMessage(content="SL-PADDY-MINNERIYA: NDVI fell 15.2% in 14 days, from 0.66 to 0.56. The field is uneven."),
    ])

    result = run(build_farm_context_agent(model, tools), "Is the Minneriya field greener than two weeks ago?")

    replies = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.name for m in replies] == ["list_fields", "get_satellite_analysis"]
    assert replies[1].artifact["structured_content"]["trend"] == "significant_decline"


def test_farm_context_agent_only_gets_satellite_tools():
    tools = asyncio.run(load_fake_fieldsat_tools())
    agent = build_farm_context_agent(ScriptedChatModel(replies=[]), tools)

    assert tool_names(agent) == {"list_fields", "get_satellite_analysis"}
