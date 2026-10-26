import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from agrivibe.agents.farm_context import FieldHealthReport, build_farm_context_agent
from agrivibe.agents.weather import WeatherReport, build_weather_agent
from tests.fakes import ScriptedChatModel, load_fake_fieldsat_tools

MINNERIYA = "SL-PADDY-MINNERIYA"

WEATHER = {
    "field_id": MINNERIYA,
    "available": True,
    "summary": "112 mm of rain fell on 5 of the last 7 days, 38 mm of it in the last 48 hours. 46 mm is forecast.",
    "rainfall_last_7_days_mm": 112.0,
    "rainy_days_last_7_days": 5,
    "rainfall_last_48_hours_mm": 38.0,
    "rainfall_forecast_next_48_hours_mm": 46.0,
    "average_humidity_last_7_days": 88.0,
    "heavy_rain_recent": True,
    "heavy_rain_expected": True,
}

REPORTS = {"WeatherReport", "FieldHealthReport"}

FIELD_HEALTH = {
    "field_id": MINNERIYA,
    "available": True,
    "summary": "NDVI fell 15.2% in 14 days, from 0.66 to 0.56, and the field is uneven. The data quality is good.",
    "comparison_days": 14,
    "current_mean_ndvi": 0.56,
    "previous_mean_ndvi": 0.66,
    "ndvi_change_percentage": -15.2,
    "trend": "significant_decline",
    "ndvi_standard_deviation": 0.17,
    "field_uniformity": "uneven",
    "valid_observations": 4,
    "average_cloud_coverage_percentage": 11.0,
    "data_quality": "good",
}


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def tool_names(agent):
    """The names of the tools an agent was built with."""
    tool_node = agent.nodes["tools"].bound
    assert isinstance(tool_node, ToolNode)
    return set(tool_node.tools_by_name)


def fieldsat_replies(result):
    """What FieldSat answered. The structured answer adds a ToolMessage of its own, which this skips."""
    return [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name not in REPORTS]


def run(agent, question):
    return asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": question}]}))


def test_weather_agent_reads_fieldsat():
    tools = asyncio.run(load_fake_fieldsat_tools())
    model = ScriptedChatModel(replies=[
        tool_call("get_weather_context", {"field_id": MINNERIYA}),
        tool_call("WeatherReport", WEATHER),
    ])

    result = run(build_weather_agent(model, tools), "How much rain has Minneriya had?")

    [reply] = fieldsat_replies(result)
    assert reply.name == "get_weather_context"
    assert reply.artifact["structured_content"]["rainfall_last_7_days_mm"] == 112.0
    # The answer is a WeatherReport: a worded summary plus the numbers, typed.
    report = result["structured_response"]
    assert isinstance(report, WeatherReport)
    assert report.available
    assert report.rainfall_last_7_days_mm == 112.0
    assert report.heavy_rain_expected is True
    assert "112 mm" in report.summary


def test_weather_agent_only_gets_weather_tools():
    tools = asyncio.run(load_fake_fieldsat_tools())
    agent = build_weather_agent(ScriptedChatModel(replies=[]), tools)

    assert tool_names(agent) == {"list_fields", "get_weather_context"}


def test_farm_context_agent_finds_the_field_then_reads_the_satellite():
    tools = asyncio.run(load_fake_fieldsat_tools())
    model = ScriptedChatModel(replies=[
        tool_call("list_fields", {}),
        tool_call("get_satellite_analysis", {"field_id": MINNERIYA, "comparison_days": 14}),
        tool_call("FieldHealthReport", FIELD_HEALTH),
    ])

    result = run(build_farm_context_agent(model, tools), "Is the Minneriya field greener than two weeks ago?")

    replies = fieldsat_replies(result)
    assert [m.name for m in replies] == ["list_fields", "get_satellite_analysis"]
    assert replies[1].artifact["structured_content"]["trend"] == "significant_decline"
    report = result["structured_response"]
    assert isinstance(report, FieldHealthReport)
    assert report.trend == "significant_decline"
    assert report.ndvi_change_percentage == -15.2
    assert report.data_quality == "good"


def test_a_missing_reading_is_reported_as_unavailable():
    tools = asyncio.run(load_fake_fieldsat_tools())
    unavailable = {key: None for key in WEATHER} | {
        "field_id": MINNERIYA,
        "available": False,
        "summary": "The weather source did not respond, so there is no rain data for this field.",
    }
    model = ScriptedChatModel(replies=[tool_call("WeatherReport", unavailable)])

    report = run(build_weather_agent(model, tools), "Rain on Minneriya?")["structured_response"]

    assert not report.available
    assert report.rainfall_last_7_days_mm is None


def test_farm_context_agent_only_gets_satellite_tools():
    tools = asyncio.run(load_fake_fieldsat_tools())
    agent = build_farm_context_agent(ScriptedChatModel(replies=[]), tools)

    assert tool_names(agent) == {"list_fields", "get_satellite_analysis"}
