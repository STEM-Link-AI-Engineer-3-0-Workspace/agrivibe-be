import asyncio

from langchain_core.messages import AIMessage

from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldsat import FieldContext
from agrivibe.agents.weather import build_weather_agent
from tests.fakes import ScriptedChatModel, load_fake_fieldsat_tools

MINNERIYA = "SL-PADDY-MINNERIYA"


def tool_call(name, args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def report(build, replies, field_id=MINNERIYA):
    """Runs a specialist with scripted replies and returns its FieldReport."""
    agent = build(ScriptedChatModel(replies=replies), asyncio.run(load_fake_fieldsat_tools()))
    result = asyncio.run(
        agent.ainvoke({"messages": [{"role": "user", "content": "Report"}]}, context=FieldContext(field_id))
    )
    return result["structured_response"]


def test_weather_report_is_the_summary_plus_fieldsat_data():
    weather = report(build_weather_agent, [
        tool_call("get_weather_context", {"field_id": MINNERIYA}),
        AIMessage(content="112 mm of rain fell in the last 7 days."),
    ])

    assert weather.field_id == MINNERIYA
    assert weather.summary == "112 mm of rain fell in the last 7 days."
    # The numbers are FieldSat's own reply, not copied by the model.
    assert weather.data["rainfall_last_7_days_mm"] == 112.0
    assert weather.data["heavy_rain_expected"] is True


def test_the_field_comes_from_the_context_not_the_model():
    health = report(build_farm_context_agent, [
        tool_call("get_satellite_analysis", {"field_id": "SOME-OTHER-FIELD", "comparison_days": 14}),
        AIMessage(content="NDVI fell 15.2% in 14 days."),
    ])

    # FieldSat was asked about the context's field, whatever the model wrote.
    assert health.data["field_id"] == MINNERIYA
    assert health.data["trend"] == "significant_decline"


def test_a_failed_reading_has_no_data():
    weather = report(build_weather_agent, [
        tool_call("get_weather_context", {"field_id": "SL-PADDY-WEATHER-FAIL"}),
        AIMessage(content="The weather source is unavailable, so there is no rain data."),
    ], field_id="SL-PADDY-WEATHER-FAIL")

    assert weather.data is None
    assert "unavailable" in weather.summary
