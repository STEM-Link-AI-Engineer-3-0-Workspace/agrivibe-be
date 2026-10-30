"""Run one specialist agent from the terminal, before the API uses it.

Start FieldSat first, in your Level 6 Lesson 3 repo:
    uv run fieldsat --demo --transport http --port 8001

Then, in this repo:
    uv run python scripts/try_agent.py weather SL-PADDY-MINNERIYA "How much rain has the field had this week?"
    uv run python scripts/try_agent.py farm_context SL-PADDY-MINNERIYA "Is the field greener than two weeks ago?"
"""

import asyncio
import sys

from langchain_core.messages import AIMessage, ToolMessage

from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldsat import FieldContext
from agrivibe.agents.weather import build_weather_agent
from agrivibe.config import Settings
from agrivibe.core.llm import create_chat_model
from agrivibe.core.mcp import load_fieldsat

BUILDERS = {
    "weather": build_weather_agent,
    "farm_context": build_farm_context_agent,
}


async def main(name, field_id, question):
    settings = Settings()  # pyright: ignore[reportCallIssue]
    fieldsat = await load_fieldsat(settings.fieldsat_url)
    agent = BUILDERS[name](create_chat_model(settings), fieldsat)

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": f"Field: {field_id}\n\n{question}"}]},
        context=FieldContext(field_id=field_id),
    )

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            for call in message.tool_calls:
                print("Agent called:", call["name"], call["args"])
        if isinstance(message, ToolMessage):
            print("FieldSat answered:", message.name, message.status)
    print()
    print(result["structured_response"].model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3]))
