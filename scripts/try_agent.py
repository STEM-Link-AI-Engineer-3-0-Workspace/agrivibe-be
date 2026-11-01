"""Run one specialist agent from the terminal, before the API uses it.

The FieldSat agents need FieldSat running, in your Level 6 Lesson 3 repo:
    uv run fieldsat --demo --transport http --port 8001

Then, in this repo:
    uv run python scripts/try_agent.py weather SL-PADDY-MINNERIYA "How much rain has the field had this week?"
    uv run python scripts/try_agent.py farm_context SL-PADDY-MINNERIYA "Is the field greener than two weeks ago?"

FieldOracle needs only Pinecone. It is the direct way to test the knowledge agent, without the orchestrator:
    uv run python scripts/try_agent.py knowledge "When is the first top dressing due for Bg 300?"
"""

import asyncio
import json
import sys

from langchain_core.messages import AIMessage, ToolMessage

from agrivibe.agents.farm_context import build_farm_context_agent
from agrivibe.agents.fieldoracle import build_fieldoracle_agent, cited_sources
from agrivibe.agents.fieldsat import FieldContext
from agrivibe.agents.weather import build_weather_agent
from agrivibe.config import Settings
from agrivibe.core.llm import create_chat_model
from agrivibe.core.mcp import load_fieldsat
from agrivibe.core.reranker import Reranker
from agrivibe.db.vectorstore import create_vector_store

BUILDERS = {
    "weather": build_weather_agent,
    "farm_context": build_farm_context_agent,
}


async def try_fieldsat_agent(settings, name, field_id, question):
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


async def try_knowledge_agent(settings, question):
    reranker = Reranker(settings) if settings.reranker != "none" else None
    agent = build_fieldoracle_agent(
        create_chat_model(settings), create_vector_store(settings), k=settings.search_k, reranker=reranker
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "search_knowledge":
                    print("Searched:", call["args"])
    answer = result["structured_response"]
    citations = list(dict.fromkeys(c for s in answer.sentences for c in s.citations))
    print()
    print(answer.model_dump_json(indent=2))
    print()
    for source in cited_sources(result["messages"], citations):
        print(json.dumps({key: value for key, value in source.items() if key != "text"}))


async def main(args):
    settings = Settings()  # pyright: ignore[reportCallIssue]
    if args[0] == "knowledge":
        await try_knowledge_agent(settings, args[1])
    else:
        await try_fieldsat_agent(settings, args[0], args[1], args[2])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
