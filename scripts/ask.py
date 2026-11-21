"""Ask AgriVibe one question from the terminal, with the real model, and print the trace link (Lesson 3).

Needs FieldSat (FIELDSAT_URL) and, for the trace link, Langfuse keys in .env.
    uv run python scripts/ask.py "Should I fertilise SL-PADDY-INJECT this week?"
    GUARDRAILS=false uv run python scripts/ask.py "Should I fertilise SL-PADDY-INJECT this week?"
"""

import argparse
import asyncio

from agrivibe.core.mcp import load_fieldsat
from agrivibe.dependencies import build_chat_service, get_observability, get_settings


async def main(question: str, user_id: str) -> None:
    settings = get_settings()
    service = build_chat_service(settings, await load_fieldsat(settings.fieldsat_url))
    reply = await service.chat(question, user_id)

    print(f"guardrails: {'on' if settings.guardrails else 'OFF'}")
    print(f"tools:      {' → '.join(reply.tools_called) or '(none)'}")
    print(f"fired:      {', '.join(reply.guardrails) or '(nothing)'}")
    print(f"\n{reply.answer}\n")
    observability = get_observability()
    if observability is not None and reply.trace_id:
        observability.client.flush()
        print(f"trace:      {observability.trace_url(reply.trace_id)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--user", default="farmer-demo")
    args = parser.parse_args()
    asyncio.run(main(args.question, args.user))
