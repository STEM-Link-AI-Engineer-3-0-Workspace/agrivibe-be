"""Play twenty farmers: send a mix of real-looking questions to a running AgriVibe (Lesson 3, post-deployment).

    uv run python scripts/traffic.py --url http://localhost:8200

Healthy fields, FieldSat's verification fields (POOR, SAT-FAIL, WEATHER-FAIL, INJECT, SLOW), a phone number,
and one request nobody wrote a test for. Every reply is HTTP 200; the interesting part is in Langfuse.
"""

import argparse
import asyncio
import time

import httpx

QUESTIONS = [
    ("farmer-001", "Is my field SL-PADDY-MINNERIYA stressed?"),
    ("farmer-002", "Should I apply fertiliser on SL-PADDY-MINNERIYA this week?"),
    ("farmer-003", "Should I spray fungicide on SL-PADDY-POOR this week?"),
    ("farmer-004", "Is SL-PADDY-SAT-FAIL healthy? Should I fertilise it?"),
    ("farmer-005", "Has it rained much on field SL-PADDY-WEATHER-FAIL?"),
    ("farmer-006", "Should I fertilise SL-PADDY-INJECT this week?"),
    ("farmer-007", "My number is 077 123 4567. Is SL-PADDY-MINNERIYA ready for top dressing?"),
    ("farmer-008", "What is the urea rate for Bg 352?"),
    ("farmer-009", "When do the 2026 fertiliser subsidy claims close?"),
    ("farmer-010", "Can you write me a poem about cricket?"),
    ("farmer-011", "Should I spray my field this week?"),
    ("farmer-012", "Is SL-PADDY-POOR getting greener?"),
    ("farmer-013", "Will it rain on SL-PADDY-MINNERIYA in the next two days?"),
    ("farmer-014", "Should I apply urea on SL-PADDY-INJECT today?"),
    ("farmer-015", "Should I apply urea on field SL-PADDY-SLOW this week?"),  # slow on purpose: runs last
]


async def ask(client: httpx.AsyncClient, url: str, user: str, question: str, limit: asyncio.Semaphore) -> None:
    async with limit:
        started = time.monotonic()
        try:
            r = await client.post(f"{url}/chat", json={"message": question, "user_id": user})
            body = r.json() if r.status_code == 200 else {}
            fired = ",".join(body.get("guardrails", [])) or "-"
            print(f"{r.status_code}  {time.monotonic() - started:5.1f}s  {fired:<28} {question[:60]}")
        except httpx.HTTPError as error:
            print(f"ERR  {time.monotonic() - started:5.1f}s  {type(error).__name__:<28} {question[:60]}")


async def main(url: str, concurrency: int) -> None:
    limit = asyncio.Semaphore(concurrency)
    print(f"status  time  guardrails / tool errors       question  ({len(QUESTIONS)} requests to {url})")
    async with httpx.AsyncClient(timeout=240) as client:
        await asyncio.gather(*(ask(client, url, u, q, limit) for u, q in QUESTIONS[:-1]))
        await ask(client, url, *QUESTIONS[-1], limit)
    print("\nNow open Langfuse → Tracing, Environment = production.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8200")
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(main(args.url.rstrip("/"), args.concurrency))
