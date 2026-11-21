"""Add the Lesson 3 model's missing prices to self-hosted Langfuse (safe to re-run).

Run with the same Langfuse settings as the backend:
    uv run python scripts/setup_langfuse.py

Prices checked 2026-11-21 against:
https://developers.openai.com/api/docs/models/gpt-5.6-luna
USD per million tokens: input 0.20, cache read 0.02, cache write 0.25,
output (including reasoning) 1.20. Above 272K input tokens: input x2, output x1.5.
Existing model definitions are preserved; review rates when changing models.
"""

from langfuse.api import PricingTierInput, PricingTierUsageConditionInput

from agrivibe.config import Settings
from agrivibe.core.tracing import create_langfuse


def setup(client):
    page = 1
    while True:
        models = client.api.models.list(page=page, limit=100)
        if any(m.model_name == "gpt-5.6-luna" for m in models.data):
            print("gpt-5.6-luna already has a model definition; preserved it.")
            return
        if page >= models.meta.total_pages:
            break
        page += 1
    prices = {
        "input": 0.20 / 1_000_000,
        "input_cache_read": 0.02 / 1_000_000,
        "input_cache_creation": 0.25 / 1_000_000,
        "output": 1.20 / 1_000_000,
        "output_reasoning": 1.20 / 1_000_000,
    }
    client.api.models.create(
        model_name="gpt-5.6-luna",
        match_pattern=r"(?i)^(openai:)?gpt-5\.6-luna(-\d{4}-\d{2}-\d{2})?$",
        unit="TOKENS",
        pricing_tiers=[
            PricingTierInput(name="Standard", is_default=True, priority=0, conditions=[], prices=prices),
            PricingTierInput(
                name="Long context (>272K input)", is_default=False, priority=1,
                conditions=[PricingTierUsageConditionInput(
                    usage_detail_pattern="^input", operator="gt", value=272_000,
                )],
                prices={k: v * (2 if k.startswith("input") else 1.5) for k, v in prices.items()},
            ),
        ],
    )
    print("Added gpt-5.6-luna pricing. Costs apply to newly ingested generations.")


if __name__ == "__main__":
    client = create_langfuse(Settings())
    if client is None:
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY first.")
    setup(client)
