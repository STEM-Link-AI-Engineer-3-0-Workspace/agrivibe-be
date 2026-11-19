"""Close the loop (Lesson 3): turn a bad production turn into a golden eval item.

    uv run python -m evals.promote_trace --match poem --id off-topic-poem \\
        --expect "Politely declines or redirects to farming help. Calls no specialist."
    uv run python -m evals.promote_trace --trace-id <id> --id my-case --expect "..."

Adds the item to evals/golden.jsonl (tags: smoke, from-production), so every future PR is gated on it, and to
the Langfuse dataset `agrivibe-golden`, linked to the trace it came from. Re-running with the same --id updates it.
"""

import argparse
import json

from agrivibe.core.tracing import create_langfuse
from agrivibe.dependencies import get_settings
from evals.harness import GOLDEN
from evals.post_deploy import RESULTS
from evals.run_evals import DATASET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--worst", action="store_true", help="the first failure in evals/results/post_deploy.json")
    source.add_argument("--match", help="the first failure whose question contains this text (review first!)")
    source.add_argument("--trace-id")
    parser.add_argument("--id", required=True, help="a short name for the new golden item")
    parser.add_argument("--expect", required=True, help="the rubric: what a good answer does")
    parser.add_argument("--must-not-call", nargs="*", default=[], help="tools this question must not use")
    args = parser.parse_args()

    rows = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    if args.worst:
        row = next((r for r in rows if not r["pass"]), None)
        if row is None:
            raise SystemExit("No failing turn in evals/results/post_deploy.json. Run evals.post_deploy first.")
    elif args.match:
        row = next((r for r in rows if not r["pass"] and args.match.lower() in r["question"].lower()), None)
        if row is None:
            raise SystemExit(f"No failing turn whose question contains {args.match!r}.")
    else:
        row = next((r for r in rows if r["trace_id"] == args.trace_id), None)
        if row is None:
            raise SystemExit(f"Trace {args.trace_id} isn't in evals/results/post_deploy.json.")

    item = {"id": args.id, "input": row["question"], "trajectory": [], "must_not": [],
            "must_not_call": args.must_not_call, "rubric": args.expect, "tags": ["smoke", "from-production"]}
    lines = [line for line in GOLDEN.read_text().splitlines() if line.strip()]
    lines = [line for line in lines if json.loads(line)["id"] != args.id] + [json.dumps(item, ensure_ascii=False)]
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"Added '{args.id}' to {GOLDEN.name}: {row['question']!r}")
    print(f"  was: {row['answer'][:120]!r}")
    print(f"  why it failed: {row['reason'][:160]}")

    langfuse = create_langfuse(get_settings())
    if langfuse is not None:
        langfuse.create_dataset_item(
            dataset_name=DATASET, id=args.id, input={"question": item["input"]},
            expected_output={"rubric": item["rubric"], "trajectory": [], "must_not": []},
            metadata={"tags": item["tags"]}, source_trace_id=row["trace_id"],
        )
        langfuse.flush()
        print(f"  and to the Langfuse dataset {DATASET}, linked to trace {row['trace_id']}")


if __name__ == "__main__":
    main()
