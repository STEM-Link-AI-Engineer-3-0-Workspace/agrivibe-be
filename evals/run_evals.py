"""Run the golden set against the real model, N times, and score every answer (Lesson 3, tier 3).

    uv run python -m evals.run_evals --runs 3              # smoke set, 3 runs (what the demo uses)
    uv run python -m evals.run_evals --set full --runs 1   # everything
    uv run python -m evals.run_evals --set smoke --runs 2 --no-langfuse   # what CI runs

Each answer gets deterministic checks first (trajectory, forbidden tools/text, step budget), then the judge.
An item passes a run only if every check passes. With Langfuse keys, the golden set is uploaded as the
dataset `agrivibe-golden` and each run is a Langfuse experiment run you can compare in the UI.
Writes evals/results/latest.json for evals.gate.
"""

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from langfuse import Evaluation
from langfuse.langchain import CallbackHandler

from agrivibe.core.tracing import create_langfuse
from agrivibe.dependencies import get_settings
from evals.harness import Item, build_service, deterministic_checks, evidence_text, load_items, pinecone_available
from evals.judge import Judge

DATASET = "agrivibe-golden"
RESULTS = Path(__file__).parent / "results"


class Runner:
    """Answers and scores golden items, with the agents and judge built once (in the one event loop we use)."""

    def __init__(self, service, judge: Judge, settings, langfuse=None):
        self.service, self.judge, self.settings, self.langfuse = service, judge, settings, langfuse

    async def answer(self, item: Item, tag: str) -> dict:
        """Runs one golden question through AgriVibe. Traced into the current experiment item when on Langfuse."""
        service = self.service
        if self.langfuse is not None:
            # A plain handler nests under the experiment item's span. Safe without a lock: chat() reads
            # service.tracer before its first await, so concurrent items can't swap it mid-call.
            service.tracer = CallbackHandler(public_key=self.settings.langfuse_public_key)
        reply = await service.chat(item.input, user_id=f"eval-{item.id}-{tag}")
        return {"answer": reply.answer, "tools": reply.tools_called, "evidence": evidence_text(reply.findings),
                "guardrails": reply.guardrails}

    async def score(self, item: Item, output: dict) -> dict[str, tuple[bool, str]]:
        checks = deterministic_checks(item, output["answer"], output["tools"])
        if all(ok for ok, _ in checks.values()):  # the judge only scores what code can't
            verdict = await self.judge.grade(item.input, output["answer"], item.rubric, output["evidence"])
            checks["judge"] = (verdict.passed, verdict.reason)
        else:
            checks["judge"] = (False, "skipped: a deterministic check failed")
        return checks


def item_result(item: Item, output: dict, checks: dict) -> dict:
    return {"pass": all(ok for ok, _ in checks.values()), "answer": output["answer"], "tools": output["tools"],
            "guardrails": output["guardrails"], "checks": {k: {"pass": ok, "reason": r} for k, (ok, r) in checks.items()}}


def errored(item: Item, reason: str) -> dict:
    return {"pass": False, "answer": "", "tools": [], "guardrails": [], "checks": {"error": {"pass": False, "reason": reason}}}


async def run_local(runner: Runner, items: list[Item], run_name: str, concurrency: int) -> dict:
    """One run without Langfuse (CI): every item, `concurrency` at a time."""
    limit = asyncio.Semaphore(concurrency)

    async def one(item):
        async with limit:
            try:
                output = await runner.answer(item, run_name)
                return item.id, item_result(item, output, await runner.score(item, output))
            except Exception as error:  # noqa: BLE001 - an error is a failed item, not a crashed eval run
                return item.id, errored(item, f"{type(error).__name__}: {error}")

    return dict(await asyncio.gather(*(one(i) for i in items)))


async def run_langfuse(runner: Runner, items: list[Item], run_name: str, concurrency: int) -> dict:
    """One run as a Langfuse experiment run on the `agrivibe-golden` dataset."""
    by_question = {i.input: i for i in items}
    results: dict[str, dict] = {}

    async def task(*, item, **_):
        return await runner.answer(by_question[item.input["question"]], run_name)

    async def evaluator(*, input, output, **_):
        golden = by_question[input["question"]]
        checks = await runner.score(golden, output)
        results[golden.id] = item_result(golden, output, checks)
        scores = [Evaluation(name=name, value=1.0 if ok else 0.0, comment=reason) for name, (ok, reason) in checks.items()]
        return [*scores, Evaluation(name="item_pass", value=1.0 if results[golden.id]["pass"] else 0.0)]

    wanted = [d for d in runner.langfuse.get_dataset(DATASET).items if d.id in {i.id for i in items}]
    # The async form of Langfuse.run_experiment. The public method starts a new event loop per call, and
    # langchain-openai's shared HTTP client is bound to the first loop, so run 2 would fail with
    # "Event loop is closed". Private API: re-check it when you upgrade langfuse (pinned in uv.lock).
    await runner.langfuse._run_experiment_async(
        name=DATASET, run_name=run_name, description=f"{len(wanted)} golden items", data=wanted, task=task,
        evaluators=[evaluator], composite_evaluator=None, run_evaluators=[], max_concurrency=concurrency,
    )
    for golden in items:  # the experiment runner logs and skips an item whose task raised: that is a failure
        results.setdefault(golden.id, errored(golden, "the task or evaluator raised; see the Langfuse trace"))
    return results


def upload_golden(langfuse, items: list[Item]) -> None:
    """Puts the golden items in the Langfuse dataset. Item ids are the golden ids, so re-uploading updates them."""
    try:
        langfuse.get_dataset(DATASET)
    except Exception:  # noqa: BLE001 - first run: the dataset doesn't exist yet
        langfuse.create_dataset(name=DATASET, description="AgriVibe golden set (evals/golden.jsonl)")
    for item in items:
        langfuse.create_dataset_item(
            dataset_name=DATASET, id=item.id, input={"question": item.input},
            expected_output={"rubric": item.rubric, "trajectory": item.trajectory, "must_not": item.must_not},
            metadata={"tags": item.tags},
        )


def summarise(runs: list[dict], items: list[Item], subset: str) -> dict:
    scores = [sum(r[i.id]["pass"] for i in items) / len(items) for r in runs]
    per_item = {i.id: sum(r[i.id]["pass"] for r in runs) / len(runs) for i in items}
    return {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "set": subset, "runs": len(runs), "items": [i.id for i in items],
        "safety_items": [i.id for i in items if "safety" in i.tags],
        "score": round(statistics.mean(scores), 3), "run_scores": [round(s, 3) for s in scores],
        "spread": round(max(scores) - min(scores), 3), "per_item": per_item, "detail": runs,
    }


def print_table(summary: dict) -> None:
    width = max(len(i) for i in summary["items"]) + 2
    print(f"\n{'item':<{width}} {'pass rate':>9}   runs")
    for item_id in summary["items"]:
        marks = " ".join("✓" if run[item_id]["pass"] else "✗" for run in summary["detail"])
        print(f"{item_id:<{width}} {summary['per_item'][item_id]:>9.2f}   {marks}")
    print(f"\noverall {summary['score']:.2f}   run scores {summary['run_scores']}   spread {summary['spread']:.2f}")
    for run_no, run in enumerate(summary["detail"], 1):
        for item_id, r in run.items():
            failed = [f"{k}: {c['reason']}" for k, c in r["checks"].items() if not c["pass"]]
            if failed:
                print(f"  run {run_no} {item_id} ✗ {failed[0][:160]}")


async def main(args) -> None:
    settings = get_settings()
    items = load_items(args.set)
    use_pinecone = pinecone_available(settings) and not args.no_pinecone
    if not use_pinecone:
        items = [i for i in items if "knowledge" not in i.tags]
    langfuse = None if args.no_langfuse else create_langfuse(settings)
    service = await build_service(settings, use_pinecone)
    runner = Runner(service, Judge(settings.judge_model, settings.openai_api_key.get_secret_value()), settings, langfuse)
    if langfuse is not None:
        upload_golden(langfuse, items)
    label = args.run_name or datetime.now().strftime("%H%M%S")
    print(f"{len(items)} items × {args.runs} runs · model {settings.chat_model} · judge {settings.judge_model} · "
          f"knowledge {'Pinecone' if use_pinecone else 'off'} · Langfuse {'on' if langfuse else 'off'}")
    started, runs = time.monotonic(), []
    for n in range(1, args.runs + 1):
        run_name = f"{label}-{n}" if args.runs > 1 else label
        run = run_langfuse if langfuse is not None else run_local
        runs.append(await run(runner, items, run_name, args.concurrency))
        print(f"run {n}/{args.runs} done ({time.monotonic() - started:.0f} s)")
    summary = summarise(runs, items, args.set)
    print_table(summary)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / (args.out or "latest.json")
    out.write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\nWrote {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    if langfuse is not None:
        langfuse.flush()
        print(f"Langfuse: Datasets → {DATASET} → Runs ({label}*)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="smoke", help="smoke (default), full, or one item id")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-name", help="label for the Langfuse runs, for example baseline or after-fix")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--no-langfuse", action="store_true", help="score locally only (CI)")
    parser.add_argument("--no-pinecone", action="store_true", help="skip knowledge items (CI, or Pinecone down)")
    parser.add_argument("--out", help="results file name in evals/results (default latest.json)")
    asyncio.run(main(parser.parse_args()))
