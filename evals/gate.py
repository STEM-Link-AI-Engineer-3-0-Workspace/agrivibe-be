"""The CI gate (Lesson 3): decide pass/fail from an eval results file, the committed baseline and gate.toml.

    uv run python -m evals.gate                                  # gate evals/results/latest.json
    uv run python -m evals.gate --results evals/results/x.json
    uv run python -m evals.gate --write-baseline                 # make latest.json the new baseline (review the diff!)

Three rules, all must hold:
  1. floor        overall score >= floor
  2. no regression overall score >= baseline score - margin
  3. safety       every item tagged with a safety tag passed in every run
Exit code 1 when a rule fails, which turns the CI check red. Writes a markdown table to $GITHUB_STEP_SUMMARY in CI.
"""

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).parent


def decide(results: dict, baseline: dict, gate: dict) -> list[tuple[str, bool, str]]:
    score, base = results["score"], baseline["score"]
    unsafe = [i for i in results["safety_items"] if results["per_item"][i] < 1.0]
    return [
        ("floor", score >= gate["floor"], f"{score:.2f} ≥ {gate['floor']:.2f}"),
        ("no regression", score >= base - gate["margin"], f"{score:.2f} ≥ {base:.2f} − {gate['margin']:.2f}"),
        ("safety items pass every run", not unsafe, ", ".join(unsafe) + " failed" if unsafe else "all passed"),
    ]


def report(results: dict, baseline: dict, rules: list[tuple[str, bool, str]]) -> str:
    ok = all(passed for _, passed, _ in rules)
    lines = [f"## Eval gate: {'✅ PASS' if ok else '❌ FAIL'}", "",
             (f"{len(results['items'])} items × {results['runs']} runs · score **{results['score']:.2f}** "
              f"(baseline {baseline['score']:.2f}) · spread {results['spread']:.2f}"), "",
             "| Rule | Result | Check |", "|---|---|---|"]
    lines += [f"| {name} | {'✅' if passed else '❌'} | {detail} |" for name, passed, detail in rules]
    lines += ["", "| Item | Pass rate | Baseline | Why it failed (first failing run) |", "|---|---:|---:|---|"]
    for item in results["items"]:
        rate, was = results["per_item"][item], baseline.get("per_item", {}).get(item)
        why = next((f"{k}: {c['reason']}" for run in results["detail"] for k, c in run[item]["checks"].items()
                    if not c["pass"]), "")
        flag = " ⬇️" if was is not None and rate < was else ""
        was_text = f"{was:.2f}" if was is not None else "new"
        lines.append(f"| {item} | {rate:.2f}{flag} | {was_text} | {why[:180].replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "results" / "latest.json"))
    parser.add_argument("--baseline", default=str(HERE / "baseline.json"))
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    results = json.loads(Path(args.results).read_text())

    if args.write_baseline:
        keep = {k: results[k] for k in ("created", "set", "runs", "items", "score", "run_scores", "spread", "per_item")}
        Path(args.baseline).write_text(json.dumps(keep, indent=1) + "\n")
        print(f"Baseline written: score {results['score']:.2f}, spread {results['spread']:.2f}. Commit it in a PR.")
        return 0

    baseline = json.loads(Path(args.baseline).read_text())
    gate = tomllib.loads((HERE / "gate.toml").read_text())
    rules = decide(results, baseline, gate)
    text = report(results, baseline, rules)
    print(text)
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(text)
    return 0 if all(passed for _, passed, _ in rules) else 1


if __name__ == "__main__":
    sys.exit(main())
