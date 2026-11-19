"""The eval gate's rules (Lesson 3). Pure logic, so it gets an ordinary unit test."""

from evals.gate import decide

GATE = {"floor": 0.75, "margin": 0.15}
BASELINE = {"score": 1.0}


def results(score, per_item=None):
    return {"score": score, "safety_items": ["inject-urea"], "per_item": per_item or {"inject-urea": 1.0}}


def passed(rules):
    return {name: ok for name, ok, _ in rules}


def test_same_as_baseline_passes():
    assert all(passed(decide(results(1.0), BASELINE, GATE)).values())


def test_a_drop_bigger_than_the_margin_fails_no_regression():
    rules = passed(decide(results(0.8), BASELINE, GATE))
    assert rules == {"floor": True, "no regression": False, "safety items pass every run": True}


def test_below_the_floor_fails_even_if_the_baseline_was_low():
    assert not passed(decide(results(0.6), {"score": 0.6}, GATE))["floor"]


def test_one_failed_safety_run_fails_the_gate_whatever_the_average():
    rules = passed(decide(results(0.95, {"inject-urea": 0.5}), BASELINE, GATE))
    assert not rules["safety items pass every run"]
