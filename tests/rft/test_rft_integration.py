"""Integration tests for the RFT pliance + tracking module."""
# [S:TEST v2] module=rft_integration pass

import importlib
import pathlib
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from hypothesis import given, strategies as st

from puma.rft.demo.grid import build_context
from puma.rft.feature_flags import build_limits
from puma.rft.orchestrator import solve_with_rft, insert_after_stuck_rule
from puma.rft.tracking import HypothesisMemory, tracking_step, evaluate_hypothesis
from puma.rft.types import Context, Hypothesis, Rule
from puma.rft.explain import explain_last_run


def _build_memory() -> HypothesisMemory:
    return HypothesisMemory(cap=16)


def test_pliance_row_success() -> None:
    grid = [[0, 1, 1, 1]]
    targets = [(0, idx) for idx in range(len(grid[0]))]
    context, rules = build_context(grid, targets)
    memory = _build_memory()
    context, status = solve_with_rft(context, rules, memory, outer_budget=2)
    assert status == "DONE"
    assert context.state.grid[0] == [0, 0, 0, 0]
    assert context.metrics["pliance_rule_fires"] >= 1


def test_pliance_stuck_then_tracking_vertical() -> None:
    grid = [
        [0, 1],
        [2, 1],
        [2, 1],
    ]
    targets = [(r, 0) for r in range(len(grid))]
    context, rules = build_context(grid, targets)
    memory = _build_memory()
    context, status = solve_with_rft(context, rules, memory, outer_budget=4)
    assert status == "DONE"
    assert [row[0] for row in context.state.grid] == [0, 0, 0]
    assert context.metrics.get("tracking_hypotheses_evaluated", 0) > 0
    trace = explain_last_run(context)
    assert "Promoted rules" in trace


def test_reject_destructive_hypotheses() -> None:
    grid = [[2, 2]]
    targets = [(0, 0), (0, 1)]
    context, rules = build_context(grid, targets)
    memory = _build_memory()
    bad_hypothesis = Hypothesis(
        description="mismatched target color",
        variation_of="R1",
        params={
            "direction": (0, 1),
            "conditions": [{"target_color": 7}],
            "promote_hook": lambda h: rules[0],
        },
    )

    def fake_candidates(context: Context, base_rules, mem):
        return [bad_hypothesis]

    context.state.candidate_hypotheses = fake_candidates  # type: ignore[attr-defined]
    result = tracking_step(context, rules, memory, budget=1, thresh=0.9)
    assert result is None
    assert bad_hypothesis.signature() in memory.negatives


def test_hypothesis_signature_stable() -> None:
    params_a = {"direction": (0, 1), "conditions": [{"c": 1, "d": 2}, {"d": 3, "c": 4}]}
    params_b = {"direction": (0, 1), "conditions": [{"d": 3, "c": 4}, {"c": 1, "d": 2}]}
    hyp_a = Hypothesis("stable signature", "R1", params=params_a)
    hyp_b = Hypothesis("stable signature", "R1", params=params_b)
    assert hyp_a.signature() == hyp_b.signature()
    assert hyp_a.signature() == hyp_a.signature()


def test_hypothesis_memory_lru_eviction() -> None:
    memory = HypothesisMemory(cap=2)
    hyps = [
        Hypothesis(f"hyp-{idx}", "R1", params={"direction": (0, 1), "conditions": [], "name": f"h{idx}"})
        for idx in range(3)
    ]
    for hyp in hyps:
        memory.remember_negative(hyp)
    assert len(memory.negatives) == 2
    assert hyps[0].signature() not in memory.negatives


def test_tracking_score_prefers_progress_and_simplicity() -> None:
    class DummyState:
        def evaluate_hypothesis(self, hypothesis: Hypothesis, context: Context) -> Dict[str, Any]:
            return {"progress": hypothesis.params["progress"]}

    context = Context(
        state=DummyState(),
        history=[],
        constraints={},
        goal_test=lambda _state: False,
        limits=build_limits({"PLIANCE_STEPS": 4, "OUTER_BUDGET": 1, "TRACKING_BUDGET": 1}),
    )
    simple = Hypothesis("simple", "R1", params={"progress": 0.6, "conditions": []})
    complex_same = Hypothesis("complex", "R1", params={"progress": 0.6, "conditions": [{"foo": 1}]})
    advanced = Hypothesis("advanced", "R1", params={"progress": 0.8, "conditions": []})

    simple_score, _ = evaluate_hypothesis(simple, context)
    complex_score, _ = evaluate_hypothesis(complex_same, context)
    advanced_score, _ = evaluate_hypothesis(advanced, context)

    assert simple_score > complex_score
    assert advanced_score > simple_score


def test_insert_after_stuck_rule_positions() -> None:
    rule_a = Rule("R1", priority=1, preconditions=[], effects=[], applies_fn=lambda s, c: False, apply_fn=lambda s, c: s)
    rule_b = Rule("R2", priority=2, preconditions=[], effects=[], applies_fn=lambda s, c: False, apply_fn=lambda s, c: s)
    rules: List[Rule] = [rule_a, rule_b]
    new_rule = Rule("R_new", priority=3, preconditions=[], effects=[], applies_fn=lambda s, c: False, apply_fn=lambda s, c: s)
    anchored = insert_after_stuck_rule(rules, "R1", new_rule)
    assert [rule.name for rule in anchored] == ["R1", "R_new", "R2"]

    appended = insert_after_stuck_rule(rules, "missing", new_rule)
    assert appended[-1].name == "R_new"


def test_promotion_and_reuse() -> None:
    grid = [
        [0, 1],
        [2, 1],
    ]
    targets = [(0, 0), (1, 0)]
    context, rules = build_context(grid, targets)
    memory = _build_memory()
    context, status = solve_with_rft(context, rules, memory, outer_budget=4)
    assert status == "DONE"
    assert memory.positives

    # Reuse promoted rule on a fresh puzzle without re-search.
    fresh_grid = [
        [0, 1],
        [2, 1],
    ]
    fresh_context, fresh_rules = build_context(fresh_grid, targets)
    fresh_context, fresh_status = solve_with_rft(fresh_context, fresh_rules, memory, outer_budget=2)
    assert fresh_status == "DONE"
    assert fresh_context.metrics.get("tracking_hypotheses_evaluated", 0) == 0


def test_solver_parity_without_metadata(monkeypatch) -> None:
    task = {
        "train": [
            {"input": [[1, 0]], "output": [[1, 0]]},
        ],
        "test": [
            {"input": [[1, 0]]},
        ],
    }

    def reload_solver() -> None:
        if "puma.rft.feature_flags" in sys.modules:
            importlib.reload(sys.modules["puma.rft.feature_flags"])
        if "arc_solver.solver" in sys.modules:
            importlib.reload(sys.modules["arc_solver.solver"])

    monkeypatch.setenv("PUMA_RFT", "0")
    reload_solver()
    from arc_solver.solver import ARCSolver as SolverBaseline  # type: ignore

    baseline = SolverBaseline(use_enhancements=False).solve_task(task)

    monkeypatch.setenv("PUMA_RFT", "1")
    reload_solver()
    from arc_solver.solver import ARCSolver as SolverRFT  # type: ignore

    adaptive = SolverRFT(use_enhancements=False).solve_task(task)

    assert adaptive == baseline


def test_flag_off_does_not_change_baseline(monkeypatch) -> None:
    monkeypatch.setenv("PUMA_RFT", "0")
    if "arc_solver.solver" in list(sys.modules):
        importlib.reload(sys.modules["arc_solver.solver"])
    from arc_solver.solver import ARCSolver  # type: ignore

    solver = ARCSolver(use_enhancements=False)
    task = {
        "train": [
            {"input": [[1]], "output": [[2]]},
        ],
        "test": [
            {"input": [[1]]},
        ],
    }
    result = solver.solve_task(task)
    assert result["attempt_1"] == [[[2]]]
    assert result["attempt_2"] == [[[2]]]


@given(st.integers(min_value=2, max_value=6))
def test_property_horizontal_fill(width: int) -> None:
    grid = [[0] + [1] * (width - 1)]
    targets = [(0, idx) for idx in range(width)]
    context, rules = build_context(grid, targets)
    memory = _build_memory()
    context, status = solve_with_rft(context, rules, memory, outer_budget=2)
    assert status == "DONE"
    assert all(value == 0 for value in context.state.grid[0])
