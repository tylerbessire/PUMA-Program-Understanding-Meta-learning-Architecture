"""Tests for behavioural reinforcement components."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from arc_solver.behavioral_engine import (
    BehavioralEngine,
    FeatureToggle,
    RewardGrader,
    load_challenges,
    load_solutions,
)
from arc_solver.behavioral_engine import RewardBreakdown
from arc_solver.neural.guidance import NeuralGuidance
from arc_solver.rft_engine.engine import RFTEngine, RFTInference


@st.composite
def grid_arrays(draw) -> np.ndarray:
    height = draw(st.integers(min_value=1, max_value=3))
    width = draw(st.integers(min_value=1, max_value=3))
    values = draw(
        st.lists(
            st.lists(st.integers(min_value=0, max_value=9), min_size=width, max_size=width),
            min_size=height,
            max_size=height,
        )
    )
    return np.array(values, dtype=int)


@st.composite
def prediction_target_pairs(draw) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    targets = draw(st.lists(grid_arrays(), min_size=1, max_size=3))
    predictions: List[np.ndarray] = []
    for target in targets:
        candidate = target.copy()
        if target.size and draw(st.booleans()):
            idx = draw(st.integers(min_value=0, max_value=target.size - 1))
            candidate = target.copy()
            candidate.flat[idx] = (candidate.flat[idx] + 1) % 10
        predictions.append(candidate)
    return predictions, targets


@given(data=prediction_target_pairs(), behavioural=st.floats(min_value=0, max_value=1))
@settings(max_examples=25, deadline=None)
def test_reward_grader_bounds(data: Tuple[List[np.ndarray], List[np.ndarray]], behavioural: float) -> None:
    predictions, targets = data
    grader = RewardGrader()
    breakdown = grader.grade(predictions, targets, program=[("identity", {})], behavioural_signal=behavioural)
    assert isinstance(breakdown, RewardBreakdown)
    assert 0.0 <= breakdown.reward <= 1.0
    assert 0.0 <= breakdown.pixel_accuracy <= 1.0
    assert 0.0 <= breakdown.shape_accuracy <= 1.0


def test_neural_guidance_reinforce_updates_stats() -> None:
    guidance = NeuralGuidance()
    train_pairs = [
        (
            np.array([[1, 0], [0, 0]], dtype=int),
            np.array([[0, 1], [0, 0]], dtype=int),
        )
    ]
    inference = RFTInference(relations=[], function_hints={"0:input:0": {"translate"}})
    guidance.reinforce(train_pairs, [("translate", {"dx": 1, "dy": 0})], reward=0.75, inference=inference)
    stats = guidance.operation_stats["translate"]
    assert stats["count"] >= 1.0
    assert 0.0 < stats["mean_reward"] <= 1.0
    scores = guidance.score_operations(train_pairs)
    assert scores["translate"] >= scores["identity"]


def test_rft_engine_detects_translation() -> None:
    arr_in = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=int)
    arr_out = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=int)
    inference = RFTEngine().analyse([(arr_in, arr_out)])
    all_hints = {hint for hints in inference.function_hints.values() for hint in hints}
    assert "translate" in all_hints


class _StubEpisodic:
    def __init__(self) -> None:
        self.calls: List[Dict[str, float]] = []
        self.saved = False

    def add_successful_solution(self, train_pairs, programs, task_id="", reward=None, metadata=None):
        self.calls.append({"reward": reward or 0.0, "task_id": task_id})

    def save(self) -> None:
        self.saved = True


class _StubSearch:
    def __init__(self) -> None:
        self.neural_guidance = NeuralGuidance()
        self.episodic_retrieval = _StubEpisodic()

    def synthesize_enhanced(self, train_pairs, max_programs=256, expected_shape=None, test_input=None):
        del train_pairs, max_programs, expected_shape, test_input
        return [[("translate", {"dx": 0, "dy": 1})]]


def test_behavioral_engine_training_cycle(tmp_path: Path) -> None:
    dataset = {
        "task_1": {
            "train": [
                {
                    "input": [[0, 1], [0, 0]],
                    "output": [[0, 0], [0, 1]],
                }
            ],
            "test": [],
        }
    }
    solutions = {"task_1": [[[0, 0], [0, 1]]]}
    dataset_path = tmp_path / "challenges.json"
    solutions_path = tmp_path / "solutions.json"
    dataset_path.write_text(json.dumps(dataset))
    solutions_path.write_text(json.dumps(solutions))

    env_var = "PUMA_BEHAVIORAL_ENGINE"
    old_value = os.environ.get(env_var)
    os.environ[env_var] = "1"
    try:
        stub_holder: Dict[str, _StubSearch] = {}

        def factory() -> _StubSearch:
            stub = _StubSearch()
            stub_holder["instance"] = stub
            return stub

        engine = BehavioralEngine(
            dataset_loader=load_challenges,
            solutions_loader=load_solutions,
            search_factory=factory,
            reward_grader=RewardGrader(pixel_weight=0.8, shape_weight=0.2, behaviour_weight=0.0, length_penalty=0.0),
            feature_toggle=FeatureToggle(env_var=env_var),
        )
        metrics = engine.train(dataset_path, solutions_path, max_tasks=1, shuffle=False)
    finally:
        if old_value is None:
            del os.environ[env_var]
        else:
            os.environ[env_var] = old_value

    stub = stub_holder["instance"]
    assert stub.episodic_retrieval.calls, "episodic memory should receive reward updates"
    translate_stats = stub.neural_guidance.operation_stats["translate"]
    assert translate_stats["count"] >= 1.0
    assert metrics.tasks_trained == 1
    assert metrics.cumulative_reward > 0.0
    assert stub.episodic_retrieval.saved is True
