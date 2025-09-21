#!/usr/bin/env python3
"""Validate ARC training data integrity and provide quick statistics.

This utility checks that the ARC training challenges and solutions are
consistent (matching task IDs, equal test-case counts, non-empty grids) and
emits aggregate stats that we can monitor for data drift.  It is intended to be
fast enough to run in CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


TRAIN_CHALLENGES = Path("data/arc-agi_training_challenges.json")
TRAIN_SOLUTIONS = Path("data/arc-agi_training_solutions.json")


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Training file missing: {path}")
    return json.loads(path.read_text())


def _grid_shape(grid: List[List[int]]) -> Tuple[int, int]:
    arr = np.asarray(grid, dtype=np.int16)
    if arr.ndim == 1:
        return (1, int(arr.shape[0]))
    if arr.ndim != 2:
        raise ValueError(f"Grid must be 2-D, got shape {arr.shape}")
    return tuple(int(x) for x in arr.shape)


def main() -> None:
    challenges = _load_json(TRAIN_CHALLENGES)
    solutions = _load_json(TRAIN_SOLUTIONS)

    challenge_ids = set(challenges.keys())
    solution_ids = set(solutions.keys())

    missing_in_solutions = sorted(challenge_ids - solution_ids)
    missing_in_challenges = sorted(solution_ids - challenge_ids)

    assert not missing_in_solutions, (
        "Tasks missing solutions: " + ", ".join(missing_in_solutions[:10])
    )
    assert not missing_in_challenges, (
        "Solutions lacking tasks: " + ", ".join(missing_in_challenges[:10])
    )

    num_tasks = len(challenges)
    print(f"Loaded {num_tasks} training tasks")

    input_shapes = Counter()
    output_shapes = Counter()
    expansion_counts = Counter()
    color_diversity = Counter()

    for task_id, task in challenges.items():
        train_pairs = task.get("train", [])
        test_cases = task.get("test", [])
        solution_cases = solutions[task_id]

        assert len(solution_cases) == len(test_cases), (
            f"Task {task_id} has {len(test_cases)} test cases but "
            f"{len(solution_cases)} solution sets"
        )

        for pair in train_pairs:
            input_shape = _grid_shape(pair["input"])
            output_shape = _grid_shape(pair["output"])
            input_shapes[input_shape] += 1
            output_shapes[output_shape] += 1
            if input_shape != output_shape:
                expansion_counts[(input_shape, output_shape)] += 1

            colors = tuple(sorted(np.unique(pair["output"])))
            color_diversity[len(colors)] += 1

        for test_idx, solutions_for_case in enumerate(solution_cases):
            assert solutions_for_case, f"Task {task_id} test {test_idx} has no solutions"
            for candidate in solutions_for_case:
                _ = _grid_shape(candidate)

    most_common_inputs = input_shapes.most_common(5)
    most_common_outputs = output_shapes.most_common(5)

    print("Top input shapes:")
    for (shape, count) in most_common_inputs:
        print(f"  {shape}: {count}")

    print("Top output shapes:")
    for (shape, count) in most_common_outputs:
        print(f"  {shape}: {count}")

    if expansion_counts:
        print("Most frequent expansion patterns:")
        for ((in_shape, out_shape), count) in expansion_counts.most_common(5):
            print(f"  {in_shape} -> {out_shape}: {count}")

    print("Output color diversity (unique colors per grid):")
    for diversity, count in sorted(color_diversity.items()):
        print(f"  {diversity}: {count}")

    print("Training data validation complete ✅")


if __name__ == "__main__":
    main()
