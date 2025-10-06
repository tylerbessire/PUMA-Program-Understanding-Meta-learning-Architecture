#!/usr/bin/env python3
"""Evaluate the solver on a subset of ARC tasks with ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from arc_solver.solver import ARCSolver


DATASETS = {
    "training": (
        Path("data/arc-agi_training_challenges.json"),
        Path("data/arc-agi_training_solutions.json"),
    ),
    "evaluation": (
        Path("data/arc-agi_evaluation_challenges.json"),
        Path("data/arc-agi_evaluation_solutions.json"),
    ),
}


def load_dataset(name: str) -> tuple[Dict[str, Dict], Dict[str, List[List[List[int]]]]]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name}; choose from {list(DATASETS.keys())}")
    challenge_path, solution_path = DATASETS[name]
    if not challenge_path.exists() or not solution_path.exists():
        raise FileNotFoundError(f"Dataset files missing for {name}")
    challenges = json.loads(challenge_path.read_text())
    solutions = json.loads(solution_path.read_text())
    return challenges, solutions


def _coerce_solution_set(solution_entry):
    if not solution_entry:
        return []
    first = solution_entry[0]
    if isinstance(first, list) and first and isinstance(first[0], int):
        return [np.asarray(solution_entry)]
    return [np.asarray(candidate) for candidate in solution_entry]


def compare_attempts(attempts: Dict[str, List[List[List[int]]]], solutions: List) -> bool:
    attempt1 = attempts.get("attempt_1", [])
    attempt2 = attempts.get("attempt_2", [])
    if len(attempt1) != len(solutions):
        return False
    for idx, solution_set in enumerate(solutions):
        truth_arrays = _coerce_solution_set(solution_set)
        pred1 = np.asarray(attempt1[idx])
        pred2 = np.asarray(attempt2[idx]) if idx < len(attempt2) else None
        if any(np.array_equal(pred1, gt) for gt in truth_arrays):
            continue
        if pred2 is not None and any(np.array_equal(pred2, gt) for gt in truth_arrays):
            continue
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="training")
    parser.add_argument("--count", type=int, default=10, help="number of tasks to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="offset into the dataset")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    challenges, solutions = load_dataset(args.dataset)
    task_items = list(challenges.items())
    subset = task_items[args.offset : args.offset + args.count]

    solver = ARCSolver(use_enhancements=True)
    total = 0
    correct = 0
    failures: List[str] = []

    for task_id, task in subset:
        total += 1
        task_with_id = dict(task)
        task_with_id["task_id"] = task_id
        attempts = solver.solve_task(task_with_id)
        ok = compare_attempts(attempts, solutions[task_id])
        if ok:
            correct += 1
            status = "✓"
        else:
            failures.append(task_id)
            status = "✗"
        if args.verbose:
            print(f"{status} {task_id}")

    accuracy = correct / total if total else 0.0
    print(f"Evaluated {total} tasks from {args.dataset}")
    print(f"Accuracy: {correct}/{total} ({accuracy*100:.1f}%)")
    if failures:
        print("Failures:")
        for task_id in failures:
            print(f"  - {task_id}")


if __name__ == "__main__":
    main()
