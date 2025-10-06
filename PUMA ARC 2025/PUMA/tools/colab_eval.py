"""Train and evaluate the ARC solver in Kaggle/Colab environments.

This script provides a minimal end-to-end pipeline for training the neural
guidance classifier and producing Kaggle-compatible submission files. When
ground-truth solutions are provided, it also reports accuracy and per-task
differences between predictions and targets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure repository root is on the path so arc_solver can be imported when this
# script runs in Kaggle/Colab notebooks.
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.solver import ARCSolver
from arc_solver.grid import to_array, eq
from arc_solver.io_utils import save_submission
from train_guidance import (
    load_training_data,
    extract_training_features_and_labels,
    train_classifier,
    save_classifier,
)


def train_guidance_model(
    train_json: str,
    solutions_json: Optional[str],
    model_path: str,
    epochs: int = 100,
) -> str:
    """Train the neural guidance classifier.

    Args:
        train_json: Path to the ARC training challenges JSON.
        solutions_json: Optional path to training solutions for supervised labels.
        model_path: Where to persist the trained classifier.
        epochs: Number of training epochs.

    Returns:
        Path to the saved model.
    """
    tasks = load_training_data(train_json, solutions_json)
    features, labels = extract_training_features_and_labels(tasks)
    classifier = train_classifier(features, labels, epochs)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    save_classifier(classifier, model_path)
    return model_path


def evaluate_solver(
    test_json: str,
    model_path: str,
    solutions_json: Optional[str],
    out_path: str,
) -> Tuple[float, Dict[str, List[List[List[int]]]]]:
    """Run the solver on evaluation tasks and optionally score against solutions.

    Args:
        test_json: Path to evaluation challenges JSON.
        model_path: Path to trained guidance model.
        solutions_json: Optional path to ground-truth solutions for scoring.
        out_path: Where to write the Kaggle submission JSON.

    Returns:
        Tuple of overall accuracy (0-1) and a mapping of task ids to diff grids.
    """
    solver = ARCSolver(use_enhancements=True, guidance_model_path=model_path)

    with open(test_json, "r") as f:
        test_tasks: Dict[str, Any] = json.load(f)

    solutions: Dict[str, Any] = {}
    if solutions_json and Path(solutions_json).exists():
        with open(solutions_json, "r") as f:
            solutions = json.load(f)

    predictions: Dict[str, Dict[str, List[List[List[int]]]]] = {}
    diffs: Dict[str, List[List[List[int]]]] = {}
    correct = 0
    total = 0

    for task_id, task in test_tasks.items():
        result = solver.solve_task(task)
        predictions[task_id] = result

        if task_id in solutions:
            target_grids = [pair["output"] for pair in solutions[task_id]["test"]]
            pred_grids = result["attempt_1"]
            diff_grids: List[List[List[int]]] = []
            all_match = True

            for pred, target in zip(pred_grids, target_grids):
                pa = to_array(pred)
                ta = to_array(target)
                all_match &= eq(pa, ta)
                diff_grids.append((pa != ta).astype(int).tolist())

            if all_match:
                correct += 1
            diffs[task_id] = diff_grids
            total += 1

    save_submission(predictions, out_path)
    accuracy = correct / total if total else 0.0
    return accuracy, diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate ARC solver")
    parser.add_argument("--train-json", help="Path to training challenges JSON")
    parser.add_argument(
        "--train-solutions", help="Path to training solutions JSON", default=None
    )
    parser.add_argument(
        "--model-path",
        default="neural_guidance_model.json",
        help="Where to save or load the guidance model",
    )
    parser.add_argument("--test-json", required=True, help="Path to evaluation JSON")
    parser.add_argument(
        "--test-solutions",
        help="Path to evaluation solutions JSON for scoring",
        default=None,
    )
    parser.add_argument(
        "--out", default="submission.json", help="Output path for submission JSON"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")

    args = parser.parse_args()

    if args.train_json:
        train_guidance_model(
            args.train_json, args.train_solutions, args.model_path, args.epochs
        )

    accuracy, diffs = evaluate_solver(
        args.test_json, args.model_path, args.test_solutions, args.out
    )

    if args.test_solutions:
        print(f"Accuracy: {accuracy * 100:.2f}%")
        for task_id, diff in diffs.items():
            if any(np.any(np.array(d)) for d in diff):
                status = "incorrect"
            else:
                status = "correct"
            print(f"Task {task_id}: {status}")

    print(f"Submission file written to {args.out}")


if __name__ == "__main__":
    main()

