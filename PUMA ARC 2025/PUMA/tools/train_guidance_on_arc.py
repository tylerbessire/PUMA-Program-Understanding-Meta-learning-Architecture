"""Train guidance model on ARC challenge and solution datasets."""
# [S:TRAIN v1] dataset=train+eval pass

import argparse
import json
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import Array
from arc_solver.neural.guidance import NeuralGuidance


def _load_tasks(ch_path: str, sol_path: str) -> List[List[Tuple[Array, Array]]]:
    with open(ch_path, "r", encoding="utf-8") as f:
        challenges = json.load(f)
    with open(sol_path, "r", encoding="utf-8") as f:
        solutions = json.load(f)
    missing = set(challenges) - set(solutions)
    if missing:
        raise ValueError(f"solutions missing for tasks: {sorted(list(missing))[:5]}")

    tasks: List[List[Tuple[Array, Array]]] = []
    for task in challenges.values():
        train_pairs: List[Tuple[Array, Array]] = []
        for pair in task.get("train", []):
            inp = np.array(pair["input"], dtype=int)
            out = np.array(pair["output"], dtype=int)
            train_pairs.append((inp, out))
        tasks.append(train_pairs)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train guidance on ARC datasets")
    parser.add_argument("--train-challenges", default="data/arc-agi_training_challenges.json")
    parser.add_argument("--train-solutions", default="data/arc-agi_training_solutions.json")
    parser.add_argument("--eval-challenges", default="data/arc-agi_evaluation_challenges.json")
    parser.add_argument("--eval-solutions", default="data/arc-agi_evaluation_solutions.json")
    parser.add_argument("--out", default="models/guidance_arc.json", help="Output model path")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    tasks: List[List[Tuple[Array, Array]]] = []
    tasks.extend(_load_tasks(args.train_challenges, args.train_solutions))
    tasks.extend(_load_tasks(args.eval_challenges, args.eval_solutions))

    guidance = NeuralGuidance()
    guidance.train_from_task_pairs(tasks, epochs=args.epochs)
    guidance.save_model(args.out)
    print(f"model trained on {len(tasks)} tasks and saved to {args.out}")


if __name__ == "__main__":
    main()
