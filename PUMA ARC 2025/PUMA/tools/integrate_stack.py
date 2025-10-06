"""Integrate neural guidance with beam search and report node reduction."""
# [S:INTEGRATION v1] beam_search+guidance pass

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from arc_solver.grid import Array
from arc_solver.neural.guidance import NeuralGuidance
from arc_solver.dsl import OPS
from arc_solver.heuristics import score_candidate
from arc_solver.neural.sketches import generate_parameter_grid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_task(challenges_path: str, task_id: str) -> List[Tuple[Array, Array]]:
    """Load a single task's training pairs."""
    with open(challenges_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task = data[task_id]
    pairs: List[Tuple[Array, Array]] = []
    for pair in task["train"]:
        inp = np.array(pair["input"], dtype=int)
        out = np.array(pair["output"], dtype=int)
        pairs.append((inp, out))
    return pairs


def evaluate_search_reduction(
    train_pairs: List[Tuple[Array, Array]], guidance: NeuralGuidance
) -> tuple[float, int, int]:
    """Compare node expansions with and without guidance."""

    def _count_expansions(order: List[str]) -> int:
        expansions = 0
        for op in order:
            for params in generate_parameter_grid(op):
                expansions += 1
                program = [(op, params)]
                try:
                    if score_candidate(program, train_pairs) >= 0.999:
                        return expansions
                except Exception:
                    continue
        return expansions

    baseline_order = list(OPS.keys())[2:] + list(OPS.keys())[:2]
    base_nodes = _count_expansions(baseline_order)

    op_scores = guidance.score_operations(train_pairs)
    guided_order = sorted(op_scores, key=op_scores.get, reverse=True)
    guided_nodes = _count_expansions(guided_order)

    reduction = 1.0 - guided_nodes / max(1, base_nodes)
    logger.info(
        "integrate_stack", extra={"baseline": base_nodes, "guided": guided_nodes, "reduction": reduction}
    )
    return reduction, base_nodes, guided_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate neural guidance integration")
    parser.add_argument("--challenges", default="data/arc-agi_training_challenges.json")
    parser.add_argument("--task-id", default="007bbfb7", help="Task ID to evaluate")
    parser.add_argument("--model", default="models/guidance_arc.json")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    pairs = load_task(args.challenges, args.task_id)
    guidance = NeuralGuidance()

    model_path = Path(args.model)
    if model_path.exists():
        guidance.load_model(str(model_path))
    else:
        with open(args.challenges, "r", encoding="utf-8") as f:
            challenges = json.load(f)
        all_tasks: List[List[Tuple[Array, Array]]] = []
        for task in challenges.values():
            t_pairs: List[Tuple[Array, Array]] = []
            for pair in task["train"]:
                inp = np.array(pair["input"], dtype=int)
                out = np.array(pair["output"], dtype=int)
                t_pairs.append((inp, out))
            all_tasks.append(t_pairs)
        guidance.train_from_task_pairs(all_tasks, epochs=args.epochs)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        guidance.save_model(str(model_path))

    reduction, base_nodes, guided_nodes = evaluate_search_reduction(pairs, guidance)
    print(
        f"baseline_nodes={base_nodes} guided_nodes={guided_nodes} reduction={reduction*100:.1f}%"
    )


if __name__ == "__main__":
    main()
