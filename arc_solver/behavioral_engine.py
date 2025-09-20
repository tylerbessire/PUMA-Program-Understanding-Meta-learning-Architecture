"""Reinforcement-oriented training loop for the ARC solver.

This module implements the behavioural control loop outlined in the
functional contextualist roadmap.  It provides a production-grade
training orchestrator that presents ARC tasks as antecedents, executes
behaviours (program synthesis attempts), and propagates consequences as
reinforcement updates to neural guidance and episodic memory modules.

The engine is intentionally deterministic and side-effect free unless
explicitly enabled via the ``PUMA_BEHAVIORAL_ENGINE`` feature flag to
guarantee safe rollouts inside evaluation pipelines.

[S:DESIGN v1] approach=behavioural_engine+reward_grader alt={offline_supervised,policy_gradient_rl} reason=online-reinforcement pass
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dsl import apply_program
from .enhanced_search import EnhancedSearch
from .grid import Array
from .neural.guidance import NeuralGuidance
from .neural.episodic import EpisodicRetrieval
from .rft_engine.engine import RFTEngine, RFTInference
from .utils.metrics import MovingAverage


TrainPair = Tuple[Array, Array]
Program = List[Tuple[str, Dict[str, Any]]]


class FeatureToggle:
    """Environment-backed feature toggle with safe default off."""

    def __init__(self, env_var: str = "PUMA_BEHAVIORAL_ENGINE") -> None:
        self.env_var = env_var

    @property
    def enabled(self) -> bool:
        value = os.environ.get(self.env_var, "0").strip().lower()
        return value in {"1", "true", "on", "yes"}


@dataclass
class RewardBreakdown:
    """Detailed reinforcement signal produced by :class:`RewardGrader`."""

    pixel_accuracy: float
    shape_accuracy: float
    behaviour_bonus: float
    program_length_penalty: float
    reward: float


class RewardGrader:
    """Compute scalar rewards with interpretable sub-metrics."""

    def __init__(
        self,
        pixel_weight: float = 0.7,
        shape_weight: float = 0.2,
        behaviour_weight: float = 0.1,
        length_penalty: float = 0.02,
    ) -> None:
        if not 0.0 <= pixel_weight <= 1.0:
            raise ValueError("pixel_weight must be within [0, 1]")
        if not 0.0 <= shape_weight <= 1.0:
            raise ValueError("shape_weight must be within [0, 1]")
        if not 0.0 <= behaviour_weight <= 1.0:
            raise ValueError("behaviour_weight must be within [0, 1]")
        self.pixel_weight = pixel_weight
        self.shape_weight = shape_weight
        self.behaviour_weight = behaviour_weight
        self.length_penalty = length_penalty

    def grade(
        self,
        predictions: Sequence[Array],
        targets: Sequence[Array],
        program: Program,
        behavioural_signal: float,
    ) -> RewardBreakdown:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets length mismatch")

        pixel_scores: List[float] = []
        shape_scores: List[float] = []
        for pred, tgt in zip(predictions, targets):
            if pred.shape != tgt.shape:
                shape_scores.append(0.0)
            else:
                shape_scores.append(1.0)
            total = tgt.size
            if total == 0:
                pixel_scores.append(1.0)
                continue
            matches = float(np.sum(pred == tgt))
            pixel_scores.append(matches / float(total))

        pixel_accuracy = float(np.mean(pixel_scores)) if pixel_scores else 0.0
        shape_accuracy = float(np.mean(shape_scores)) if shape_scores else 0.0
        behaviour_bonus = max(0.0, min(1.0, behavioural_signal))
        penalty = self.length_penalty * max(0, len(program) - 1)
        reward = (
            pixel_accuracy * self.pixel_weight
            + shape_accuracy * self.shape_weight
            + behaviour_bonus * self.behaviour_weight
        )
        reward = max(0.0, min(1.0, reward - penalty))
        return RewardBreakdown(
            pixel_accuracy=pixel_accuracy,
            shape_accuracy=shape_accuracy,
            behaviour_bonus=behaviour_bonus,
            program_length_penalty=penalty,
            reward=reward,
        )


@dataclass
class BehaviouralMetrics:
    """Aggregated telemetry for monitoring training runs."""

    tasks_trained: int = 0
    successful_tasks: int = 0
    cumulative_reward: float = 0.0
    pixel_moving_avg: MovingAverage = field(default_factory=lambda: MovingAverage(window=50))

    def update(self, breakdown: RewardBreakdown, solved: bool) -> None:
        self.tasks_trained += 1
        if solved:
            self.successful_tasks += 1
        self.cumulative_reward += breakdown.reward
        self.pixel_moving_avg.add_sample(breakdown.pixel_accuracy)

    def as_dict(self) -> Dict[str, float]:
        return {
            "tasks_trained": float(self.tasks_trained),
            "successful_tasks": float(self.successful_tasks),
            "success_rate": (
                float(self.successful_tasks) / float(self.tasks_trained)
                if self.tasks_trained
                else 0.0
            ),
            "cumulative_reward": self.cumulative_reward,
            "pixel_accuracy_ma": self.pixel_moving_avg.value,
        }


class BehavioralEngine:
    """Top-level operant conditioning loop for the solver."""

    def __init__(
        self,
        dataset_loader: Optional[Callable[[Path], Dict[str, Dict[str, List]]]] = None,
        solutions_loader: Optional[Callable[[Path], Dict[str, List[List[List[int]]]]]] = None,
        search_factory: Callable[[], EnhancedSearch] = EnhancedSearch,
        reward_grader: Optional[RewardGrader] = None,
        feature_toggle: Optional[FeatureToggle] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._toggle = feature_toggle or FeatureToggle()
        self._dataset_loader = dataset_loader or load_challenges
        self._solutions_loader = solutions_loader or load_solutions
        self._search_factory = search_factory
        self._reward_grader = reward_grader or RewardGrader()
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        self.metrics = BehaviouralMetrics()
        self._rng = random.Random(0)

    def train(
        self,
        dataset_path: Path,
        solutions_path: Path,
        max_tasks: Optional[int] = None,
        seed: int = 0,
        shuffle: bool = True,
    ) -> BehaviouralMetrics:
        """Execute the reinforcement training loop."""

        if not self._toggle.enabled:
            raise RuntimeError(
                "Behavioural engine disabled – set PUMA_BEHAVIORAL_ENGINE=1 to train"
            )

        dataset_path = dataset_path.resolve(strict=True)
        solutions_path = solutions_path.resolve(strict=True)

        tasks = self._dataset_loader(dataset_path)
        solutions = self._solutions_loader(solutions_path)
        task_ids = list(tasks.keys())
        self._rng.seed(seed)
        if shuffle:
            self._rng.shuffle(task_ids)
        if max_tasks is not None:
            task_ids = task_ids[:max_tasks]

        search = self._search_factory()
        guidance: NeuralGuidance = search.neural_guidance
        episodic: EpisodicRetrieval = search.episodic_retrieval
        rft_engine = RFTEngine()

        for task_id in task_ids:
            task = tasks[task_id]
            try:
                train_pairs = self._convert_pairs(task["train"])
            except KeyError as exc:
                raise ValueError(f"task {task_id} missing train pairs") from exc
            if not train_pairs:
                continue
            gt_outputs = self._ground_truth_outputs(task_id, task, solutions)
            inference = rft_engine.analyse(train_pairs)
            best_program, best_breakdown = self._evaluate_programs(search, train_pairs, gt_outputs, inference)
            if best_program is None or best_breakdown is None:
                continue
            solved = best_breakdown.reward > 0.999
            guidance.reinforce(train_pairs, best_program, best_breakdown.reward, inference)
            episodic.add_successful_solution(
                train_pairs,
                [best_program],
                task_id=task_id,
                reward=best_breakdown.reward,
                metadata={
                    "pixel_accuracy": best_breakdown.pixel_accuracy,
                    "shape_accuracy": best_breakdown.shape_accuracy,
                    "behaviour_bonus": best_breakdown.behaviour_bonus,
                },
            )
            self.metrics.update(best_breakdown, solved)
            self._emit_metrics(task_id, best_program, best_breakdown)

        episodic.save()
        return self.metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _convert_pairs(self, raw_pairs: Iterable[Dict[str, List[List[int]]]]) -> List[TrainPair]:
        pairs: List[TrainPair] = []
        for pair in raw_pairs:
            inp = np.array(pair["input"], dtype=int)
            out = np.array(pair["output"], dtype=int)
            pairs.append((inp, out))
        return pairs

    def _ground_truth_outputs(
        self,
        task_id: str,
        task: Dict[str, List],
        solutions: Dict[str, List[List[List[int]]]],
    ) -> List[Array]:
        outputs: List[Array] = []
        if task_id in solutions:
            for grid in solutions[task_id]:
                outputs.append(np.array(grid, dtype=int))
        else:
            for pair in task.get("train", []):
                outputs.append(np.array(pair["output"], dtype=int))
        return outputs

    def _evaluate_programs(
        self,
        search: EnhancedSearch,
        train_pairs: List[TrainPair],
        outputs: List[Array],
        inference: RFTInference,
        max_candidates: int = 16,
    ) -> Tuple[Optional[Program], Optional[RewardBreakdown]]:
        programs = search.synthesize_enhanced(train_pairs, max_programs=max_candidates)
        best_program: Optional[Program] = None
        best_breakdown: Optional[RewardBreakdown] = None
        for program in programs:
            predictions: List[Array] = []
            try:
                for inp, _ in train_pairs:
                    predictions.append(apply_program(inp, program))
            except Exception:
                continue
            behavioural_signal = inference.estimate_behavioural_signal(program)
            breakdown = self._reward_grader.grade(predictions, outputs[: len(predictions)], program, behavioural_signal)
            if best_breakdown is None or breakdown.reward > best_breakdown.reward:
                best_breakdown = breakdown
                best_program = program
        return best_program, best_breakdown

    def _emit_metrics(
        self,
        task_id: str,
        program: Program,
        breakdown: RewardBreakdown,
    ) -> None:
        payload = {
            "task_id": task_id,
            "reward": breakdown.reward,
            "pixel_accuracy": breakdown.pixel_accuracy,
            "shape_accuracy": breakdown.shape_accuracy,
            "behaviour_bonus": breakdown.behaviour_bonus,
            "program_length": len(program),
            "global": self.metrics.as_dict(),
        }
        self._logger.info(json.dumps(payload, sort_keys=True))


# [S:API v1] route=BehavioralEngine.train feature_flag=PUMA_BEHAVIORAL_ENGINE metrics=structured_json pass


def load_challenges(path: Path) -> Dict[str, Dict[str, List]]:
    """Load ARC challenge file into a dictionary keyed by task id."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("challenge file must contain a mapping of tasks")
    return {str(task_id): task for task_id, task in data.items()}


def load_solutions(path: Path) -> Dict[str, List[List[List[int]]]]:
    """Load ARC solutions JSON and normalise keys to strings."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("solutions file must contain a mapping of tasks")
    return {str(task_id): value for task_id, value in data.items()}
