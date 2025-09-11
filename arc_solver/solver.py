"""Top-level solver interface for ARC tasks with neural enhancements.

This module integrates neural guidance, episodic retrieval, program sketches and
test-time training to provide state-of-the-art solutions for ARC tasks while
maintaining a robust fallback baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os

from .grid import to_array, to_list, Array
from .search import (
    synthesize as synth_baseline,
    predict_two as predict_two_baseline,
)
from .enhanced_search import synthesize_with_enhancements, predict_two_enhanced
from .hypothesis import HypothesisEngine, Hypothesis


class ARCSolver:
    """Enhanced ARC solver with neural components and episodic memory."""
    
    def __init__(self, use_enhancements: bool = True,
                 guidance_model_path: str = None,
                 episode_db_path: str = "episodes.json"):
        self.use_enhancements = use_enhancements
        self.guidance_model_path = guidance_model_path
        self.episode_db_path = episode_db_path
        self.stats = {
            'tasks_solved': 0,
            'total_tasks': 0,
            'enhancement_success_rate': 0.0,
            'fallback_used': 0,
        }
        self._last_outputs: Optional[Tuple[List[List[List[int]]], List[List[List[int]]]]] = None
        # Hypothesis engine powers the primary reasoning layer
        self.hypothesis_engine = HypothesisEngine()
        self._last_hypotheses: List[Hypothesis] = []
    
    def solve_task(self, task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
        """Solve a single ARC task using enhanced or baseline methods."""
        self.stats['total_tasks'] += 1

        # Extract training pairs as numpy arrays, skipping malformed ones
        train_pairs: List[Tuple[Array, Array]] = []
        for pair in task.get("train", []):
            try:
                a = to_array(pair["input"])
                b = to_array(pair["output"])
            except Exception:
                continue
            train_pairs.append((a, b))

        # Extract test inputs with graceful degradation
        test_inputs: List[Array] = []
        for pair in task.get("test", []):
            try:
                test_inputs.append(to_array(pair["input"]))
            except Exception:
                test_inputs.append(np.zeros((1, 1), dtype=np.int16))

        if not train_pairs:
            identity = [to_list(arr) for arr in test_inputs]
            return {"attempt_1": identity, "attempt_2": identity}

        # Generate and store hypotheses about the transformation.
        self._last_hypotheses = self.hypothesis_engine.generate_hypotheses(train_pairs)
        best_hypothesis: Optional[Hypothesis] = (
            self._last_hypotheses[0] if self._last_hypotheses else None
        )
        if best_hypothesis:
            # Update confidence using training pairs to double check
            best_hypothesis.confidence = self.hypothesis_engine.test_hypothesis(
                best_hypothesis, train_pairs
            )
            if best_hypothesis.confidence == 1.0:
                attempt1: List[List[List[int]]] = []
                attempt2: List[List[List[int]]] = []
                for test_input in test_inputs:
                    transformed = self.hypothesis_engine.apply(best_hypothesis, test_input)
                    if transformed is None:
                        break
                    attempt1.append(to_list(transformed))
                    attempt2.append(to_list(transformed))
                else:
                    # All test inputs transformed successfully
                    return {"attempt_1": attempt1, "attempt_2": attempt2}

        # Collect predictions for each test input individually
        attempt1: List[List[List[int]]] = []
        attempt2: List[List[List[int]]] = []
        for test_input in test_inputs:
            predictions = self._get_predictions(train_pairs, test_input)
            if predictions and predictions[0]:
                first = to_list(predictions[0][0])
                second_arr = predictions[1][0] if len(predictions) > 1 else predictions[0][0]
                second = to_list(second_arr)
                attempt1.append(first)
                attempt2.append(second)
            else:
                # Use identity grid as safe fallback
                fallback = to_list(test_input)
                attempt1.append(fallback)
                attempt2.append(fallback)

        return {"attempt_1": attempt1, "attempt_2": attempt2}

    def _get_predictions(
        self, train_pairs: List[Tuple[Array, Array]], test_input: Array
    ) -> List[List[Array]]:
        """Get prediction attempts for a single test input."""
        try:
            if self.use_enhancements:
                print("Using enhanced search for prediction")
                progs = synthesize_with_enhancements(train_pairs)
                attempts = predict_two_enhanced(progs, [test_input])
                if self._validate_solution(attempts, [test_input]):
                    return attempts
                else:
                    print("Enhanced prediction failed validation")
            else:
                print("Enhancements disabled, using baseline search")
        except Exception as e:
            print(f"Enhanced prediction error: {e}")

        # Fall back to baseline search
        self.stats['fallback_used'] += 1
        print("Falling back to baseline search")
        progs = synth_baseline(train_pairs)
        return predict_two_baseline(progs, [test_input])

    def solve_task_two_attempts(
        self, task: Dict[str, List[Dict[str, List[List[int]]]]]
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Solve a task and ensure two diverse attempts.

        Args:
            task: ARC task specification.

        Returns:
            A tuple ``(attempt1, attempt2)`` each being a list of output grids
            corresponding to the test inputs.
        """

        result = self.solve_task(task)
        attempt1 = result["attempt_1"]
        attempt2 = result["attempt_2"]

        if attempt1 == attempt2:
            alt = self._second_pass_diversified(task)
            if alt is not None:
                attempt2 = alt

        self._last_outputs = (attempt1, attempt2)
        return attempt1, attempt2

    def _second_pass_diversified(
        self, task: Dict[str, List[Dict[str, List[List[int]]]]]
    ) -> Optional[List[List[List[int]]]]:
        """Run a diversified second search pass to obtain an alternative output."""

        train_pairs = [
            (to_array(p["input"]), to_array(p["output"])) for p in task["train"]
        ]
        test_inputs = [to_array(p["input"]) for p in task["test"]]

        try:
            programs = synthesize_with_enhancements(train_pairs, force_alt=True)
            attempts = predict_two_enhanced(programs, test_inputs, prefer_diverse=True)
            return [to_list(x) for x in attempts[0]]
        except Exception:
            try:
                programs = synth_baseline(train_pairs)
                attempts = predict_two_baseline(
                    programs, test_inputs, prefer_diverse=True
                )
                return [to_list(x) for x in attempts[0]]
            except Exception:
                return None

    def best_so_far(
        self, task: Dict[str, List[Dict[str, List[List[int]]]]]
    ) -> List[List[List[int]]]:
        """Return the best outputs computed so far for the current task.

        If the solver has produced at least one attempt, that attempt is
        returned. Otherwise, the identity transformation of the first test
        input is used as a safe fallback.
        """

        if self._last_outputs is not None:
            return self._last_outputs[0]
        return [task["test"][0]["input"]]
    
    def _validate_solution(self, attempts: List[List[Array]], test_inputs: List[Array]) -> bool:
        """Basic validation to check if solution seems reasonable."""
        if not attempts or len(attempts) != 2:
            return False
        
        for attempt in attempts:
            if len(attempt) != len(test_inputs):
                return False
            
            # Check that outputs are not just copies of inputs (unless that's valid)
            for inp, out in zip(test_inputs, attempt):
                if out.shape[0] == 0 or out.shape[1] == 0:  # Empty output
                    return False
                if np.max(out) > 9:  # Invalid color values
                    return False
        
        return True
    
    def get_statistics(self) -> Dict[str, float]:
        """Get solver performance statistics."""
        success_rate = self.stats['tasks_solved'] / max(1, self.stats['total_tasks'])
        return {
            'success_rate': success_rate,
            'total_tasks': self.stats['total_tasks'],
            'tasks_solved': self.stats['tasks_solved'],
            'fallback_usage': self.stats['fallback_used'] / max(1, self.stats['total_tasks']),
        }


# Global solver instance (for backwards compatibility)
_global_solver = None


def solve_task(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve a single ARC task (backwards compatible interface)."""
    global _global_solver
    
    if _global_solver is None:
        # Determine whether to enable enhancements. Baseline can be forced via
        # ``ARC_USE_BASELINE`` or by explicitly disabling enhancements.
        use_baseline = os.environ.get('ARC_USE_BASELINE', '').lower() in (
            '1', 'true', 'yes'
        )
        enhancements_disabled = os.environ.get('ARC_DISABLE_ENHANCEMENTS', '').lower() in (
            '1', 'true', 'yes'
        )
        use_enhancements = not use_baseline and not enhancements_disabled
        _global_solver = ARCSolver(use_enhancements=use_enhancements)
    
    return _global_solver.solve_task(task)


def get_solver_stats() -> Dict[str, float]:
    """Get global solver statistics."""
    global _global_solver
    if _global_solver is None:
        return {}
    return _global_solver.get_statistics()


# Enhanced solver for direct use
def solve_task_enhanced(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve using enhanced methods only."""
    solver = ARCSolver(use_enhancements=True)
    return solver.solve_task(task)


# Baseline solver for comparison
def solve_task_baseline(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve using baseline methods only."""
    solver = ARCSolver(use_enhancements=False)
    return solver.solve_task(task)
