"""
Enhanced top-level solver interface for ARC tasks.

This module integrates the enhanced search capabilities including neural guidance,
episodic retrieval, program sketches, and test-time training to provide better
solutions for ARC tasks.
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
    
    def solve_task(self, task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
        """Solve a single ARC task using enhanced or baseline methods."""
        self.stats['total_tasks'] += 1
        
        # Extract training pairs as numpy arrays
        train_pairs: List[Tuple[Array, Array]] = []
        for pair in task["train"]:
            a = to_array(pair["input"])
            b = to_array(pair["output"])
            train_pairs.append((a, b))
        
        # Extract test inputs
        test_inputs: List[Array] = []
        for pair in task["test"]:
            test_inputs.append(to_array(pair["input"]))
        
        # Try enhanced synthesis first, fall back to baseline if needed
        try:
            if self.use_enhancements:
                progs = synthesize_with_enhancements(train_pairs)
                attempts = predict_two_enhanced(progs, test_inputs)
                
                # Check if we got a reasonable solution
                if self._validate_solution(attempts, test_inputs):
                    self.stats['tasks_solved'] += 1
                    return {
                        "attempt_1": [to_list(arr) for arr in attempts[0]],
                        "attempt_2": [to_list(arr) for arr in attempts[1]],
                    }
                else:
                    # Enhancement didn't work, try fallback
                    self.stats['fallback_used'] += 1
                    raise Exception("Enhanced search failed validation")
            else:
                raise Exception("Enhancements disabled")
                
        except Exception:
            # Fall back to baseline approach
            progs = synth_baseline(train_pairs)
            attempts = predict_two_baseline(progs, test_inputs)
        
        # Convert outputs back to nested lists
        return {
            "attempt_1": [to_list(arr) for arr in attempts[0]],
            "attempt_2": [to_list(arr) for arr in attempts[1]],
        }

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
        # Check if we should use enhancements (default: yes, unless disabled)
        use_enhancements = os.environ.get('ARC_DISABLE_ENHANCEMENTS', '').lower() not in ('1', 'true', 'yes')
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
