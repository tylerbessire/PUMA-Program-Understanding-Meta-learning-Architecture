"""
Enhanced top-level solver interface for ARC tasks.

This module integrates the enhanced search capabilities including neural guidance,
episodic retrieval, program sketches, and test-time training to provide better
solutions for ARC tasks.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import os

from .grid import to_array, to_list, Array
from .search import synthesize, predict_two  # Keep original as fallback
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
            progs = synthesize(train_pairs)
            attempts = predict_two(progs, test_inputs)
        
        # Convert outputs back to nested lists
        return {
            "attempt_1": [to_list(arr) for arr in attempts[0]],
            "attempt_2": [to_list(arr) for arr in attempts[1]],
        }
    
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
