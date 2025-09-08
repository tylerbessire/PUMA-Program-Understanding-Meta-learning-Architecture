"""
Top-level solver interface for ARC tasks.

This module ties together the grid utilities, program synthesis search, and
heuristics to produce solutions for ARC tasks. Given a task dictionary with
training and test input/output pairs (in the ARC JSON format), it returns
predicted outputs for the test inputs in the required format.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .grid import to_array, to_list, Array
from .search import synthesize, predict_two
from .enhanced_solver import solve_task as solve_task_enhanced
import os


def solve_task(task: Dict[str, List[Dict[str, List[List[int]]]]]) -> Dict[str, List[List[List[int]]]]:
    """Solve a single ARC task.

    The task dictionary contains 'train' and 'test' lists. Each train item has
    'input' and 'output' grids. Each test item has 'input' and optionally
    'output' (if known). This function synthesizes a program using the train
    pairs and applies the best two programs to the test inputs.
    
    By default, uses enhanced solver with neural guidance, episodic retrieval,
    and test-time training. Set ARC_USE_BASELINE=1 to use baseline only.
    """
    # Check if enhanced solving is disabled
    use_baseline = os.environ.get('ARC_USE_BASELINE', '').lower() in ('1', 'true', 'yes')
    
    if not use_baseline:
        try:
            return solve_task_enhanced(task)
        except Exception:
            # Fall back to baseline if enhanced fails
            pass
    
    # Baseline implementation
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
    # Synthesize candidate programs and predict outputs
    progs = synthesize(train_pairs)
    attempts = predict_two(progs, test_inputs)
    # Convert outputs back to nested lists
    return {
        "attempt_1": [to_list(arr) for arr in attempts[0]],
        "attempt_2": [to_list(arr) for arr in attempts[1]],
    }