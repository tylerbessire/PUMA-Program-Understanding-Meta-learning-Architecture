"""
Minimum Description Length (MDL) utilities for program scoring.

This module provides utilities to compute the description length of 
programs and transformations, which is used to prefer simpler solutions
in accordance with Occam's razor.
"""

from typing import List, Dict, Any, Union
from dataclasses import dataclass


# Base costs for different operation types
BASE_OPERATION_COSTS = {
    "identity": 0,
    "recolor": 1,
    "translate": 1,
    "reflect": 1,
    "rotate": 1,
    "align": 2,
    "duplicate": 3,
    "palette_permute": 1,
    "resize": 2,
    "crop": 2,
    "flood_fill": 2,
    "connect": 3,
    "pattern": 4,
    "conditional": 3,
}


@dataclass
class Operation:
    """Represents a single operation with parameters."""
    kind: str
    params: tuple
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def operation_cost(op: Operation) -> float:
    """
    Compute the MDL cost of a single operation.
    Cost includes base operation cost plus parameter complexity.
    """
    base_cost = BASE_OPERATION_COSTS.get(op.kind, 2.0)
    
    # Add parameter complexity
    param_cost = 0.0
    if op.params:
        for param in op.params:
            if isinstance(param, int):
                # Cost increases with magnitude
                param_cost += max(1, abs(param).bit_length()) * 0.1
            elif isinstance(param, str):
                # String parameters have length-based cost
                param_cost += len(param) * 0.1
            elif isinstance(param, (list, tuple)):
                # Collection parameters have size-based cost
                param_cost += len(param) * 0.2
            else:
                # Generic parameter cost
                param_cost += 0.5
    
    return base_cost + param_cost


def program_length(operations: List[Operation]) -> float:
    """
    Compute the total MDL cost of a program (sequence of operations).
    """
    if not operations:
        return 0.0
    
    total_cost = sum(operation_cost(op) for op in operations)
    
    # Add small penalty for program length to prefer shorter programs
    length_penalty = len(operations) * 0.1
    
    return total_cost + length_penalty


def program_length_from_ops(ops: List[Dict[str, Any]]) -> float:
    """
    Compute program length from a list of operation dictionaries.
    Compatible with legacy operation representations.
    """
    operations = []
    for op_dict in ops:
        kind = op_dict.get('kind', op_dict.get('type', 'unknown'))
        params = op_dict.get('params', ())
        if not isinstance(params, tuple):
            params = tuple(params) if hasattr(params, '__iter__') else (params,)
        
        operations.append(Operation(kind=kind, params=params, metadata=op_dict))
    
    return program_length(operations)


def complexity_bonus(operations: List[Operation]) -> float:
    """
    Compute a complexity bonus for programs that demonstrate sophistication.
    This can help break ties between simple programs.
    """
    if not operations:
        return 0.0
    
    bonus = 0.0
    
    # Bonus for diverse operations
    unique_ops = len(set(op.kind for op in operations))
    if unique_ops > 2:
        bonus += (unique_ops - 2) * 0.1
    
    # Bonus for parameter diversity
    param_types = set()
    for op in operations:
        if op.params:
            for param in op.params:
                param_types.add(type(param).__name__)
    
    if len(param_types) > 2:
        bonus += (len(param_types) - 2) * 0.05
    
    return min(bonus, 1.0)  # Cap bonus at 1.0


def mdl_score(operations: List[Operation], accuracy: float = 1.0) -> float:
    """
    Compute MDL-based score combining program length and accuracy.
    
    Args:
        operations: List of operations in the program
        accuracy: Accuracy of the program on training data (0.0 to 1.0)
    
    Returns:
        Score where lower is better. Perfect accuracy and minimal length = 0.
    """
    if accuracy <= 0:
        return float('inf')  # Invalid programs get infinite cost
    
    length_cost = program_length(operations)
    accuracy_cost = -np.log(accuracy) if accuracy < 1.0 else 0.0
    
    # Combine costs (accuracy is more important than brevity)
    return accuracy_cost * 10.0 + length_cost


def compare_programs(prog1: List[Operation], prog2: List[Operation], 
                    acc1: float = 1.0, acc2: float = 1.0) -> int:
    """
    Compare two programs using MDL principle.
    
    Returns:
        -1 if prog1 is better, 1 if prog2 is better, 0 if equivalent
    """
    score1 = mdl_score(prog1, acc1)
    score2 = mdl_score(prog2, acc2)
    
    if abs(score1 - score2) < 1e-6:
        return 0
    return -1 if score1 < score2 else 1


def estimate_grid_complexity(grid) -> float:
    """
    Estimate the intrinsic complexity of a grid for MDL calculations.
    More complex grids might require more complex programs.
    """
    import numpy as np
    
    # Basic measures of grid complexity
    h, w = grid.shape
    num_colors = len(np.unique(grid))
    
    # Measure regularity (entropy-like measure)
    from collections import Counter
    hist = Counter(grid.flatten())
    total_pixels = h * w
    entropy = -sum((count / total_pixels) * np.log2(count / total_pixels) 
                  for count in hist.values())
    
    # Measure spatial regularity (simplified)
    spatial_changes = 0
    for y in range(h):
        for x in range(w - 1):
            if grid[y, x] != grid[y, x + 1]:
                spatial_changes += 1
    for y in range(h - 1):
        for x in range(w):
            if grid[y, x] != grid[y + 1, x]:
                spatial_changes += 1
    
    spatial_complexity = spatial_changes / (2 * h * w - h - w) if h * w > 1 else 0
    
    # Combine measures
    return (entropy + spatial_complexity + np.log2(num_colors)) / 3.0


# Make numpy available for mdl_score
import numpy as np