"""
Object-Agentic solver implementation.

This module implements the main agentic solver that coordinates multiple
object-focused agents using a beam search with blackboard architecture.
"""

from typing import List, Tuple, Dict, Any, Optional, Set
import numpy as np
import heapq
from dataclasses import dataclass
from collections import defaultdict

from ..grid import Array, to_array, eq
from ..common.objects import connected_components
from ..common.invariants import (
    palette_equiv, object_count_invariant, evaluate_invariants
)
from ..common.mdl import program_length_from_ops, Operation
from ..common.eval_utils import exact_match
from .ops import Op, execute_program_on_grid, propose_ops_for_object, propose_global_ops, COST


@dataclass
class SearchState:
    """Represents a state in the beam search."""
    program: List[Op]
    score: float
    train_successes: int
    mdl_cost: float
    
    def __lt__(self, other):
        # Higher score is better, but heapq is a min-heap
        return self.score > other.score


class AgentBlackboard:
    """
    Blackboard for coordinating agents and managing program construction.
    """
    
    def __init__(self):
        self.active_programs: List[List[Op]] = []
        self.constraints: Dict[str, Any] = {}
        self.object_assignments: Dict[int, List[Op]] = defaultdict(list)
    
    def add_constraint(self, name: str, value: Any):
        """Add a constraint that programs must satisfy."""
        self.constraints[name] = value
    
    def check_constraints(self, program: List[Op], test_grid: Array, 
                         expected_grid: Array) -> bool:
        """Check if a program satisfies all constraints."""
        result = execute_program_on_grid(program, test_grid)
        if result is None:
            return False
        
        # Check invariants
        invariants = evaluate_invariants(test_grid, result)
        
        # Must preserve shape unless explicitly allowed
        if not self.constraints.get('allow_shape_change', False):
            if not invariants['shape_preserved']:
                return False
        
        # Check object count constraints
        if not invariants['object_count_stable']:
            if not self.constraints.get('allow_object_count_change', False):
                return False
        
        # Check palette constraints
        if self.constraints.get('preserve_palette', True):
            if not invariants['palette_permutation']:
                return False
        
        return True
    
    def score_program(self, program: List[Op], train_pairs: List[Tuple[Array, Array]]) -> Tuple[float, int]:
        """
        Score a program based on training performance and MDL.
        Returns (score, num_successes).
        """
        if not program:
            return (0.0, 0)
        
        successes = 0
        total_pairs = len(train_pairs)
        
        for input_grid, expected_grid in train_pairs:
            result = execute_program_on_grid(program, input_grid)
            if result is not None and exact_match(result, expected_grid):
                successes += 1
        
        accuracy = successes / max(1, total_pairs)
        
        # Compute MDL cost
        ops_dict = [{'kind': op.kind, 'params': op.params} for op in program]
        mdl_cost = program_length_from_ops(ops_dict)
        
        # Score combines accuracy and simplicity
        # Perfect accuracy gets base score of 100, then subtract MDL cost
        score = accuracy * 100.0 - mdl_cost
        
        return (score, successes)


def beam_search_agentic(train_pairs: List[Tuple[Array, Array]], 
                       beam_width: int = 64, max_depth: int = 4) -> Tuple[List[Op], float]:
    """
    Perform beam search to find the best program for the training pairs.
    """
    if not train_pairs:
        return ([], 0.0)
    
    # Initialize blackboard
    blackboard = AgentBlackboard()
    
    # Analyze training data to set constraints
    first_input, first_output = train_pairs[0]
    
    # Check if shape changes
    shape_changes = any(inp.shape != out.shape for inp, out in train_pairs)
    blackboard.add_constraint('allow_shape_change', shape_changes)
    
    # Check if object count changes
    obj_count_changes = any(
        not object_count_invariant(inp, out) for inp, out in train_pairs
    )
    blackboard.add_constraint('allow_object_count_change', obj_count_changes)
    
    # Check if palette changes
    palette_changes = any(
        not palette_equiv(inp, out) for inp, out in train_pairs
    )
    blackboard.add_constraint('preserve_palette', not palette_changes)
    
    # Initialize beam with empty program
    beam = [SearchState([], 0.0, 0, 0.0)]
    
    best_program = []
    best_score = -float('inf')
    
    for depth in range(max_depth):
        next_beam = []
        
        for state in beam:
            # Generate successor states
            successors = generate_successors(state, train_pairs, blackboard)
            next_beam.extend(successors)
        
        # Keep top beam_width states
        next_beam.sort(reverse=True, key=lambda s: s.score)
        beam = next_beam[:beam_width]
        
        # Update best program
        for state in beam:
            if state.score > best_score and state.train_successes > 0:
                best_score = state.score
                best_program = state.program.copy()
        
        # Early stopping if we find a perfect solution
        if any(state.train_successes == len(train_pairs) for state in beam):
            break
    
    return (best_program, best_score)


def generate_successors(state: SearchState, train_pairs: List[Tuple[Array, Array]],
                       blackboard: AgentBlackboard) -> List[SearchState]:
    """Generate successor states by adding one more operation."""
    successors = []
    
    if not train_pairs:
        return successors
    
    # Analyze first training example to propose operations
    input_grid, expected_grid = train_pairs[0]
    objects = connected_components(input_grid)
    
    # Propose operations for each object
    all_proposals = []
    
    # Object-specific operations
    for obj_idx, obj in enumerate(objects):
        context = {
            'obj_idx': obj_idx,
            'grid': input_grid,
            'all_objects': objects,
            'common_colors': list(range(10))  # Standard ARC colors
        }
        proposals = propose_ops_for_object(obj, context)
        all_proposals.extend(proposals)
    
    # Global operations
    global_proposals = propose_global_ops(input_grid, objects)
    all_proposals.extend(global_proposals)
    
    # Limit total proposals to prevent explosion
    all_proposals = all_proposals[:100]
    
    for op in all_proposals:
        new_program = state.program + [op]
        
        # Quick constraint check
        test_result = execute_program_on_grid(new_program, input_grid)
        if test_result is None:
            continue
        
        if not blackboard.check_constraints(new_program, input_grid, expected_grid):
            continue
        
        # Score the new program
        score, successes = blackboard.score_program(new_program, train_pairs)
        
        # Compute MDL cost
        ops_dict = [{'kind': op.kind, 'params': op.params} for op in new_program]
        mdl_cost = program_length_from_ops(ops_dict)
        
        new_state = SearchState(new_program, score, successes, mdl_cost)
        successors.append(new_state)
    
    return successors


def solve_task_agentic(task: Dict[str, Any], beam_width: int = 64, 
                      max_depth: int = 4) -> List[Array]:
    """
    Solve an ARC task using the agentic approach.
    
    Args:
        task: ARC task dictionary with 'train' and 'test' keys
        beam_width: Width of the beam search
        max_depth: Maximum depth of the search
    
    Returns:
        List of predicted grids for test cases
    """
    # Extract training pairs
    train_pairs = []
    for pair in task.get("train", []):
        try:
            input_grid = to_array(pair["input"])
            output_grid = to_array(pair["output"])
            train_pairs.append((input_grid, output_grid))
        except Exception:
            continue
    
    if not train_pairs:
        # No valid training data, return identity for test cases
        return [to_array(test_case["input"]) for test_case in task.get("test", [])]
    
    # Find best program using beam search
    best_program, best_score = beam_search_agentic(
        train_pairs, beam_width=beam_width, max_depth=max_depth
    )
    
    # Apply program to test cases
    predictions = []
    for test_case in task.get("test", []):
        try:
            test_input = to_array(test_case["input"])
            prediction = execute_program_on_grid(best_program, test_input)
            
            if prediction is not None:
                predictions.append(prediction)
            else:
                # Fallback to identity
                predictions.append(test_input)
        except Exception:
            # Error in processing, use identity
            predictions.append(to_array([[0]]))
    
    return predictions


def solve_task_agentic_dict(task: Dict[str, Any], beam_width: int = 64, 
                           max_depth: int = 4) -> Dict[str, List[List[List[int]]]]:
    """
    Solve an ARC task and return in the standard dictionary format.
    This matches the interface expected by the solver registry.
    """
    predictions = solve_task_agentic(task, beam_width, max_depth)
    
    # Convert to list format
    pred_lists = []
    for pred in predictions:
        if isinstance(pred, np.ndarray):
            pred_lists.append(pred.astype(int).tolist())
        else:
            pred_lists.append([[0]])  # Fallback
    
    return {
        "attempt_1": pred_lists,
        "attempt_2": pred_lists  # For now, return same predictions
    }