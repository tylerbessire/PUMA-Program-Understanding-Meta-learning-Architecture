#!/usr/bin/env python3
"""
Comprehensive evaluation of PUMA solver performance on ARC evaluation set.

This script evaluates both enhanced and baseline solvers against the official 
ARC evaluation challenges and provides detailed accuracy metrics.
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from arc_solver.solver import ARCSolver, solve_task_enhanced, solve_task_baseline
from arc_solver.grid import to_array


def load_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load evaluation challenges and solutions."""
    with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('data/arc-agi_evaluation_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    return challenges, solutions


def arrays_equal(arr1: List[List[int]], arr2: List[List[int]]) -> bool:
    """Check if two grid arrays are equal."""
    if len(arr1) != len(arr2):
        return False
    
    for row1, row2 in zip(arr1, arr2):
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False
    
    return True


def evaluate_task(task_id: str, task: Dict[str, Any], solutions: List[List[List[int]]], 
                  solver_func) -> Tuple[bool, bool, float]:
    """
    Evaluate a single task.
    
    Returns:
        (attempt1_correct, attempt2_correct, solve_time)
    """
    start_time = time.time()
    
    try:
        result = solver_func(task)
        solve_time = time.time() - start_time
        
        attempt1 = result.get("attempt_1", [])
        attempt2 = result.get("attempt_2", [])
        
        # Check if either attempt matches any of the expected solutions
        attempt1_correct = False
        attempt2_correct = False
        
        for i, expected in enumerate(solutions):
            if i < len(attempt1) and arrays_equal(attempt1[i], expected):
                attempt1_correct = True
            if i < len(attempt2) and arrays_equal(attempt2[i], expected):
                attempt2_correct = True
        
        return attempt1_correct, attempt2_correct, solve_time
        
    except Exception as e:
        print(f"Error evaluating task {task_id}: {str(e)}")
        return False, False, time.time() - start_time


def run_evaluation(max_tasks: int = None) -> Dict[str, Any]:
    """
    Run comprehensive evaluation on ARC evaluation set.
    
    Args:
        max_tasks: Maximum number of tasks to evaluate (None for all)
    
    Returns:
        Dictionary with detailed results
    """
    print("Loading ARC evaluation data...")
    challenges, solutions = load_data()
    
    # Limit tasks if specified
    task_ids = list(challenges.keys())
    if max_tasks:
        task_ids = task_ids[:max_tasks]
    
    print(f"Evaluating {len(task_ids)} tasks...")
    
    # Initialize solvers
    enhanced_solver = ARCSolver(use_enhancements=True)
    baseline_solver = ARCSolver(use_enhancements=False)
    
    results = {
        'enhanced': {
            'attempt1_correct': 0,
            'attempt2_correct': 0,
            'task_solved': 0,  # Either attempt correct
            'total_tasks': len(task_ids),
            'solve_times': [],
            'task_results': {}
        },
        'baseline': {
            'attempt1_correct': 0,
            'attempt2_correct': 0,
            'task_solved': 0,
            'total_tasks': len(task_ids),
            'solve_times': [],
            'task_results': {}
        }
    }
    
    for i, task_id in enumerate(task_ids):
        if i % 10 == 0:
            print(f"Progress: {i+1}/{len(task_ids)} tasks processed")
        
        task = challenges[task_id]
        expected_solutions = solutions[task_id]
        
        # Evaluate enhanced solver
        try:
            enh_a1, enh_a2, enh_time = evaluate_task(
                task_id, task, expected_solutions, enhanced_solver.solve_task
            )
            results['enhanced']['attempt1_correct'] += int(enh_a1)
            results['enhanced']['attempt2_correct'] += int(enh_a2)
            results['enhanced']['task_solved'] += int(enh_a1 or enh_a2)
            results['enhanced']['solve_times'].append(enh_time)
            results['enhanced']['task_results'][task_id] = {
                'attempt1_correct': enh_a1,
                'attempt2_correct': enh_a2,
                'solve_time': enh_time
            }
        except Exception as e:
            print(f"Enhanced solver failed on task {task_id}: {str(e)}")
            results['enhanced']['solve_times'].append(0.0)
            results['enhanced']['task_results'][task_id] = {
                'attempt1_correct': False,
                'attempt2_correct': False,
                'solve_time': 0.0,
                'error': str(e)
            }
        
        # Evaluate baseline solver
        try:
            base_a1, base_a2, base_time = evaluate_task(
                task_id, task, expected_solutions, baseline_solver.solve_task
            )
            results['baseline']['attempt1_correct'] += int(base_a1)
            results['baseline']['attempt2_correct'] += int(base_a2)
            results['baseline']['task_solved'] += int(base_a1 or base_a2)
            results['baseline']['solve_times'].append(base_time)
            results['baseline']['task_results'][task_id] = {
                'attempt1_correct': base_a1,
                'attempt2_correct': base_a2,
                'solve_time': base_time
            }
        except Exception as e:
            print(f"Baseline solver failed on task {task_id}: {str(e)}")
            results['baseline']['solve_times'].append(0.0)
            results['baseline']['task_results'][task_id] = {
                'attempt1_correct': False,
                'attempt2_correct': False,
                'solve_time': 0.0,
                'error': str(e)
            }
    
    # Calculate final metrics
    for solver_type in ['enhanced', 'baseline']:
        r = results[solver_type]
        r['attempt1_accuracy'] = r['attempt1_correct'] / r['total_tasks']
        r['attempt2_accuracy'] = r['attempt2_correct'] / r['total_tasks'] 
        r['task_success_rate'] = r['task_solved'] / r['total_tasks']
        r['avg_solve_time'] = np.mean(r['solve_times']) if r['solve_times'] else 0.0
        r['total_solve_time'] = sum(r['solve_times'])
    
    return results


def print_results(results: Dict[str, Any]):
    """Print comprehensive evaluation results."""
    print("\n" + "="*80)
    print("PUMA ARC SOLVER EVALUATION RESULTS")
    print("="*80)
    
    for solver_type in ['enhanced', 'baseline']:
        r = results[solver_type]
        print(f"\n{solver_type.upper()} SOLVER:")
        print(f"  Tasks Evaluated: {r['total_tasks']}")
        print(f"  Task Success Rate: {r['task_success_rate']:.1%} ({r['task_solved']}/{r['total_tasks']})")
        print(f"  Attempt 1 Accuracy: {r['attempt1_accuracy']:.1%} ({r['attempt1_correct']}/{r['total_tasks']})")
        print(f"  Attempt 2 Accuracy: {r['attempt2_accuracy']:.1%} ({r['attempt2_correct']}/{r['total_tasks']})")
        print(f"  Average Solve Time: {r['avg_solve_time']:.3f}s")
        print(f"  Total Solve Time: {r['total_solve_time']:.1f}s")
    
    # Enhancement comparison
    enh = results['enhanced']
    base = results['baseline']
    improvement = enh['task_success_rate'] - base['task_success_rate']
    
    print(f"\nENHANCEMENT COMPARISON:")
    print(f"  Success Rate Improvement: {improvement:+.1%}")
    print(f"  Enhanced vs Baseline: {enh['task_success_rate']:.1%} vs {base['task_success_rate']:.1%}")
    
    if improvement > 0:
        print(f"  🎉 Enhanced solver performs {improvement/.01:.1f} percentage points better!")
    elif improvement < 0:
        print(f"  ⚠️  Enhanced solver performs {abs(improvement)/.01:.1f} percentage points worse")
    else:
        print(f"  🤔 Enhanced and baseline solvers perform equally")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function."""
    print("Starting PUMA ARC Solver Evaluation...")
    print("This will test both enhanced and baseline solvers on the ARC evaluation set.")
    
    # Run evaluation on first 50 tasks for quick testing
    # Change to None to run on all tasks (400+ tasks, takes much longer)
    max_tasks = 50
    
    if max_tasks:
        print(f"\nRunning quick evaluation on first {max_tasks} tasks...")
    else:
        print(f"\nRunning full evaluation on all tasks...")
    
    start_time = time.time()
    results = run_evaluation(max_tasks=max_tasks)
    total_time = time.time() - start_time
    
    print_results(results)
    print(f"\nTotal evaluation time: {total_time:.1f}s")
    
    # Save detailed results
    results_file = 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()
