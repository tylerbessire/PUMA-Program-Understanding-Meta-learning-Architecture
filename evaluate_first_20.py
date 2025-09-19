#!/usr/bin/env python3
"""
Comprehensive evaluation of first 20 ARC tasks with real-time GUI and detailed logging.
"""

import json
import numpy as np
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import os
import sys
from pathlib import Path

# Set up detailed logging
log_dir = Path("evaluation_logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"arc_eval_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import solver
try:
    from arc_solver.solver import solve_task
    logger.info("Successfully imported ARC solver")
except ImportError as e:
    logger.error(f"Failed to import solver: {e}")
    sys.exit(1)

def to_grid(a):
    """Convert to standardized grid format."""
    a = np.asarray(a, dtype=np.uint8)
    if a.ndim == 1: 
        a = a[None, :]
    if a.ndim == 3 and a.shape[-1] == 1: 
        a = a[..., 0]
    assert a.ndim == 2
    return a

def grids_equal(pred_raw, gold_raw):
    """Check if two grids are exactly equal."""
    try:
        pred = to_grid(pred_raw)
        gold = to_grid(gold_raw)
        
        if pred.shape != gold.shape:
            return False, f"Shape mismatch: {pred.shape} != {gold.shape}", 0.0
        
        if np.array_equal(pred, gold):
            return True, "Exact match", 1.0
        
        # Calculate accuracy
        matches = np.sum(pred == gold)
        total = pred.size
        accuracy = matches / total
        
        # Find mismatches
        ys, xs = np.where(pred != gold)
        mismatch_count = len(ys)
        first_mismatches = [(int(ys[i]), int(xs[i])) for i in range(min(5, len(ys)))]
        
        return False, f"Pixel mismatch: {mismatch_count}/{total} wrong, accuracy={accuracy:.3f}, first_errors={first_mismatches}", accuracy
        
    except Exception as e:
        return False, f"Comparison error: {e}", 0.0

class RealTimeEvaluator:
    """Real-time ARC evaluation with GUI display."""
    
    def __init__(self):
        self.results = {}
        self.total_score = 0
        self.total_tasks = 0
        self.start_time = None
        
    def print_header(self):
        """Print evaluation header."""
        print("\n" + "="*80)
        print("🏆 ARC CHALLENGE EVALUATION - FIRST 20 TASKS")
        print("="*80)
        print("Task ID          | Status | Score | Accuracy | Time    | Details")
        print("-"*80)
        
    def print_task_result(self, task_id: str, status: str, score: int, accuracy: float, 
                         duration: float, details: str):
        """Print individual task result in real-time."""
        status_emoji = "✅" if status == "PASS" else "❌"
        accuracy_str = f"{accuracy*100:5.1f}%"
        time_str = f"{duration:6.2f}s"
        
        # Truncate details if too long
        if len(details) > 40:
            details = details[:37] + "..."
            
        print(f"{task_id:15} | {status_emoji} {status:4} | {score:5} | {accuracy_str:8} | {time_str:7} | {details}")
        
        # Update running totals
        self.total_score += score
        self.total_tasks += 1
        current_percentage = (self.total_score / self.total_tasks) * 100
        
        # Show running total
        if self.total_tasks % 5 == 0 or status == "PASS":
            print(f"{'':15} | 📊 Running total: {self.total_score}/{self.total_tasks} = {current_percentage:.1f}%")
            
    def print_summary(self):
        """Print final evaluation summary."""
        total_time = time.time() - self.start_time
        avg_time = total_time / max(1, self.total_tasks)
        final_percentage = (self.total_score / self.total_tasks) * 100
        
        print("\n" + "="*80)
        print("🎯 FINAL EVALUATION RESULTS")
        print("="*80)
        print(f"Total ARC Score:     {self.total_score}/{self.total_tasks}")
        print(f"Success Rate:        {final_percentage:.1f}%")
        print(f"Total Time:          {total_time:.1f}s")
        print(f"Average Time/Task:   {avg_time:.1f}s")
        print(f"Log File:            {log_file}")
        
        # Performance tier
        if final_percentage >= 50:
            tier = "🥇 GOLD (State-of-the-art)"
        elif final_percentage >= 25:
            tier = "🥈 SILVER (Strong performance)"
        elif final_percentage >= 10:
            tier = "🥉 BRONZE (Good baseline)"
        else:
            tier = "📊 BASELINE"
            
        print(f"Performance Tier:    {tier}")
        print("="*80)

def load_data():
    """Load challenge and solution data."""
    logger.info("Loading ARC evaluation data...")
    
    challenge_file = "data/arc-agi_evaluation_challenges.json"
    solution_file = "data/arc-agi_evaluation_solutions.json"
    
    if not os.path.exists(challenge_file):
        logger.error(f"Challenge file not found: {challenge_file}")
        sys.exit(1)
        
    if not os.path.exists(solution_file):
        logger.error(f"Solution file not found: {solution_file}")
        sys.exit(1)
    
    with open(challenge_file, 'r') as f:
        challenges = json.load(f)
        
    with open(solution_file, 'r') as f:
        solutions = json.load(f)
    
    logger.info(f"Loaded {len(challenges)} challenges and {len(solutions)} solutions")
    return challenges, solutions

def evaluate_task(task_id: str, task: Dict, gold_solution: List, evaluator: RealTimeEvaluator) -> Dict[str, Any]:
    """Evaluate a single task."""
    start_time = time.time()
    
    logger.info(f"Starting evaluation of task {task_id}")
    
    try:
        # Solve the task
        logger.info(f"Solving task {task_id}...")
        result = solve_task(task)
        
        if 'attempt_1' not in result:
            raise Exception("No attempt_1 in result")
            
        # Get prediction
        prediction = result['attempt_1'][0]
        
        # Compare with gold solution
        is_correct, details, accuracy = grids_equal(prediction, gold_solution)
        
        # Calculate results
        score = 1 if is_correct else 0
        status = "PASS" if is_correct else "FAIL"
        duration = time.time() - start_time
        
        # Log detailed results
        logger.info(f"Task {task_id}: {status} (score={score}, accuracy={accuracy:.3f}, time={duration:.2f}s)")
        logger.info(f"Task {task_id} details: {details}")
        
        # Print real-time result
        evaluator.print_task_result(task_id, status, score, accuracy, duration, details)
        
        return {
            'task_id': task_id,
            'status': status,
            'score': score,
            'accuracy': accuracy,
            'duration': duration,
            'details': details,
            'prediction_shape': to_grid(prediction).shape,
            'gold_shape': to_grid(gold_solution).shape
        }
        
    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Solver error: {str(e)[:50]}"
        
        logger.error(f"Task {task_id} failed: {e}")
        evaluator.print_task_result(task_id, "ERROR", 0, 0.0, duration, error_msg)
        
        return {
            'task_id': task_id,
            'status': 'ERROR',
            'score': 0,
            'accuracy': 0.0,
            'duration': duration,
            'details': str(e),
            'prediction_shape': None,
            'gold_shape': to_grid(gold_solution).shape if gold_solution else None
        }

def main():
    """Main evaluation function."""
    logger.info("Starting ARC evaluation of first 20 tasks")
    
    # Load data
    challenges, solutions = load_data()
    
    # Get first 20 task IDs
    task_ids = list(challenges.keys())[:20]
    logger.info(f"Evaluating first 20 tasks: {task_ids}")
    
    # Initialize evaluator
    evaluator = RealTimeEvaluator()
    evaluator.start_time = time.time()
    evaluator.print_header()
    
    # Store detailed results
    detailed_results = []
    
    # Evaluate each task
    for i, task_id in enumerate(task_ids, 1):
        logger.info(f"=== Task {i}/20: {task_id} ===")
        
        task = challenges[task_id]
        gold_solution = solutions[task_id][0]  # First test case solution
        
        result = evaluate_task(task_id, task, gold_solution, evaluator)
        detailed_results.append(result)
        
        # Small delay for readability
        time.sleep(0.1)
    
    # Print final summary
    evaluator.print_summary()
    
    # Save detailed results
    results_file = log_dir / f"detailed_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_score': evaluator.total_score,
            'total_tasks': evaluator.total_tasks,
            'success_rate': (evaluator.total_score / evaluator.total_tasks) * 100,
            'results': detailed_results
        }, f, indent=2)
    
    logger.info(f"Detailed results saved to: {results_file}")
    logger.info(f"Evaluation complete. Final score: {evaluator.total_score}/{evaluator.total_tasks}")
    
    return evaluator.total_score, evaluator.total_tasks

if __name__ == "__main__":
    try:
        score, total = main()
        sys.exit(0 if score > 0 else 1)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)