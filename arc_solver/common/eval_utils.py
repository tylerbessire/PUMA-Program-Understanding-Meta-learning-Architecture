"""
Evaluation utilities for scoring ARC solutions.

This module provides utilities for evaluating the quality of solutions,
computing metrics, and logging results for analysis.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
import json
from pathlib import Path
import numpy as np

from ..grid import Array, eq


def exact_match(predicted: Array, expected: Array) -> bool:
    """Check if two grids match exactly."""
    return eq(predicted, expected)


def partial_match_score(predicted: Array, expected: Array) -> float:
    """
    Compute partial match score (pixel-wise accuracy).
    Returns value between 0.0 and 1.0.
    """
    if predicted.shape != expected.shape:
        return 0.0
    
    if predicted.size == 0:
        return 1.0 if expected.size == 0 else 0.0
    
    matching_pixels = np.sum(predicted == expected)
    total_pixels = predicted.size
    return matching_pixels / total_pixels


def shape_match_score(predicted: Array, expected: Array) -> float:
    """
    Score based on shape similarity.
    1.0 if shapes match exactly, decreases with shape difference.
    """
    pred_h, pred_w = predicted.shape
    exp_h, exp_w = expected.shape
    
    if pred_h == exp_h and pred_w == exp_w:
        return 1.0
    
    # Penalize based on relative size difference
    size_diff = abs((pred_h * pred_w) - (exp_h * exp_w))
    max_size = max(pred_h * pred_w, exp_h * exp_w, 1)
    return max(0.0, 1.0 - (size_diff / max_size))


def color_palette_score(predicted: Array, expected: Array) -> float:
    """
    Score based on color palette similarity.
    1.0 if same colors used, decreases with palette differences.
    """
    pred_colors = set(predicted.flatten())
    exp_colors = set(expected.flatten())
    
    if pred_colors == exp_colors:
        return 1.0
    
    intersection = len(pred_colors & exp_colors)
    union = len(pred_colors | exp_colors)
    
    return intersection / union if union > 0 else 0.0


def comprehensive_score(predicted: Array, expected: Array, 
                       weights: Dict[str, float] = None) -> Dict[str, float]:
    """
    Compute comprehensive scoring metrics.
    
    Returns:
        Dictionary with various scoring metrics
    """
    if weights is None:
        weights = {
            'exact': 1.0,
            'partial': 0.5,
            'shape': 0.3,
            'palette': 0.2
        }
    
    scores = {
        'exact': 1.0 if exact_match(predicted, expected) else 0.0,
        'partial': partial_match_score(predicted, expected),
        'shape': shape_match_score(predicted, expected),
        'palette': color_palette_score(predicted, expected)
    }
    
    # Compute weighted overall score
    total_weight = sum(weights.values())
    if total_weight > 0:
        scores['overall'] = sum(weights.get(metric, 0) * score 
                               for metric, score in scores.items()) / total_weight
    else:
        scores['overall'] = 0.0
    
    return scores


class TaskLogger:
    """Logger for tracking task solving progress and results."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.results: List[Dict[str, Any]] = []
        self.start_time = None
    
    def start_task(self, task_id: str, solver_name: str):
        """Start logging a new task."""
        self.start_time = time.time()
        self.current_log = {
            'task_id': task_id,
            'solver': solver_name,
            'start_time': self.start_time,
            'train_results': [],
            'test_results': [],
            'metadata': {}
        }
    
    def log_train_result(self, pair_idx: int, predicted: Array, expected: Array, 
                        program_info: Dict[str, Any] = None):
        """Log result for a training pair."""
        if not hasattr(self, 'current_log'):
            return
        
        scores = comprehensive_score(predicted, expected)
        result = {
            'pair_idx': pair_idx,
            'exact_match': scores['exact'],
            'scores': scores,
            'predicted_shape': predicted.shape,
            'expected_shape': expected.shape
        }
        
        if program_info:
            result['program_info'] = program_info
        
        self.current_log['train_results'].append(result)
    
    def log_test_result(self, test_idx: int, predicted: Array, 
                       expected: Array = None, program_info: Dict[str, Any] = None):
        """Log result for a test case."""
        if not hasattr(self, 'current_log'):
            return
        
        result = {
            'test_idx': test_idx,
            'predicted_shape': predicted.shape,
        }
        
        if expected is not None:
            scores = comprehensive_score(predicted, expected)
            result.update({
                'exact_match': scores['exact'],
                'scores': scores,
                'expected_shape': expected.shape
            })
        
        if program_info:
            result['program_info'] = program_info
        
        self.current_log['test_results'].append(result)
    
    def finish_task(self, success: bool = None, error: str = None, 
                   final_program: Any = None):
        """Finish logging the current task."""
        if not hasattr(self, 'current_log'):
            return
        
        end_time = time.time()
        self.current_log.update({
            'end_time': end_time,
            'duration_ms': int((end_time - self.start_time) * 1000),
            'success': success,
            'error': error
        })
        
        if final_program:
            self.current_log['final_program'] = str(final_program)
        
        # Compute summary statistics
        train_exact = sum(1 for r in self.current_log['train_results'] 
                         if r.get('exact_match', False))
        total_train = len(self.current_log['train_results'])
        
        test_exact = sum(1 for r in self.current_log['test_results'] 
                        if r.get('exact_match', False))
        total_test = len(self.current_log['test_results'])
        
        self.current_log.update({
            'train_exact_matches': train_exact,
            'total_train_pairs': total_train,
            'train_accuracy': train_exact / max(1, total_train),
            'test_exact_matches': test_exact,
            'total_test_cases': total_test,
            'test_accuracy': test_exact / max(1, total_test) if total_test > 0 else None
        })
        
        self.results.append(self.current_log.copy())
        
        # Write to file if specified
        if self.log_file:
            self._write_to_file()
        
        delattr(self, 'current_log')
    
    def _write_to_file(self):
        """Write results to log file."""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not write to log file {self.log_file}: {e}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all logged tasks."""
        if not self.results:
            return {}
        
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.get('success', False))
        
        train_accuracies = [r['train_accuracy'] for r in self.results 
                           if 'train_accuracy' in r]
        test_accuracies = [r['test_accuracy'] for r in self.results 
                          if r.get('test_accuracy') is not None]
        
        durations = [r['duration_ms'] for r in self.results if 'duration_ms' in r]
        
        return {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks,
            'avg_train_accuracy': np.mean(train_accuracies) if train_accuracies else 0,
            'avg_test_accuracy': np.mean(test_accuracies) if test_accuracies else None,
            'avg_duration_ms': np.mean(durations) if durations else 0,
            'total_duration_ms': sum(durations) if durations else 0
        }


def evaluate_solver_on_dataset(solver_func, dataset: List[Dict[str, Any]], 
                              solver_name: str = "unknown",
                              log_file: str = None) -> Dict[str, Any]:
    """
    Evaluate a solver function on a dataset and return comprehensive results.
    
    Args:
        solver_func: Function that takes a task dict and returns predictions
        dataset: List of task dictionaries  
        solver_name: Name of the solver for logging
        log_file: Optional path to save detailed logs
    
    Returns:
        Dictionary with evaluation results and statistics
    """
    logger = TaskLogger(log_file)
    results = []
    
    for i, task in enumerate(dataset):
        task_id = task.get('id', f'task_{i}')
        logger.start_task(task_id, solver_name)
        
        try:
            # Get predictions from solver
            predictions = solver_func(task)
            
            # Evaluate on test cases if solutions available
            test_results = []
            if 'test' in task and isinstance(predictions, list):
                for j, (test_case, pred) in enumerate(zip(task['test'], predictions)):
                    if 'output' in test_case:  # Has ground truth
                        expected = np.array(test_case['output'])
                        pred_array = np.array(pred) if not isinstance(pred, np.ndarray) else pred
                        
                        exact = exact_match(pred_array, expected)
                        test_results.append(exact)
                        logger.log_test_result(j, pred_array, expected)
                    else:
                        logger.log_test_result(j, np.array(pred))
            
            success = len(test_results) > 0 and all(test_results)
            logger.finish_task(success=success)
            
            results.append({
                'task_id': task_id,
                'success': success,
                'test_accuracy': np.mean(test_results) if test_results else None
            })
            
        except Exception as e:
            logger.finish_task(success=False, error=str(e))
            results.append({
                'task_id': task_id,
                'success': False,
                'error': str(e)
            })
    
    # Compute overall statistics
    summary = logger.get_summary_stats()
    summary.update({
        'solver_name': solver_name,
        'dataset_size': len(dataset),
        'detailed_results': results
    })
    
    return summary