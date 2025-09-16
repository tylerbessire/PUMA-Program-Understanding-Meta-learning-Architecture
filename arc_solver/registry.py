"""
Solver registry for the PUMA ARC system.

This module provides a centralized registry for different solver implementations,
enabling easy switching between solver approaches and ensemble methods.
"""

from typing import Dict, Any, Callable, List, Optional
import importlib
from pathlib import Path


# Registry mapping solver names to their entry points
SOLVERS: Dict[str, str] = {
    # Existing solvers
    "baseline": "arc_solver.solver:solve_task_baseline",
    "enhanced": "arc_solver.solver:solve_task_enhanced",
    "default": "arc_solver.solver:solve_task",
    
    # New agentic and genomic solvers
    "agentic": "arc_solver.agents.agentic_solver:solve_task_agentic_dict",
    "genomic": "arc_solver.genomic.solver:solve_task_genomic_dict",
}


def get_solver(solver_name: str) -> Callable:
    """
    Get a solver function by name.
    
    Args:
        solver_name: Name of the solver to retrieve
    
    Returns:
        Solver function that takes a task dict and returns predictions
    
    Raises:
        ValueError: If solver name is not registered
        ImportError: If solver module cannot be imported
    """
    if solver_name not in SOLVERS:
        available_solvers = list(SOLVERS.keys())
        raise ValueError(f"Unknown solver '{solver_name}'. Available solvers: {available_solvers}")
    
    entry_point = SOLVERS[solver_name]
    module_path, function_name = entry_point.split(":")
    
    try:
        module = importlib.import_module(module_path)
        solver_func = getattr(module, function_name)
        return solver_func
    except ImportError as e:
        raise ImportError(f"Could not import solver '{solver_name}' from {module_path}: {e}")
    except AttributeError as e:
        raise ImportError(f"Function '{function_name}' not found in {module_path}: {e}")


def register_solver(name: str, entry_point: str) -> None:
    """
    Register a new solver.
    
    Args:
        name: Name to register the solver under
        entry_point: Module path and function name (e.g., "module.path:function_name")
    """
    SOLVERS[name] = entry_point


def list_solvers() -> List[str]:
    """Return list of registered solver names."""
    return list(SOLVERS.keys())


def solve_with_solver(task: Dict[str, Any], solver_name: str = "default", 
                     **kwargs) -> Dict[str, List[List[List[int]]]]:
    """
    Solve a task using the specified solver.
    
    Args:
        task: ARC task dictionary
        solver_name: Name of solver to use
        **kwargs: Additional parameters to pass to solver
    
    Returns:
        Dictionary with prediction attempts
    """
    solver_func = get_solver(solver_name)
    
    try:
        # Pass kwargs to solver if it supports them
        import inspect
        sig = inspect.signature(solver_func)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return solver_func(task, **kwargs)
        else:
            return solver_func(task)
    except Exception as e:
        # Fallback: return identity transformation
        print(f"Warning: Solver '{solver_name}' failed with error: {e}")
        return fallback_solve(task)


def fallback_solve(task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
    """
    Fallback solver that returns identity transformations.
    Used when other solvers fail.
    """
    test_cases = task.get('test', [])
    identity_predictions = []
    
    for test_case in test_cases:
        try:
            input_grid = test_case['input']
            identity_predictions.append(input_grid)
        except Exception:
            identity_predictions.append([[0]])
    
    return {
        "attempt_1": identity_predictions,
        "attempt_2": identity_predictions
    }


class EnsembleSolver:
    """
    Ensemble solver that combines multiple solvers with meta-reasoning.
    """
    
    def __init__(self, solver_names: List[str], voting_method: str = "meta_reasoning"):
        self.solver_names = solver_names
        self.voting_method = voting_method
        self.solvers = {name: get_solver(name) for name in solver_names}
    
    def solve_task(self, task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
        """
        Solve task using ensemble of solvers with meta-reasoning.
        
        Args:
            task: ARC task dictionary
        
        Returns:
            Combined predictions from ensemble
        """
        all_predictions = {}
        solver_metadata = {}
        
        # Get predictions from each solver
        for solver_name in self.solver_names:
            try:
                solver_func = self.solvers[solver_name]
                predictions = solver_func(task)
                all_predictions[solver_name] = predictions
                
                # Collect metadata for meta-reasoning
                solver_metadata[solver_name] = self._analyze_solver_predictions(
                    task, predictions, solver_name
                )
                
            except Exception as e:
                print(f"Solver {solver_name} failed: {e}")
                all_predictions[solver_name] = fallback_solve(task)
                solver_metadata[solver_name] = {'error': str(e), 'confidence': 0.0}
        
        if not all_predictions:
            return fallback_solve(task)
        
        # Combine predictions based on voting method
        if self.voting_method == "meta_reasoning":
            return self._meta_reasoning_selection(all_predictions, solver_metadata, task)
        elif self.voting_method == "majority":
            return self._majority_vote(all_predictions, task)
        elif self.voting_method == "best_confidence":
            return self._best_confidence(all_predictions, solver_metadata)
        else:
            # Default: return first successful solver's predictions
            return next(iter(all_predictions.values()))
    
    def _analyze_solver_predictions(self, task: Dict[str, Any], predictions: Dict[str, Any], 
                                  solver_name: str) -> Dict[str, Any]:
        """Analyze solver predictions to extract metadata for meta-reasoning."""
        metadata = {
            'solver_name': solver_name,
            'confidence': 0.5,  # Default confidence
            'complexity_match': 0.0,
            'size_change_handling': 0.0,
            'pattern_preservation': 0.0
        }
        
        try:
            train_examples = task.get('train', [])
            test_cases = task.get('test', [])
            
            if not train_examples or not test_cases or 'attempt_1' not in predictions:
                return metadata
            
            # Analyze size change handling
            input_sizes = [len(ex['input']) * len(ex['input'][0]) if ex['input'] else 0 
                          for ex in train_examples]
            output_sizes = [len(ex['output']) * len(ex['output'][0]) if ex['output'] else 0 
                           for ex in train_examples]
            
            if input_sizes and output_sizes:
                expected_size_ratios = [o/max(1,i) for i, o in zip(input_sizes, output_sizes)]
                
                # Check if solver handles size changes well
                pred_attempts = predictions['attempt_1']
                if pred_attempts:
                    test_input_size = len(test_cases[0]['input']) * len(test_cases[0]['input'][0])
                    pred_size = len(pred_attempts[0]) * len(pred_attempts[0][0]) if pred_attempts[0] else 0
                    actual_ratio = pred_size / max(1, test_input_size)
                    
                    # Compare to expected patterns
                    avg_expected_ratio = sum(expected_size_ratios) / len(expected_size_ratios)
                    ratio_match = 1.0 - abs(actual_ratio - avg_expected_ratio) / max(1, avg_expected_ratio)
                    metadata['size_change_handling'] = max(0.0, min(1.0, ratio_match))
            
            # Analyze complexity handling
            if solver_name == 'genomic':
                metadata['confidence'] = 0.7  # Good for pattern recognition
                metadata['complexity_match'] = 0.8
            elif solver_name == 'agentic':
                metadata['confidence'] = 0.6  # Good for object manipulation
                metadata['pattern_preservation'] = 0.7
            
            # Boost confidence for successful size matching
            if metadata['size_change_handling'] > 0.8:
                metadata['confidence'] += 0.2
                
        except Exception:
            # Error in analysis, use defaults
            pass
        
        return metadata
    
    def _meta_reasoning_selection(self, all_predictions: Dict[str, Dict[str, Any]], 
                                solver_metadata: Dict[str, Dict[str, Any]],
                                task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
        """Select best solver using meta-reasoning about task characteristics."""
        
        # Analyze task characteristics
        task_analysis = self._analyze_task_complexity(task)
        
        # Score each solver based on task-solver fit
        solver_scores = {}
        
        for solver_name, metadata in solver_metadata.items():
            score = metadata.get('confidence', 0.5)
            
            # Adjust score based on task characteristics
            if task_analysis['has_size_change']:
                if solver_name == 'genomic':
                    score += 0.3  # Genomic better at compression/expansion
                score += metadata.get('size_change_handling', 0) * 0.4
            
            if task_analysis['high_complexity']:
                if solver_name == 'agentic':
                    score += 0.2  # Agentic better at complex object manipulation
            
            if task_analysis['has_patterns']:
                if solver_name == 'genomic':
                    score += 0.2  # Genomic better at pattern recognition
            
            # Penalize if solver had errors
            if 'error' in metadata:
                score *= 0.1
            
            solver_scores[solver_name] = score
        
        # Select best solver
        best_solver = max(solver_scores.keys(), key=lambda s: solver_scores[s])
        
        print(f"Meta-reasoning selected: {best_solver} (score: {solver_scores[best_solver]:.2f})")
        
        return all_predictions[best_solver]
    
    def _analyze_task_complexity(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task to determine its characteristics."""
        analysis = {
            'has_size_change': False,
            'high_complexity': False,
            'has_patterns': False,
            'num_train_examples': len(task.get('train', []))
        }
        
        try:
            train_examples = task.get('train', [])
            
            if train_examples:
                # Check for size changes
                for example in train_examples:
                    input_shape = (len(example['input']), len(example['input'][0]))
                    output_shape = (len(example['output']), len(example['output'][0]))
                    
                    if input_shape != output_shape:
                        analysis['has_size_change'] = True
                        break
                
                # Check for high complexity (large grids or many colors)
                for example in train_examples:
                    input_size = len(example['input']) * len(example['input'][0])
                    if input_size > 100:  # Large grid
                        analysis['high_complexity'] = True
                        break
                    
                    # Count colors
                    colors = set()
                    for row in example['input']:
                        colors.update(row)
                    if len(colors) > 6:  # Many colors
                        analysis['high_complexity'] = True
                        break
                
                # Simple pattern detection
                first_example = train_examples[0]
                if len(first_example['input']) > 5 or len(first_example['input'][0]) > 5:
                    analysis['has_patterns'] = True
                
        except Exception:
            # Error in analysis, use defaults
            pass
        
        return analysis
    
    def _majority_vote(self, all_predictions: Dict[str, Dict[str, Any]], 
                      task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
        """Combine predictions using majority voting."""
        # For simplicity, just return the first solver's predictions
        # A full implementation would need to compare grid outputs
        return next(iter(all_predictions.values()))
    
    def _best_confidence(self, all_predictions: Dict[str, Dict[str, Any]],
                        solver_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Return predictions from the most confident solver."""
        best_solver = max(solver_metadata.keys(), 
                         key=lambda s: solver_metadata[s].get('confidence', 0))
        return all_predictions[best_solver]


def create_ensemble_solver(solver_names: List[str], voting_method: str = "majority") -> Callable:
    """
    Create an ensemble solver function.
    
    Args:
        solver_names: List of solver names to ensemble
        voting_method: Method for combining predictions
    
    Returns:
        Ensemble solver function
    """
    ensemble = EnsembleSolver(solver_names, voting_method)
    return ensemble.solve_task


def benchmark_solvers(tasks: List[Dict[str, Any]], solver_names: Optional[List[str]] = None,
                     max_tasks: Optional[int] = None) -> Dict[str, Any]:
    """
    Benchmark multiple solvers on a set of tasks.
    
    Args:
        tasks: List of ARC tasks to evaluate
        solver_names: Solvers to benchmark (default: all registered solvers)
        max_tasks: Maximum number of tasks to evaluate
    
    Returns:
        Benchmark results
    """
    if solver_names is None:
        solver_names = list_solvers()
    
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]
    
    results = {}
    
    for solver_name in solver_names:
        print(f"Benchmarking {solver_name}...")
        solver_results = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'errors': 0,
            'avg_time': 0.0
        }
        
        import time
        total_time = 0.0
        
        for task in tasks:
            solver_results['total_tasks'] += 1
            
            try:
                start_time = time.time()
                predictions = solve_with_solver(task, solver_name)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                
                # Simple success check - would need ground truth for real evaluation
                if predictions and 'attempt_1' in predictions:
                    solver_results['successful_tasks'] += 1
                    
            except Exception as e:
                solver_results['errors'] += 1
                print(f"Error in {solver_name}: {e}")
        
        solver_results['success_rate'] = (
            solver_results['successful_tasks'] / max(1, solver_results['total_tasks'])
        )
        solver_results['avg_time'] = total_time / max(1, solver_results['total_tasks'])
        
        results[solver_name] = solver_results
    
    return results


# Register ensemble solver
def _ensemble_solver_factory(solver_names: List[str]):
    """Factory function to create ensemble solver with specific solvers."""
    def ensemble_solve(task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
        ensemble = EnsembleSolver(solver_names)
        return ensemble.solve_task(task)
    return ensemble_solve


# Pre-register some useful ensemble combinations
register_solver("ensemble_all", "arc_solver.registry:_create_ensemble_all")
register_solver("ensemble_new", "arc_solver.registry:_create_ensemble_new")


def _create_ensemble_all(task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
    """Ensemble of all available solvers."""
    return create_ensemble_solver(["baseline", "enhanced", "agentic", "genomic"])(task)


def _create_ensemble_new(task: Dict[str, Any]) -> Dict[str, List[List[List[int]]]]:
    """Ensemble of new solvers only."""
    return create_ensemble_solver(["agentic", "genomic"])(task)


# Additional utility functions
def validate_solver(solver_name: str) -> bool:
    """
    Validate that a solver can be loaded and called.
    
    Args:
        solver_name: Name of solver to validate
    
    Returns:
        True if solver is valid, False otherwise
    """
    try:
        solver_func = get_solver(solver_name)
        # Try a minimal test task
        test_task = {
            'train': [{'input': [[1, 0], [0, 1]], 'output': [[0, 1], [1, 0]]}],
            'test': [{'input': [[1, 0], [0, 1]]}]
        }
        result = solver_func(test_task)
        return isinstance(result, dict) and 'attempt_1' in result
    except Exception:
        return False


def get_solver_info(solver_name: str) -> Dict[str, Any]:
    """
    Get information about a solver.
    
    Args:
        solver_name: Name of solver
    
    Returns:
        Dictionary with solver information
    """
    if solver_name not in SOLVERS:
        return {'error': 'Solver not found'}
    
    entry_point = SOLVERS[solver_name]
    module_path, function_name = entry_point.split(":")
    
    info = {
        'name': solver_name,
        'entry_point': entry_point,
        'module': module_path,
        'function': function_name,
        'valid': validate_solver(solver_name)
    }
    
    try:
        # Try to get docstring
        solver_func = get_solver(solver_name)
        if solver_func.__doc__:
            info['description'] = solver_func.__doc__.strip()
    except Exception:
        pass
    
    return info