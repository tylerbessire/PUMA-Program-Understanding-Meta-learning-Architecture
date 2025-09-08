"""
Configuration and benchmarking utilities for the enhanced ARC solver.

This module provides tools to configure the solver, run benchmarks, and
evaluate performance on test datasets.
"""

from __future__ import annotations

import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import os

from arc_solver.enhanced_solver import ARCSolver, solve_task_enhanced, solve_task_baseline
from arc_solver.neural.episodic import EpisodicRetrieval
from arc_solver.neural.sketches import SketchMiner


class SolverConfig:
    """Configuration for the enhanced ARC solver."""
    
    def __init__(self):
        self.use_neural_guidance = True
        self.use_episodic_retrieval = True
        self.use_program_sketches = True
        self.use_test_time_training = True
        self.max_programs = 256
        self.neural_guidance_model_path = "neural_guidance_model.json"
        self.episode_db_path = "episodes.json"
        self.sketches_path = "sketches.json"
        self.timeout_per_task = 30.0  # seconds
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'use_neural_guidance': self.use_neural_guidance,
            'use_episodic_retrieval': self.use_episodic_retrieval,
            'use_program_sketches': self.use_program_sketches,
            'use_test_time_training': self.use_test_time_training,
            'max_programs': self.max_programs,
            'neural_guidance_model_path': self.neural_guidance_model_path,
            'episode_db_path': self.episode_db_path,
            'sketches_path': self.sketches_path,
            'timeout_per_task': self.timeout_per_task,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolverConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def save(self, filepath: str):
        """Save config to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SolverConfig':
        """Load config from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class Benchmark:
    """Benchmarking utilities for ARC solver evaluation."""
    
    def __init__(self, config: SolverConfig):
        self.config = config
        self.results = {}
    
    def run_benchmark(self, test_data_path: str, 
                     output_path: str = "benchmark_results.json",
                     max_tasks: Optional[int] = None) -> Dict[str, Any]:
        """Run benchmark on test dataset."""
        print(f"Loading test data from {test_data_path}")
        
        if not os.path.exists(test_data_path):
            print(f"Test data file not found: {test_data_path}")
            return {}
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        tasks = list(test_data.items())
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        print(f"Running benchmark on {len(tasks)} tasks...")
        
        results = {
            'config': self.config.to_dict(),
            'total_tasks': len(tasks),
            'completed_tasks': 0,
            'successful_tasks': 0,
            'task_results': {},
            'performance_stats': {},
            'timing': {},
        }
        
        total_time = 0
        solver = ARCSolver(use_enhancements=True)
        
        for i, (task_id, task_data) in enumerate(tasks):
            print(f"Processing task {i+1}/{len(tasks)}: {task_id}")
            
            start_time = time.time()
            try:
                solution = solver.solve_task(task_data)
                task_time = time.time() - start_time
                total_time += task_time
                
                # Evaluate solution if ground truth is available
                success = self._evaluate_solution(task_data, solution)
                
                results['task_results'][task_id] = {
                    'success': success,
                    'time': task_time,
                    'solution': solution,
                }
                
                if success:
                    results['successful_tasks'] += 1
                
                results['completed_tasks'] += 1
                
            except Exception as e:
                print(f"Error processing task {task_id}: {e}")
                results['task_results'][task_id] = {
                    'success': False,
                    'error': str(e),
                    'time': time.time() - start_time,
                }
                results['completed_tasks'] += 1
        
        # Calculate performance statistics
        success_rate = results['successful_tasks'] / results['completed_tasks'] if results['completed_tasks'] > 0 else 0
        avg_time = total_time / results['completed_tasks'] if results['completed_tasks'] > 0 else 0
        
        results['performance_stats'] = {
            'success_rate': success_rate,
            'average_time_per_task': avg_time,
            'total_time': total_time,
        }
        
        # Get solver statistics
        solver_stats = solver.get_statistics()
        results['solver_stats'] = solver_stats
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Benchmark completed! Results saved to {output_path}")
        print(f"Success rate: {success_rate:.3f} ({results['successful_tasks']}/{results['completed_tasks']})")
        print(f"Average time per task: {avg_time:.2f}s")
        
        return results
    
    def _evaluate_solution(self, task_data: Dict[str, Any], solution: Dict[str, Any]) -> bool:
        """Evaluate if solution is correct (requires ground truth)."""
        # This is a placeholder - in practice, you'd compare against known outputs
        # For ARC-AGI-2, the test set doesn't have public ground truth
        return False  # Cannot evaluate without ground truth
    
    def compare_methods(self, test_data_path: str, max_tasks: int = 10) -> Dict[str, Any]:
        """Compare enhanced solver vs baseline solver."""
        print(f"Comparing enhanced vs baseline solver on {max_tasks} tasks...")
        
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
        
        tasks = list(test_data.items())[:max_tasks]
        
        enhanced_results = []
        baseline_results = []
        
        for task_id, task_data in tasks:
            print(f"Processing {task_id}...")
            
            # Test enhanced solver
            start_time = time.time()
            try:
                enhanced_solution = solve_task_enhanced(task_data)
                enhanced_time = time.time() - start_time
                enhanced_success = True  # Placeholder
            except Exception:
                enhanced_time = time.time() - start_time
                enhanced_success = False
            
            # Test baseline solver
            start_time = time.time()
            try:
                baseline_solution = solve_task_baseline(task_data)
                baseline_time = time.time() - start_time
                baseline_success = True  # Placeholder
            except Exception:
                baseline_time = time.time() - start_time
                baseline_success = False
            
            enhanced_results.append({
                'task_id': task_id,
                'success': enhanced_success,
                'time': enhanced_time,
            })
            
            baseline_results.append({
                'task_id': task_id,
                'success': baseline_success,
                'time': baseline_time,
            })
        
        # Calculate comparison statistics
        enhanced_success_rate = np.mean([r['success'] for r in enhanced_results])
        baseline_success_rate = np.mean([r['success'] for r in baseline_results])
        enhanced_avg_time = np.mean([r['time'] for r in enhanced_results])
        baseline_avg_time = np.mean([r['time'] for r in baseline_results])
        
        comparison = {
            'enhanced': {
                'success_rate': enhanced_success_rate,
                'avg_time': enhanced_avg_time,
                'results': enhanced_results,
            },
            'baseline': {
                'success_rate': baseline_success_rate,
                'avg_time': baseline_avg_time,
                'results': baseline_results,
            },
            'improvement': {
                'success_rate_delta': enhanced_success_rate - baseline_success_rate,
                'time_ratio': enhanced_avg_time / baseline_avg_time if baseline_avg_time > 0 else float('inf'),
            }
        }
        
        print(f"Enhanced success rate: {enhanced_success_rate:.3f}")
        print(f"Baseline success rate: {baseline_success_rate:.3f}")
        print(f"Success rate improvement: {comparison['improvement']['success_rate_delta']:.3f}")
        print(f"Time ratio (enhanced/baseline): {comparison['improvement']['time_ratio']:.2f}")
        
        return comparison


def create_default_config() -> SolverConfig:
    """Create default configuration."""
    return SolverConfig()


def setup_solver_environment():
    """Set up the solver environment with default components."""
    print("Setting up ARC solver environment...")
    
    # Create default config
    config = create_default_config()
    config.save("solver_config.json")
    print("Saved default configuration to solver_config.json")
    
    # Initialize empty episodic database
    episodic = EpisodicRetrieval()
    episodic.save()
    print("Initialized episodic retrieval database")
    
    # Initialize sketch miner
    sketch_miner = SketchMiner()
    sketch_miner.save_sketches("sketches.json")
    print("Initialized program sketches database")
    
    print("Environment setup complete!")


if __name__ == "__main__":
    # Set up environment
    setup_solver_environment()
    
    # Example benchmark run (would need actual test data)
    # config = SolverConfig()
    # benchmark = Benchmark(config)
    # results = benchmark.run_benchmark("arc_test_data.json", max_tasks=5)
