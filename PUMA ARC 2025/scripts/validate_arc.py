#!/usr/bin/env python3
"""
Automated validation script for PUMA ARC Solver.

Runs solver on curated ARC subset with rule logging for regression detection.
Tracks:
- Accuracy per task
- Rule application success rates
- Object inventory quality
- LLM reasoning effectiveness
- Performance metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PUMA.arc_solver.solver import ARCSolver
from PUMA.arc_solver.rft_tracking import RFTTracker, TaskTrackingCache
from PUMA.arc_solver.pliance_engine import PlianceEngine
from PUMA.arc_solver.grid import Array


class ValidationSuite:
    """Validation suite for ARC solver."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        llm_enabled: bool = False
    ):
        """Initialize validation suite.

        Args:
            data_dir: Directory with ARC tasks (JSON files)
            output_dir: Directory for validation outputs
            llm_enabled: Whether to enable LLM reasoning
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.llm_enabled = llm_enabled

        # Initialize solver
        llm_config = {
            'enabled': llm_enabled,
            'use_llm_reasoning': llm_enabled
        } if llm_enabled else None

        runtime_flags = {
            'rft_first': True,
            'use_tracking': True,
            'use_pliance_fusion': True
        }

        self.solver = ARCSolver(
            llm_config=llm_config,
            runtime_flags=runtime_flags
        )

        self.tracking_cache = TaskTrackingCache(cache_dir=str(self.output_dir / "tracking_cache"))
        self.results: List[Dict[str, Any]] = []

    def load_task(self, task_file: Path) -> Dict[str, Any]:
        """Load ARC task from JSON file.

        Args:
            task_file: Path to task JSON

        Returns:
            Task dict with train/test pairs
        """
        with open(task_file, 'r') as f:
            return json.load(f)

    def run_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run solver on single task.

        Args:
            task_id: Task identifier
            task_data: Task data with train/test pairs

        Returns:
            Result dict with metrics
        """
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"{'='*60}")

        # Convert to numpy arrays
        train_pairs = [
            (
                np.array(pair['input'], dtype=np.int16),
                np.array(pair['output'], dtype=np.int16)
            )
            for pair in task_data['train']
        ]

        test_inputs = [
            np.array(pair['input'], dtype=np.int16)
            for pair in task_data['test']
        ]

        # Initialize tracking
        tracker = RFTTracker(task_id)
        tracker.ingest_training_pairs(train_pairs)
        self.tracking_cache.save_tracking_state(tracker)

        # Initialize pliance engine
        engine = PlianceEngine()

        # Emit rules from patterns
        schema = tracker.inventory.get_rule_friendly_schema()
        for pattern in schema.get('patterns', []):
            if pattern.get('type') == 'consistent_transformation':
                from PUMA.arc_solver.pliance_engine import ObjectSelector, RuleAction

                engine.emit_provisional_rule(
                    name=f"auto_{pattern['transformation']}",
                    selector=ObjectSelector(tags={'auto_detected'}),
                    action=RuleAction(action_type=pattern['transformation']),
                    confidence=0.7,
                    provenance='automated'
                )

        # Validate rules
        rule_validation = engine.validate_rules(train_pairs)

        # Solve
        start_time = time.time()
        programs = self.solver.solve(train_pairs, max_programs=256)
        solve_time = time.time() - start_time

        # Test predictions
        test_predictions = []
        test_correct = 0

        if programs:
            for i, test_input in enumerate(test_inputs):
                try:
                    from PUMA.arc_solver.dsl import apply_program
                    prediction = apply_program(programs[0], test_input)
                    test_predictions.append(prediction)

                    # Check if correct (if test output available)
                    if task_data['test'][i].get('output'):
                        expected = np.array(task_data['test'][i]['output'], dtype=np.int16)
                        if np.array_equal(prediction, expected):
                            test_correct += 1
                except Exception as e:
                    print(f"  Error applying program to test {i}: {e}")
                    test_predictions.append(None)

        # Gather metrics
        result = {
            'task_id': task_id,
            'status': 'success' if programs else 'no_solution',
            'solve_time': solve_time,
            'num_programs': len(programs),
            'test_accuracy': test_correct / len(test_inputs) if test_inputs else 0.0,
            'inventory': {
                'num_objects': len(tracker.inventory.entries),
                'num_patterns': len(schema.get('patterns', [])),
                'num_transformation_rules': len(schema.get('transformation_rules', []))
            },
            'tracking': {
                'num_conflicts': len(tracker.conflicts),
                'pending_repairs': sum(1 for t in tracker.repair_queue if t.status == 'queued')
            },
            'rules': {
                'total_rules': len(engine.rules),
                'rule_accuracy': {
                    rule_id: results['accuracy']
                    for rule_id, results in rule_validation.items()
                }
            },
            'search_stats': getattr(self.solver.search, 'search_stats', {})
        }

        self.results.append(result)

        # Print summary
        print(f"\nResult: {result['status']}")
        print(f"Solve time: {solve_time:.2f}s")
        print(f"Programs found: {len(programs)}")
        print(f"Test accuracy: {result['test_accuracy']:.2%}")
        print(f"Objects tracked: {result['inventory']['num_objects']}")
        print(f"Patterns detected: {result['inventory']['num_patterns']}")
        print(f"Rules created: {result['rules']['total_rules']}")
        print(f"Conflicts: {result['tracking']['num_conflicts']}")

        return result

    def run_suite(self, task_files: List[Path]) -> None:
        """Run validation on all tasks.

        Args:
            task_files: List of task file paths
        """
        print(f"Running validation on {len(task_files)} tasks...")
        print(f"LLM enabled: {self.llm_enabled}\n")

        for task_file in task_files:
            task_id = task_file.stem
            task_data = self.load_task(task_file)

            try:
                self.run_task(task_id, task_data)
            except Exception as e:
                print(f"ERROR in task {task_id}: {e}")
                import traceback
                traceback.print_exc()

                self.results.append({
                    'task_id': task_id,
                    'status': 'error',
                    'error': str(e)
                })

    def save_results(self) -> None:
        """Save validation results to JSON."""
        output_file = self.output_dir / "validation_results.json"

        # Compute summary
        summary = self._compute_summary()

        output = {
            'summary': summary,
            'results': self.results
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")

    def _compute_summary(self) -> Dict[str, Any]:
        """Compute summary statistics.

        Returns:
            Summary dict
        """
        if not self.results:
            return {}

        success_count = sum(1 for r in self.results if r['status'] == 'success')
        error_count = sum(1 for r in self.results if r['status'] == 'error')

        test_accuracies = [r['test_accuracy'] for r in self.results if 'test_accuracy' in r]
        solve_times = [r['solve_time'] for r in self.results if 'solve_time' in r]

        return {
            'total_tasks': len(self.results),
            'success_count': success_count,
            'error_count': error_count,
            'success_rate': success_count / len(self.results),
            'avg_test_accuracy': np.mean(test_accuracies) if test_accuracies else 0.0,
            'avg_solve_time': np.mean(solve_times) if solve_times else 0.0,
            'median_solve_time': np.median(solve_times) if solve_times else 0.0
        }

    def print_summary(self) -> None:
        """Print validation summary."""
        summary = self._compute_summary()

        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total tasks: {summary.get('total_tasks', 0)}")
        print(f"Success: {summary.get('success_count', 0)} ({summary.get('success_rate', 0):.1%})")
        print(f"Errors: {summary.get('error_count', 0)}")
        print(f"Avg test accuracy: {summary.get('avg_test_accuracy', 0):.1%}")
        print(f"Avg solve time: {summary.get('avg_solve_time', 0):.2f}s")
        print(f"Median solve time: {summary.get('median_solve_time', 0):.2f}s")
        print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate PUMA ARC Solver on curated task subset"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/arc_tasks',
        help='Directory with ARC task JSON files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='validation_output',
        help='Directory for validation outputs'
    )
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Enable LLM meta-reasoning'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        help='Specific task IDs to run (default: all)'
    )
    parser.add_argument(
        '--max-tasks',
        type=int,
        default=None,
        help='Maximum number of tasks to run'
    )

    args = parser.parse_args()

    # Find task files
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    if args.tasks:
        task_files = [data_dir / f"{tid}.json" for tid in args.tasks]
        task_files = [f for f in task_files if f.exists()]
    else:
        task_files = list(data_dir.glob("*.json"))

    if args.max_tasks:
        task_files = task_files[:args.max_tasks]

    if not task_files:
        print("Error: No task files found")
        sys.exit(1)

    # Run validation
    suite = ValidationSuite(
        data_dir=str(data_dir),
        output_dir=args.output_dir,
        llm_enabled=args.llm
    )

    suite.run_suite(task_files)
    suite.save_results()
    suite.print_summary()


if __name__ == '__main__':
    main()
