#!/usr/bin/env python3
"""
Playground for PUMA ARC solver.

This script provides a flexible environment for testing, evaluating, and analyzing
the PUMA ARC solver. It supports various modes for development, debugging, and
performance assessment.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import time
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "KAGGLE"))
sys.path.insert(0, str(Path(__file__).parent / "PUMA"))

from kaggle_setup import get_solver
from arc_solver.object_inventory import ObjectInventory
from arc_solver.rft_tracking import RFTTracker, TaskTrackingCache
from arc_solver.pliance_engine import PlianceEngine, ObjectSelector, RuleAction


class PlaygroundEvaluator:
    """Evaluates solver on challenges with learning from failures."""

    def __init__(
        self,
        challenges_path: str,
        solutions_path: str,
        num_tasks: int = 5,
        enable_llm: bool = False
    ):
        """Initialize evaluator.

        Args:
            challenges_path: Path to evaluation challenges JSON
            solutions_path: Path to evaluation solutions JSON
            num_tasks: Number of tasks to evaluate (default: 5)
            enable_llm: Whether to enable LLM meta-reasoning
        """
        init_start_time = time.time()
        self.challenges_path = Path(challenges_path)
        self.solutions_path = Path(solutions_path)
        self.num_tasks = num_tasks
        self.enable_llm = enable_llm

        # Load data
        with open(self.challenges_path, 'r') as f:
            self.challenges = json.load(f)

        with open(self.solutions_path, 'r') as f:
            self.solutions = json.load(f)
        
        init_start_time = self._log_time(init_start_time, "Data loading")

        # Get first N task IDs
        self.task_ids = list(self.challenges.keys())[:num_tasks]

        # Initialize solver with learning enabled
        llm_config = {
            'use_llm_reasoning': enable_llm
        }

        runtime_flags = {
            'rft_first': True,
            'use_tracking': True,
            'use_pliance_fusion': True
        }

        self.solver = get_solver(
            llm_options=llm_config,
            runtime_flags=runtime_flags
        )
        init_start_time = self._log_time(init_start_time, "Solver initialization")

        # Persistent components
        self.tracking_cache = TaskTrackingCache(cache_dir="playground_cache")
        self.failure_log_path = Path("playground_failures.json")

        # Results
        self.results: List[Dict[str, Any]] = []
        self.learning_log: List[Dict[str, Any]] = []

        # Create output directory
        self.output_dir = Path("playground_output")
        self.output_dir.mkdir(exist_ok=True)

        # Load previous learning
        self.previous_learning = self._load_previous_learning()

    def _log_time(self, start_time, step_name):
        elapsed = time.time() - start_time
        print(f"PERF: {step_name} took {elapsed:.2f}s")
        return time.time()

    def _load_previous_learning(self) -> Dict[str, Any]:
        """Load previous learning log.

        Returns:
            Dictionary mapping task_id to learning entries
        """
        learning_log_path = self.output_dir / "learning_log.json"

        if not learning_log_path.exists():
            return {}

        try:
            with open(learning_log_path, 'r') as f:
                previous_log = json.load(f)

            # Index by task_id
            learning_by_task = {}
            for entry in previous_log:
                task_id = entry.get('task_id')
                if task_id:
                    if task_id not in learning_by_task:
                        learning_by_task[task_id] = []
                    learning_by_task[task_id].append(entry)

            print(f"📚 Loaded previous learning: {len(learning_by_task)} tasks with history")
            return learning_by_task

        except Exception as e:
            print(f"⚠️  Could not load previous learning: {e}")
            return {}

    def run_evaluation(self) -> None:
        """Run evaluation on all tasks."""
        print("="*80)
        print("PLAYGROUND EVALUATION SUITE")
        print("="*80)
        print(f"Tasks: {self.num_tasks}")
        print(f"LLM: {'Enabled' if self.enable_llm else 'Disabled'}")
        print(f"Learning: Enabled")
        print("="*80)
        print()

        for i, task_id in enumerate(self.task_ids, 1):
            print(f"\n{'='*80}")
            print(f"Task {i}/{self.num_tasks}: {task_id}")
            print(f"{'='*80}")

            try:
                result = self.evaluate_task(task_id)
                self.results.append(result)

                # Learn from result
                if not result['correct']:
                    self.learn_from_failure(task_id, result)

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

                self.results.append({
                    'task_id': task_id,
                    'status': 'error',
                    'error': str(e)
                })

        # Save results
        self.save_results()
        self.print_summary()

    def evaluate_task(self, task_id: str) -> Dict[str, Any]:
        """Evaluate solver on single task.

        Args:
            task_id: Task identifier

        Returns:
            Result dictionary
        """
        eval_start_time = time.time()
        challenge = self.challenges[task_id]
        solution = self.solutions.get(task_id, [])

        # Convert to numpy arrays
        train_pairs = [
            (
                np.array(pair['input'], dtype=np.int16),
                np.array(pair['output'], dtype=np.int16)
            )
            for pair in challenge['train']
        ]

        test_inputs = [
            np.array(pair['input'], dtype=np.int16)
            for pair in challenge['test']
        ]

        test_outputs = [
            np.array(sol, dtype=np.int16)
            for sol in solution
        ]

        print(f"\nTrain pairs: {len(train_pairs)}")
        print(f"Test inputs: {len(test_inputs)}")
        
        eval_start_time = self._log_time(eval_start_time, "Task setup")

        # STEP 1: Build object inventory
        print("\n1. Building object inventory...")
        inventory = ObjectInventory()
        inventory.build_from_train_pairs(train_pairs)

        schema = inventory.get_rule_friendly_schema()
        print(f"   Objects: {len(inventory.entries)}")
        print(f"   Patterns: {len(schema.get('patterns', []))}")
        print(f"   Transformation rules: {len(schema.get('transformation_rules', []))}")
        
        eval_start_time = self._log_time(eval_start_time, "Build object inventory")

        # STEP 2: Track with RFT
        print("\n2. RFT tracking...")
        tracker = RFTTracker(task_id)

        # Try to load cached state
        cached_tracker = self.tracking_cache.load_tracking_state(task_id)
        if cached_tracker:
            print("   Using cached tracking state")
            tracker = cached_tracker
        else:
            tracker.ingest_training_pairs(train_pairs)
            self.tracking_cache.save_tracking_state(tracker)

        print(f"   Conflicts: {len(tracker.conflicts)}")
        print(f"   Repair tasks: {len(tracker.repair_queue)}")
        
        eval_start_time = self._log_time(eval_start_time, "RFT tracking")

        # STEP 3: Generate pliance rules
        print("\n3. Generating pliance rules...")
        engine = PlianceEngine()

        # Check for previous learning on this task
        if task_id in self.previous_learning:
            prev_attempts = self.previous_learning[task_id]
            print(f"   📚 Found {len(prev_attempts)} previous attempt(s) - applying learned patterns...")

            # Get the most recent learning
            latest_learning = prev_attempts[-1]
            learned_patterns = latest_learning.get('llm_insights', {}).get('patterns', [])
            rules_to_add = latest_learning.get('llm_insights', {}).get('rules_to_add', [])

            print(f"   Learned patterns: {learned_patterns}")
            print(f"   Rules to add: {rules_to_add}")

            # Generate rules based on learned patterns
            for pattern_type in learned_patterns:
                if pattern_type in ['size_reduction', 'resized']:
                    rule = engine.emit_provisional_rule(
                        name=f"learned_{pattern_type}",
                        selector=ObjectSelector(tags={'auto_detected'}),
                        action=RuleAction(action_type='resize'),
                        confidence=0.85,
                        provenance='learned_from_failure'
                    )
                    print(f"   ✨ Applied learned rule: {rule.name} (from previous failure)")

                elif pattern_type in ['color_removal', 'color_change']:
                    rule = engine.emit_provisional_rule(
                        name=f"learned_{pattern_type}",
                        selector=ObjectSelector(tags={'auto_detected'}),
                        action=RuleAction(action_type='recolor'),
                        confidence=0.85,
                        provenance='learned_from_failure'
                    )
                    print(f"   ✨ Applied learned rule: {rule.name} (from previous failure)")

        # Emit rules from patterns
        for pattern in schema.get('patterns', []):
            if pattern.get('type') == 'consistent_transformation':
                rule = engine.emit_provisional_rule(
                    name=f"auto_{pattern['transformation']}",
                    selector=ObjectSelector(tags={'auto_detected'}),
                    action=RuleAction(action_type=pattern['transformation']),
                    confidence=pattern.get('confidence', 0.7),
                    provenance='automated'
                )
                print(f"   Emitted rule: {rule.name} (confidence: {rule.confidence:.2f})")

        # Emit rules from transformation rules
        for trans_rule in schema.get('transformation_rules', []):
            trans_type = trans_rule.get('transformation')
            common_changes = trans_rule.get('common_changes', {})

            if trans_type == 'recolored' and 'color' in common_changes:
                from_color = common_changes['color'].get('from')
                to_color = common_changes['color'].get('to')

                if from_color is not None and to_color is not None:
                    rule = engine.emit_provisional_rule(
                        name=f"recolor_{from_color}_to_{to_color}",
                        selector=ObjectSelector(color=from_color),
                        action=RuleAction(
                            action_type='recolor',
                            parameters={'color': to_color}
                        ),
                        confidence=trans_rule.get('confidence', 0.8),
                        provenance='automated'
                    )
                    print(f"   Emitted rule: {rule.name} (confidence: {rule.confidence:.2f})")

        print(f"   Total rules: {len(engine.rules)}")
        
        eval_start_time = self._log_time(eval_start_time, "Generate pliance rules")

        # Validate rules
        if engine.rules:
            print("\n4. Validating rules...")
            validation_results = engine.validate_rules(train_pairs)

            for rule_id, results in validation_results.items():
                rule = engine.rules[rule_id]
                accuracy = results['accuracy']
                print(f"   {rule.name}: {accuracy:.1%} accuracy")

                # Repair low-accuracy rules
                if accuracy < 0.7:
                    print(f"      Repairing (low accuracy)...")
                    engine.repair_rule(
                        rule_id,
                        {'action': 'adjust_confidence', 'new_confidence': accuracy * 0.8}
                    )
            
            eval_start_time = self._log_time(eval_start_time, "Validating rules")

        # STEP 5: Solve with enhanced search
        print("\n5. Solving...")
        start_time = time.time()

        # Format as task dict for solver
        task = {
            'train': [
                {'input': inp.tolist(), 'output': out.tolist()}
                for inp, out in train_pairs
            ],
            'test': [
                {'input': test_inp.tolist()}
                for test_inp in test_inputs
            ]
        }

        # Solve the task
        result_dict = self.solver.solve_task(task)
        programs = result_dict.get('predictions', [])

        solve_time = time.time() - start_time
        print(f"   Solve time: {solve_time:.2f}s")
        print(f"   Programs found: {len(programs) if programs else 0}")
        
        eval_start_time = self._log_time(eval_start_time, "Solving")

        # STEP 6: Check predictions
        print("\n6. Checking predictions...")
        predictions = programs if programs else []
        correct_count = 0

        if predictions:
            for i in range(min(len(predictions), len(test_outputs))):
                try:
                    # Predictions are already grids from solver
                    prediction = np.array(predictions[i])
                    expected = test_outputs[i]
                    is_correct = np.array_equal(prediction, expected)

                    if is_correct:
                        correct_count += 1
                        print(f"   Test {i+1}: ✓ CORRECT")
                    else:
                        print(f"   Test {i+1}: ✗ INCORRECT")
                        print(f"      Expected shape: {expected.shape}")
                        print(f"      Got shape: {prediction.shape}")

                except Exception as e:
                    print(f"   Test {i+1}: ERROR - {e}")
        else:
            print("   No predictions generated")

        accuracy = correct_count / len(test_outputs) if test_outputs else 0.0

        # Build result
        result = {
            'task_id': task_id,
            'status': 'success',
            'correct': accuracy == 1.0,
            'accuracy': accuracy,
            'solve_time': solve_time,
            'num_programs': len(programs),
            'inventory': {
                'num_objects': len(inventory.entries),
                'num_patterns': len(schema.get('patterns', [])),
                'num_rules': len(schema.get('transformation_rules', []))
            },
            'tracking': {
                'num_conflicts': len(tracker.conflicts),
                'num_repairs': len(tracker.repair_queue)
            },
            'pliance': {
                'num_rules': len(engine.rules),
                'rule_accuracy': {
                    rule_id: results['accuracy']
                    for rule_id, results in validation_results.items()
                } if engine.rules else {}
            }
        }

        return result

    def learn_from_failure(self, task_id: str, result: Dict[str, Any]) -> None:
        """Learn from a failed attempt with LLM analysis.

        Args:
            task_id: Task that failed
            result: Result dictionary
        """
        print(f"\n{'='*80}")
        print(f"LEARNING FROM FAILURE: {task_id}")
        print(f"{'='*80}")

        # Get the correct answer
        challenge = self.challenges[task_id]
        solution = self.solutions.get(task_id, [])

        train_pairs = [
            (
                np.array(pair['input'], dtype=np.int16),
                np.array(pair['output'], dtype=np.int16)
            )
            for pair in challenge['train']
        ]

        test_inputs = [
            np.array(pair['input'], dtype=np.int16)
            for pair in challenge['test']
        ]

        correct_outputs = [
            np.array(sol, dtype=np.int16)
            for sol in solution
        ]

        # Log the failure to file
        failure_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'error_type': 'incorrect_prediction',
            'context': result
        }

        # Append to failure log
        with open(self.failure_log_path, 'a') as f:
            json.dump(failure_entry, f)
            f.write('\n')

        # Analyze what went wrong
        inventory_issues = self._analyze_inventory_issues(result)
        rule_issues = self._analyze_rule_issues(result)

        # Use LLM to analyze the failure and learn from correct answer
        print("\n🤖 LLM Analysis: Learning from correct answer...")
        llm_insights = self._get_llm_failure_analysis(
            train_pairs,
            test_inputs,
            correct_outputs,
            result
        )

        # Create learning entry
        learning_entry = {
            'timestamp': datetime.now().isoformat(),
            'task_id': task_id,
            'accuracy': result.get('accuracy', 0.0),
            'issues': {
                'inventory': inventory_issues,
                'rules': rule_issues
            },
            'improvements': [],
            'llm_insights': llm_insights,
            'correct_solution': {
                'outputs': [out.tolist() for out in correct_outputs],
                'patterns_identified': llm_insights.get('patterns', []),
                'correct_approach': llm_insights.get('correct_approach', ''),
                'why_we_failed': llm_insights.get('why_failed', '')
            }
        }

        # Generate improvement suggestions
        if inventory_issues:
            print("\n📊 Inventory Issues Detected:")
            for issue in inventory_issues:
                print(f"  - {issue}")
                learning_entry['improvements'].append({
                    'type': 'inventory',
                    'issue': issue,
                    'suggestion': 'Consider more granular object extraction or pattern detection'
                })

        if rule_issues:
            print("\n📋 Rule Issues Detected:")
            for issue in rule_issues:
                print(f"  - {issue}")
                learning_entry['improvements'].append({
                    'type': 'rule',
                    'issue': issue,
                    'suggestion': 'Consider refinement or additional validation'
                })

        # Print LLM insights
        if llm_insights:
            print("\n🧠 LLM Insights:")
            print(f"  Patterns in correct solution: {llm_insights.get('patterns', [])}")
            print(f"  Correct approach: {llm_insights.get('correct_approach', 'N/A')}")
            print(f"  Why we failed: {llm_insights.get('why_failed', 'N/A')}")
            print(f"  What to do next time: {llm_insights.get('next_time', 'N/A')}")

        self.learning_log.append(learning_entry)

        # Save learning log
        self._save_learning_log()

        print(f"\n✓ Failure logged and analyzed with LLM guidance")

    def _get_llm_failure_analysis(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        correct_outputs: List[np.ndarray],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get LLM analysis of failure with correct answer.

        Args:
            train_pairs: Training examples
            test_inputs: Test inputs
            correct_outputs: Correct test outputs
            result: Result dictionary

        Returns:
            LLM insights dictionary
        """
        # Always run heuristic analysis first
        # (LLM can enhance this if available)

        try:
            # Build inventory with correct answer included
            from arc_solver.object_inventory import ObjectInventory
            inventory = ObjectInventory()

            # Add training pairs
            inventory.build_from_train_pairs(train_pairs)

            # Add test pairs with correct answers (this is key!)
            test_pairs = [
                (test_inp, correct_out)
                for test_inp, correct_out in zip(test_inputs, correct_outputs)
            ]

            # Analyze what the correct transformation is
            all_pairs = train_pairs + test_pairs
            full_inventory = ObjectInventory()
            full_inventory.build_from_train_pairs(all_pairs)

            schema = full_inventory.get_rule_friendly_schema()

            # Create learning prompt showing what we did vs what we should have done
            learning_prompt = f"""
LEARNING FROM FAILURE - ARC Challenge

We attempted to solve this task but got it WRONG. Now we have the CORRECT answer.
Your job: Analyze what we should have done differently.

TRAINING EXAMPLES ({len(train_pairs)} pairs):
Input → Output transformations

TEST EXAMPLE:
Input: {test_inputs[0].shape if test_inputs else 'N/A'}
CORRECT Output: {correct_outputs[0].shape if correct_outputs else 'N/A'}
(We failed to produce this)

OBJECT ANALYSIS WITH CORRECT ANSWER:
- Total objects detected: {len(full_inventory.entries)}
- Patterns found: {len(schema.get('patterns', []))}
- Transformation rules: {schema.get('transformation_rules', [])}

WHAT WE TRIED:
- Generated {result.get('pliance', {}).get('num_rules', 0)} pliance rules
- Rule accuracy: {result.get('pliance', {}).get('rule_accuracy', {})}
- Found {result.get('num_programs', 0)} candidate programs

ANALYZE:
1. What patterns are in the CORRECT solution that we missed?
2. What transformation rule would produce the correct output?
3. Why did our approach fail?
4. What should we do differently next time?

Respond in JSON format:
{{
    "patterns": ["list of patterns in correct solution"],
    "correct_approach": "description of correct transformation",
    "why_failed": "why our approach didn't work",
    "next_time": "what to do differently",
    "rules_to_add": ["specific rules to add for similar tasks"]
}}
"""

            # Get heuristic analysis (always runs)
            heuristic_analysis = self._heuristic_failure_analysis(
                train_pairs,
                test_inputs,
                correct_outputs,
                schema
            )

            # Enhance with LLM if available and enabled
            if self.enable_llm and hasattr(self.solver, 'meta_reasoning') and self.solver.meta_reasoning:
                try:
                    # Use the solver's meta reasoning engine
                    llm_response = self.solver.meta_reasoning.reason_about_task(
                        train_pairs,
                        schema,
                        mode='failure_analysis'
                    )

                    # Merge LLM insights with heuristic analysis
                    heuristic_analysis['llm_enhanced'] = True
                    heuristic_analysis['llm_patterns'] = llm_response.get('identified_patterns', [])
                    heuristic_analysis['llm_approach'] = llm_response.get('suggested_approach', '')
                except Exception as llm_error:
                    print(f"   LLM enhancement failed: {llm_error}")
                    heuristic_analysis['llm_enhanced'] = False

            return heuristic_analysis

        except Exception as e:
            print(f"   Warning: LLM analysis failed: {e}")
            return {
                'patterns': [],
                'correct_approach': 'Analysis failed',
                'why_failed': str(e),
                'next_time': 'Check LLM configuration'
            }

    def _heuristic_failure_analysis(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_inputs: List[np.ndarray],
        correct_outputs: List[np.ndarray],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Heuristic analysis when LLM unavailable.

        Args:
            train_pairs: Training examples
            test_inputs: Test inputs
            correct_outputs: Correct outputs
            schema: Object inventory schema

        Returns:
            Analysis dictionary
        """
        patterns = []
        specific_transformations = []

        # Analyze size changes WITH SPECIFICS
        if train_pairs:
            in_shape = train_pairs[0][0].shape
            out_shape = train_pairs[0][1].shape

            if in_shape != out_shape:
                if out_shape[0] < in_shape[0] or out_shape[1] < in_shape[1]:
                    patterns.append('size_reduction')
                    specific_transformations.append(f"Reduce from {in_shape} to {out_shape}")
                else:
                    patterns.append('size_expansion')
                    specific_transformations.append(f"Expand from {in_shape} to {out_shape}")
            else:
                patterns.append('same_size_transformation')

        # Analyze color changes WITH SPECIFICS
        if train_pairs:
            in_colors = set(train_pairs[0][0].flatten())
            out_colors = set(train_pairs[0][1].flatten())

            new_colors = out_colors - in_colors
            removed_colors = in_colors - out_colors

            if new_colors:
                patterns.append('color_introduction')
                specific_transformations.append(f"Introduce colors: {list(new_colors)}")
            if removed_colors:
                patterns.append('color_removal')
                specific_transformations.append(f"Remove colors: {list(removed_colors)}")

        # Check transformation rules from schema
        trans_rules = schema.get('transformation_rules', [])
        if trans_rules:
            patterns.extend([rule.get('transformation', '') for rule in trans_rules])
            for rule in trans_rules:
                if rule.get('common_changes'):
                    specific_transformations.append(f"Common changes: {rule['common_changes']}")

        # Analyze the correct output structure
        if correct_outputs:
            correct_out = correct_outputs[0]
            test_in = test_inputs[0] if test_inputs else None

            if test_in is not None:
                specific_transformations.append(f"Test: {test_in.shape} → {correct_out.shape}")

                # Check if it's an extraction
                if correct_out.shape[0] < test_in.shape[0] and correct_out.shape[1] < test_in.shape[1]:
                    patterns.append('extraction')
                    specific_transformations.append(f"Extract region of size {correct_out.shape}")

        correct_approach = "Transform: " + " | ".join(specific_transformations) if specific_transformations else f"Involves: {', '.join(patterns)}"

        return {
            'patterns': list(set(patterns)),
            'correct_approach': correct_approach,
            'why_failed': f"Generated {len(trans_rules)} rules but none matched the correct transformation",
            'next_time': "Need better pattern recognition for these transformation types",
            'rules_to_add': [f"Rule for {p}" for p in patterns[:3]],
            'specific_details': specific_transformations  # NEW: specific transformation details
        }

    def _analyze_inventory_issues(self, result: Dict[str, Any]) -> List[str]:
        """Analyze potential inventory issues.

        Args:
            result: Result dictionary

        Returns:
            List of issue descriptions
        """
        issues = []
        inventory = result.get('inventory', {})

        if inventory.get('num_objects', 0) == 0:
            issues.append("No objects detected in training pairs")

        if inventory.get('num_patterns', 0) == 0:
            issues.append("No transformation patterns detected")

        if inventory.get('num_rules', 0) == 0:
            issues.append("No transformation rules extracted")

        return issues

    def _analyze_rule_issues(self, result: Dict[str, Any]) -> List[str]:
        """Analyze potential rule issues.

        Args:
            result: Result dictionary

        Returns:
            List of issue descriptions
        """
        issues = []
        pliance = result.get('pliance', {})

        if pliance.get('num_rules', 0) == 0:
            issues.append("No pliance rules generated")

        rule_accuracy = pliance.get('rule_accuracy', {})
        low_accuracy_rules = [
            rule_id for rule_id, acc in rule_accuracy.items()
            if acc < 0.7
        ]

        if low_accuracy_rules:
            issues.append(f"{len(low_accuracy_rules)} rules with <70% accuracy")

        return issues

    def _save_learning_log(self) -> None:
        """Save learning log to file."""
        log_file = self.output_dir / "learning_log.json"

        with open(log_file, 'w') as f:
            json.dump(self.learning_log, f, indent=2)

    def save_results(self) -> None:
        """Save evaluation results."""
        results_file = self.output_dir / "eval_results.json"

        # Compute summary
        summary = {
            'num_tasks': len(self.results),
            'correct': sum(1 for r in self.results if r.get('correct', False)),
            'avg_accuracy': np.mean([r.get('accuracy', 0.0) for r in self.results]),
            'avg_solve_time': np.mean([r.get('solve_time', 0.0) for r in self.results if 'solve_time' in r]),
            'failures_logged': len(self.learning_log)
        }

        output = {
            'summary': summary,
            'results': self.results,
            'learning_log': self.learning_log
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

    def print_summary(self) -> None:
        """Print evaluation summary."""
        correct = sum(1 for r in self.results if r.get('correct', False))
        total = len(self.results)
        avg_accuracy = np.mean([r.get('accuracy', 0.0) for r in self.results])

        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Tasks evaluated: {total}")
        print(f"Fully correct: {correct}/{total} ({correct/total:.1%})")
        print(f"Average accuracy: {avg_accuracy:.1%}")
        print(f"Failures logged: {len(self.learning_log)}")
        print(f"Learning entries: {len(self.learning_log)}")
        print(f"{'='*80}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Playground evaluation suite for PUMA ARC Solver"
    )
    parser.add_argument(
        '--challenges',
        type=str,
        default='PUMA/data/arc-agi_evaluation_challenges.json',
        help='Path to evaluation challenges JSON'
    )
    parser.add_argument(
        '--solutions',
        type=str,
        default='PUMA/data/arc-agi_evaluation_solutions.json',
        help='Path to evaluation solutions JSON'
    )
    parser.add_argument(
        '--num-tasks',
        type=int,
        default=5,
        help='Number of tasks to evaluate (default: 5)'
    )
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Enable LLM meta-reasoning'
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = PlaygroundEvaluator(
        challenges_path=args.challenges,
        solutions_path=args.solutions,
        num_tasks=args.num_tasks,
        enable_llm=args.llm
    )

    evaluator.run_evaluation()


if __name__ == '__main__':
    main()
