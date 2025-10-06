#!/usr/bin/env python3
"""
Submission Validator

Validates submission.json format for Kaggle ARC competition.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple


def validate_grid(grid: Any) -> Tuple[bool, str]:
    """
    Validate a grid is properly formatted.

    Returns: (is_valid, error_message)
    """
    if not isinstance(grid, list):
        return False, f"Grid must be list, got {type(grid).__name__}"

    if len(grid) == 0:
        return False, "Grid cannot be empty"

    for i, row in enumerate(grid):
        if not isinstance(row, list):
            return False, f"Row {i} must be list, got {type(row).__name__}"

        if len(row) == 0:
            return False, f"Row {i} cannot be empty"

        for j, cell in enumerate(row):
            if not isinstance(cell, int):
                return False, f"Cell [{i},{j}] must be int, got {type(cell).__name__}"

            if not (0 <= cell <= 9):
                return False, f"Cell [{i},{j}] must be 0-9, got {cell}"

    # Check all rows same length
    row_lengths = [len(row) for row in grid]
    if len(set(row_lengths)) > 1:
        return False, f"Inconsistent row lengths: {row_lengths}"

    return True, ""


def validate_test_case(test_case: Any, task_id: str, test_idx: int) -> List[str]:
    """
    Validate a single test case.

    Returns: List of error messages (empty if valid)
    """
    errors = []

    if not isinstance(test_case, dict):
        errors.append(f"Test case must be dict, got {type(test_case).__name__}")
        return errors

    # Check required keys
    if 'attempt_1' not in test_case:
        errors.append("Missing 'attempt_1' key")
    if 'attempt_2' not in test_case:
        errors.append("Missing 'attempt_2' key")

    if errors:
        return errors

    # Validate grids
    for attempt_key in ['attempt_1', 'attempt_2']:
        grid = test_case[attempt_key]
        is_valid, error_msg = validate_grid(grid)
        if not is_valid:
            errors.append(f"{attempt_key}: {error_msg}")

    return errors


def validate_task_entry(task_entry: Any, task_id: str) -> List[str]:
    """
    Validate a task entry (list of test cases).

    Returns: List of error messages (empty if valid)
    """
    errors = []

    if not isinstance(task_entry, list):
        errors.append(f"Task entry must be list, got {type(task_entry).__name__}")
        return errors

    if len(task_entry) == 0:
        errors.append("Task entry cannot be empty list")
        return errors

    # Validate each test case
    for i, test_case in enumerate(task_entry):
        test_errors = validate_test_case(test_case, task_id, i)
        for error in test_errors:
            errors.append(f"Test case {i}: {error}")

    return errors


def validate_submission(submission: Dict[str, Any], verbose: bool = True) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validate entire submission.

    Returns: (is_valid, errors_by_task)
    """

    if not isinstance(submission, dict):
        print("❌ FATAL: Submission must be a dictionary")
        return False, {}

    errors_by_task = {}
    tasks_checked = 0
    tasks_with_errors = 0

    for task_id, task_entry in submission.items():
        tasks_checked += 1
        task_errors = validate_task_entry(task_entry, task_id)

        if task_errors:
            tasks_with_errors += 1
            errors_by_task[task_id] = task_errors

    is_valid = len(errors_by_task) == 0

    if verbose:
        print("=" * 70)
        print("SUBMISSION VALIDATION REPORT")
        print("=" * 70)
        print(f"Tasks checked: {tasks_checked}")
        print(f"Tasks with errors: {tasks_with_errors}")
        print(f"Tasks valid: {tasks_checked - tasks_with_errors}")
        print()

        if is_valid:
            print("✅ VALIDATION PASSED - Submission format is correct!")
        else:
            print("❌ VALIDATION FAILED - Issues found:")
            print()

            # Show first 10 tasks with errors
            for task_id in list(errors_by_task.keys())[:10]:
                task_errors = errors_by_task[task_id]
                print(f"Task {task_id}:")
                for error in task_errors[:5]:  # First 5 errors per task
                    print(f"  - {error}")
                if len(task_errors) > 5:
                    print(f"  ... and {len(task_errors) - 5} more errors")
                print()

            if len(errors_by_task) > 10:
                print(f"... and {len(errors_by_task) - 10} more tasks with errors")

        print("=" * 70)

    return is_valid, errors_by_task


def analyze_submission_stats(submission: Dict[str, Any]) -> None:
    """Print statistics about the submission."""

    print("\n" + "=" * 70)
    print("SUBMISSION STATISTICS")
    print("=" * 70)

    total_tasks = len(submission)
    total_test_cases = 0
    grid_sizes = []

    for task_id, task_entry in submission.items():
        if isinstance(task_entry, list):
            total_test_cases += len(task_entry)

            for test_case in task_entry:
                if isinstance(test_case, dict) and 'attempt_1' in test_case:
                    grid = test_case['attempt_1']
                    if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
                        height = len(grid)
                        width = len(grid[0])
                        grid_sizes.append((height, width))

    print(f"Total tasks: {total_tasks}")
    print(f"Total test cases: {total_test_cases}")
    print(f"Avg test cases per task: {total_test_cases / total_tasks:.2f}")

    if grid_sizes:
        avg_height = sum(h for h, w in grid_sizes) / len(grid_sizes)
        avg_width = sum(w for h, w in grid_sizes) / len(grid_sizes)
        print(f"Avg grid size: {avg_height:.1f} × {avg_width:.1f}")

        # Size distribution
        from collections import Counter
        size_counts = Counter(grid_sizes)
        print(f"\nMost common grid sizes:")
        for size, count in size_counts.most_common(5):
            print(f"  {size[0]}×{size[1]}: {count} grids")

    print("=" * 70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Kaggle ARC submission format")
    parser.add_argument(
        'submission_file',
        nargs='?',
        default='submission.json',
        help='Path to submission.json (default: submission.json)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (exit code only)'
    )

    args = parser.parse_args()

    submission_path = Path(args.submission_file)

    if not submission_path.exists():
        print(f"❌ File not found: {submission_path}", file=sys.stderr)
        sys.exit(1)

    # Load submission
    try:
        with open(submission_path, 'r') as f:
            submission = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    is_valid, errors = validate_submission(submission, verbose=not args.quiet)

    # Show stats if requested
    if args.stats and not args.quiet:
        analyze_submission_stats(submission)

    # Exit with appropriate code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()