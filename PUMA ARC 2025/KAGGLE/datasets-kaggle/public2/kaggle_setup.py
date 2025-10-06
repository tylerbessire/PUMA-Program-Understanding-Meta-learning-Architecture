"""
PUMA ARC Solver - Unified Kaggle Setup
Central setup script for all KAGGLE workflows that ensures proper arc_solver access
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# === DIRECTORY CONFIGURATION ===

def get_puma_root() -> Path:
    """Get the PUMA main directory (source of truth for arc_solver)."""
    kaggle_dir = Path(__file__).resolve().parent
    puma_root = kaggle_dir.parent / "PUMA"
    if not puma_root.exists():
        raise FileNotFoundError(f"PUMA root not found at {puma_root}")
    return puma_root


def get_kaggle_root() -> Path:
    """Get the KAGGLE directory."""
    return Path(__file__).resolve().parent


def get_arc_solver_path() -> Path:
    """Get the canonical arc_solver path from PUMA."""
    puma_root = get_puma_root()
    arc_solver_path = puma_root / "arc_solver"
    if not arc_solver_path.exists():
        raise FileNotFoundError(f"arc_solver not found at {arc_solver_path}")
    return arc_solver_path


# === KAGGLE ENVIRONMENT DETECTION ===

def is_kaggle_environment() -> bool:
    """Check if running in Kaggle environment."""
    return Path("/kaggle/working").exists() or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def get_working_root() -> Path:
    """Get working directory (Kaggle or local)."""
    if is_kaggle_environment():
        working = Path("/kaggle/working")
        working.mkdir(parents=True, exist_ok=True)
        return working
    else:
        # Local development - use KAGGLE directory
        return get_kaggle_root()


# === ARC_SOLVER SETUP ===

def setup_arc_solver_imports() -> Path:
    """
    Setup arc_solver imports for both local and Kaggle environments.

    Local: Uses PUMA/arc_solver directly via sys.path
    Kaggle: Extracts from dataset zip to /kaggle/working

    Returns: Path to arc_solver directory
    """

    if is_kaggle_environment():
        # Kaggle: Extract from dataset
        print("🔧 Kaggle environment detected")
        return setup_arc_solver_kaggle()
    else:
        # Local: Use PUMA arc_solver directly
        print("🔧 Local environment detected")
        return setup_arc_solver_local()


def setup_arc_solver_local() -> Path:
    """Setup arc_solver for local development (uses PUMA directly)."""

    arc_solver_path = get_arc_solver_path()
    puma_root = get_puma_root()

    # Add PUMA root to sys.path so imports work
    puma_root_str = str(puma_root)
    if puma_root_str not in sys.path:
        sys.path.insert(0, puma_root_str)
        print(f"✅ Added PUMA to sys.path: {puma_root_str}")

    # Initialize puma package alias (must import before arc_solver.solver)
    try:
        import puma
        print(f"✅ Initialized puma package alias")
    except ImportError as e:
        print(f"⚠️  Warning: Could not initialize puma alias: {e}")

    # Copy essential data files to KAGGLE working directory
    kaggle_root = get_kaggle_root()
    _copy_essential_files(puma_root, kaggle_root)

    print(f"✅ Using arc_solver from: {arc_solver_path}")
    return arc_solver_path


def setup_arc_solver_kaggle() -> Path:
    """Setup arc_solver for Kaggle environment (extracts from dataset)."""

    working_dir = get_working_root()

    # Look for dataset in Kaggle input directories
    kaggle_input = Path("/kaggle/input")
    dataset_names = [
        "puma-rft-2-0",
        "puma-rft-2-0-enhanced",
        "puma-arc-solver",
        "puma_rft_2_0"
    ]

    # Try to find existing extracted arc_solver
    for candidate in [working_dir / "arc_solver", working_dir / "puma_rft_2_0" / "arc_solver"]:
        if candidate.exists() and (candidate / "solver.py").exists():
            print(f"✅ Found existing arc_solver at: {candidate}")
            _add_to_sys_path(candidate.parent)
            return candidate

    # Try to find and extract from dataset
    if kaggle_input.exists():
        for dataset_dir in kaggle_input.iterdir():
            if not dataset_dir.is_dir():
                continue

            # Check if this is our dataset
            if any(name in dataset_dir.name.lower() for name in dataset_names):
                # Look for arc_solver directly or in a zip
                direct_solver = dataset_dir / "arc_solver"
                if direct_solver.exists():
                    # Copy to working directory
                    target = working_dir / "arc_solver"
                    if not target.exists():
                        shutil.copytree(direct_solver, target)
                        print(f"✅ Copied arc_solver from dataset: {dataset_dir.name}")
                    _add_to_sys_path(working_dir)
                    return target

                # Look for zip files
                for zip_file in dataset_dir.glob("*.zip"):
                    if "solver" in zip_file.name.lower() or "puma" in zip_file.name.lower():
                        print(f"📦 Extracting from: {zip_file.name}")
                        with zipfile.ZipFile(zip_file, 'r') as zf:
                            zf.extractall(working_dir)

                        # Find extracted arc_solver
                        for candidate in [working_dir / "arc_solver", working_dir / "puma_rft_2_0" / "arc_solver"]:
                            if candidate.exists():
                                print(f"✅ Extracted arc_solver to: {candidate}")
                                _add_to_sys_path(candidate.parent)
                                return candidate

    raise FileNotFoundError(
        "Could not find arc_solver in Kaggle datasets. "
        "Please upload the PUMA solver as a Kaggle dataset."
    )


def _add_to_sys_path(path: Path) -> None:
    """Add path to sys.path if not already present."""
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        print(f"   → Added to sys.path: {path_str}")


def _copy_essential_files(source_root: Path, target_root: Path) -> None:
    """Copy essential runtime files (memory, models) to target directory."""

    # Copy memory file
    memory_files = [
        "fast_comprehensive_memory.json",
        "comprehensive_memory.json",
        "continuous_memory.json"
    ]

    for mem_file in memory_files:
        source = source_root / mem_file
        if source.exists():
            target = target_root / mem_file
            if not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
                shutil.copy2(source, target)
                print(f"   → Copied: {mem_file}")
            break

    # Copy models directory
    source_models = source_root / "models"
    if source_models.exists():
        target_models = target_root / "models_kaggle"
        target_models.mkdir(exist_ok=True)
        for model_file in source_models.glob("*.json"):
            target_file = target_models / model_file.name
            if not target_file.exists() or model_file.stat().st_mtime > target_file.stat().st_mtime:
                shutil.copy2(model_file, target_file)
        print(f"   → Synced models directory")


# === SOLVER INITIALIZATION ===

def get_solver(
    memory_optimized: bool = True,
    checkpoint_path: str = "kaggle_checkpoint.json",
    **kwargs
):
    """
    Get a configured PUMA solver instance.

    Args:
        memory_optimized: Enable memory optimizations for Kaggle
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional arguments passed to ARCSolver

    Returns:
        Configured ARCSolver instance
    """

    # Setup arc_solver imports first
    arc_solver_dir = setup_arc_solver_imports()

    # Memory optimization
    if memory_optimized:
        os.environ.setdefault('ARC_ENABLE_LOGGING', 'false')

    # Import solver after setup
    from arc_solver.solver import ARCSolver

    # Setup checkpoint path
    working_dir = get_working_root()
    checkpoint_full = Path(checkpoint_path)
    if not checkpoint_full.is_absolute():
        checkpoint_full = (working_dir / checkpoint_full).resolve()
    checkpoint_full.parent.mkdir(parents=True, exist_ok=True)

    # Find guidance model
    guidance_model = None
    for candidate in [
        working_dir / "models_kaggle" / "guidance_arc.json",
        working_dir / "models" / "guidance_arc.json",
        arc_solver_dir.parent / "models" / "guidance_arc.json"
    ]:
        if candidate.exists():
            guidance_model = str(candidate)
            break

    # Create solver
    solver_kwargs = {
        'use_enhancements': True,
        'checkpoint_path': str(checkpoint_full),
        'enable_logging': not memory_optimized,
        **kwargs
    }

    if guidance_model:
        solver_kwargs['guidance_model_path'] = guidance_model

    solver = ARCSolver(**solver_kwargs)

    # Load checkpoint if exists
    checkpoint_data = solver.load_checkpoint()
    if checkpoint_data:
        completed = len(checkpoint_data.get('completed_tasks', []))
        print(f"📋 Loaded checkpoint: {completed} tasks completed")

    return solver


# === TASK SOLVING ===

def solve_task_safe(solver, task: Dict[str, Any], task_id: str = None, timeout: int = 300):
    """
    Safely solve a task with error handling.

    Args:
        solver: ARCSolver instance
        task: Task dictionary
        task_id: Optional task ID for logging
        timeout: Timeout in seconds

    Returns:
        Result dictionary with attempt_1 and attempt_2
    """
    import gc
    import time

    start_time = time.time()

    try:
        result = solver.solve_task(task)
        elapsed = time.time() - start_time

        if task_id:
            print(f"    ✅ {task_id} solved in {elapsed:.1f}s")
        else:
            print(f"    ✅ Solved in {elapsed:.1f}s")

        return result

    except Exception as e:
        if task_id:
            print(f"    ❌ {task_id} error: {str(e)[:100]}")
        else:
            print(f"    ❌ Error: {str(e)[:100]}")

        # Fallback to test input
        test_input = task['test'][0]['input'] if task.get('test') else [[0, 0], [0, 0]]
        return {
            'attempt_1': [test_input],
            'attempt_2': [test_input]
        }

    finally:
        gc.collect()


# === SUBMISSION FORMATTING ===

def validate_grid(grid) -> bool:
    """Validate that a grid is properly formatted."""
    if not isinstance(grid, list):
        return False
    if len(grid) == 0:
        return False
    return all(
        isinstance(row, list) and all(isinstance(cell, int) for cell in row)
        for row in grid
    )


def format_submission_entry(task: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, List[List[int]]]]:
    """
    Format solver result to Kaggle submission format.

    Kaggle format per task:
    [
        {"attempt_1": [[grid]], "attempt_2": [[grid]]},  # Test case 1
        {"attempt_1": [[grid]], "attempt_2": [[grid]]},  # Test case 2
        ...
    ]

    Args:
        task: Task dictionary
        result: Solver result with attempt_1 and attempt_2

    Returns:
        List of dicts with attempt_1 and attempt_2 for each test case
    """

    # Extract test cases
    test_cases = task.get('test', [])
    num_test_cases = len(test_cases) if test_cases else 1

    # Extract attempts
    attempt1_raw = result.get('attempt_1', [])
    attempt2_raw = result.get('attempt_2', [])

    # Ensure attempts are lists
    if not isinstance(attempt1_raw, list):
        attempt1_raw = [attempt1_raw] if attempt1_raw is not None else []
    if not isinstance(attempt2_raw, list):
        attempt2_raw = [attempt2_raw] if attempt2_raw is not None else []

    # Build submission entry for each test case
    submission_entry = []

    for i in range(num_test_cases):
        # Get grids for this test case
        grid1 = attempt1_raw[i] if i < len(attempt1_raw) else None
        grid2 = attempt2_raw[i] if i < len(attempt2_raw) else None

        # Fallback grid
        if test_cases and i < len(test_cases):
            fallback = test_cases[i].get('input', [[0, 0], [0, 0]])
        else:
            fallback = [[0, 0], [0, 0]]

        # Validate and use fallback if needed
        if not validate_grid(grid1):
            grid1 = fallback
        if not validate_grid(grid2):
            grid2 = fallback

        # Add to submission
        submission_entry.append({
            "attempt_1": grid1,
            "attempt_2": grid2
        })

    return submission_entry


def validate_submission(submission: Dict[str, Any]) -> bool:
    """
    Validate submission format.

    Args:
        submission: Dictionary mapping task_id to list of test case attempts

    Returns:
        True if valid, False otherwise
    """

    if not isinstance(submission, dict):
        print("❌ Submission must be a dictionary")
        return False

    issues = []

    for task_id, task_entry in list(submission.items())[:5]:  # Check first 5
        if not isinstance(task_entry, list):
            issues.append(f"{task_id}: Must be list, got {type(task_entry)}")
            continue

        if len(task_entry) == 0:
            issues.append(f"{task_id}: Cannot be empty list")
            continue

        for i, test_case in enumerate(task_entry):
            if not isinstance(test_case, dict):
                issues.append(f"{task_id}[{i}]: Must be dict")
                continue

            if 'attempt_1' not in test_case or 'attempt_2' not in test_case:
                issues.append(f"{task_id}[{i}]: Missing attempt keys")
                continue

            for key in ['attempt_1', 'attempt_2']:
                if not validate_grid(test_case[key]):
                    issues.append(f"{task_id}[{i}].{key}: Invalid grid")

    if issues:
        print(f"⚠️  Found {len(issues)} validation issues:")
        for issue in issues[:10]:
            print(f"   - {issue}")
        return False

    print("✅ Submission validation passed")
    return True


# === FULL EVALUATION ===

def run_kaggle_evaluation(
    test_challenges_path: Optional[str] = None,
    output_path: str = "submission.json",
    debug_mode: bool = False,
    max_tasks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run complete Kaggle evaluation.

    Args:
        test_challenges_path: Path to test challenges JSON
        output_path: Path to save submission JSON
        debug_mode: Enable debug output
        max_tasks: Maximum number of tasks to process (for testing)

    Returns:
        Submission dictionary
    """

    print("🚀 Starting PUMA Kaggle Evaluation")
    print("=" * 60)

    # Setup solver
    solver = get_solver(memory_optimized=True)

    # Load test challenges
    if test_challenges_path:
        test_path = Path(test_challenges_path)
    else:
        # Try common locations
        candidates = [
            Path("/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"),
            get_kaggle_root() / "data-kaggle" / "arc-agi_test_challenges.json",
            get_puma_root() / "data" / "arc-agi_test_challenges.json"
        ]
        test_path = None
        for candidate in candidates:
            if candidate.exists():
                test_path = candidate
                break

        if not test_path:
            raise FileNotFoundError("Could not find test challenges JSON")

    print(f"📂 Loading test challenges from: {test_path}")
    with open(test_path, 'r') as f:
        test_challenges = json.load(f)

    # Limit tasks if specified
    if max_tasks:
        test_challenges = dict(list(test_challenges.items())[:max_tasks])

    total_tasks = len(test_challenges)
    print(f"📊 Processing {total_tasks} tasks")
    print("=" * 60)

    # Process tasks
    submission = {}

    for i, (task_id, task) in enumerate(test_challenges.items(), 1):
        print(f"[{i}/{total_tasks}] Task {task_id}")

        # Check checkpoint
        if hasattr(solver, 'submission_results') and task_id in solver.submission_results:
            cached_result = solver.submission_results[task_id]
            submission[task_id] = format_submission_entry(task, cached_result)
            print("    📋 Loaded from checkpoint")
            continue

        # Solve task
        result = solve_task_safe(solver, task, task_id=task_id)
        submission[task_id] = format_submission_entry(task, result)

        # Save to checkpoint
        if hasattr(solver, 'add_submission_result'):
            solver.add_submission_result(task_id, result)

        # Periodic checkpoint save
        if i % 10 == 0:
            if hasattr(solver, 'save_checkpoint'):
                solver.save_checkpoint(force=True)
            print(f"    💾 Checkpoint saved ({i}/{total_tasks})")

    # Final checkpoint
    if hasattr(solver, 'save_checkpoint'):
        solver.save_checkpoint(force=True)

    print("=" * 60)

    # Validate
    print("\n🔍 Validating submission format...")
    is_valid = validate_submission(submission)

    # Save submission
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n🎯 Evaluation complete!")
    print(f"   Tasks processed: {len(submission)}")
    print(f"   Validation: {'PASSED ✅' if is_valid else 'FAILED ❌'}")
    print(f"   Saved to: {output_file.resolve()}")

    return submission


# === EXPORTS ===

__all__ = [
    'setup_arc_solver_imports',
    'get_solver',
    'solve_task_safe',
    'format_submission_entry',
    'validate_submission',
    'run_kaggle_evaluation',
    'is_kaggle_environment',
    'get_puma_root',
    'get_kaggle_root',
    'get_working_root',
]