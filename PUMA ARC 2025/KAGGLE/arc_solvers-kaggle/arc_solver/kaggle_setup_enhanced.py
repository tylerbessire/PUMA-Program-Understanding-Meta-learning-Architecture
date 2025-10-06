"""
PUMA ARC Solver - Enhanced Kaggle Setup (v2.0)
Extract and setup the complete ARC solver with fixed submission logic
"""

import os
import sys
import zipfile
import json
from pathlib import Path
from typing import Optional
import numpy as np

def _resolve_local_file(filename: str) -> Path:
    """Resolve local files (zip, checkpoints) across environments."""

    candidate_names = [filename, f"content/{filename}", f"/content/{filename}"]
    candidate_roots = [
        Path('.'),
        Path('content'),
        Path('/content'),
        Path('/kaggle/working'),
    ]

    for name in candidate_names:
        path = Path(name)
        if path.exists():
            return path

    for root in candidate_roots:
        candidate = root / filename
        if candidate.exists():
            return candidate

    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        for dataset_dir in kaggle_input.iterdir():
            candidate = dataset_dir / filename
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Unable to locate required file: {filename}")


def _resolve_data_file(filename: str, required: bool = True) -> Optional[Path]:
    """Resolve data file paths inside extracted package or Kaggle inputs."""

    candidate_dirs = [
        Path('content/data'),
        Path('content/arc_solver/data'),
        Path('data'),
        Path('arc_solver/data'),
        Path('/content/data'),
        Path('/kaggle/working/data'),
    ]

    for base in candidate_dirs:
        candidate = base / filename
        if candidate.exists():
            return candidate

    kaggle_input = Path('/kaggle/input')
    if kaggle_input.exists():
        for dataset_dir in kaggle_input.iterdir():
            direct = dataset_dir / filename
            if direct.exists():
                return direct
            nested = dataset_dir / 'data' / filename
            if nested.exists():
                return nested

    if not required:
        return None

    raise FileNotFoundError(f"Unable to locate required data file: {filename}")


def setup_puma_solver(zip_path="arc_solver_complete.zip"):
    """Extract and setup PUMA ARC solver from zip file."""

    zip_file = _resolve_local_file(zip_path)
    extract_base = Path('content') if Path('content').exists() else Path('.')
    extract_base.mkdir(parents=True, exist_ok=True)

    print(f"📦 Extracting PUMA ARC solver from {zip_file} -> {extract_base.resolve()}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_base)

    for candidate in [extract_base, extract_base / 'arc_solver']:
        candidate_path = candidate.resolve()
        if candidate.exists() and str(candidate_path) not in sys.path:
            sys.path.insert(0, str(candidate_path))

    print("✅ PUMA solver extracted and ready!")
    return extract_base

def get_solver(memory_optimized=True, checkpoint_path="kaggle_checkpoint.json"):
    """Get a configured PUMA solver instance."""
    
    # Memory optimization for Kaggle
    if memory_optimized:
        os.environ['ARC_ENABLE_LOGGING'] = 'false'
    
    # Import after setup
    from arc_solver.solver import ARCSolver
    
    checkpoint_root = Path('content') if Path('content').exists() else Path('.')
    checkpoint_full = Path(checkpoint_path)
    if not checkpoint_full.is_absolute():
        checkpoint_full = (checkpoint_root / checkpoint_full).resolve()
    checkpoint_full.parent.mkdir(parents=True, exist_ok=True)

    # Create solver instance
    solver = ARCSolver(
        use_enhancements=True,
        checkpoint_path=str(checkpoint_full),
        enable_logging=not memory_optimized
    )
    
    # Load any existing checkpoint
    checkpoint_data = solver.load_checkpoint()
    if checkpoint_data:
        completed_count = len(checkpoint_data.get('completed_tasks', []))
        print(f"📋 Loaded checkpoint: {completed_count} tasks completed")
    
    return solver

def validate_grid(grid):
    """Validate that a grid is properly formatted."""
    if not isinstance(grid, list):
        return False
    if len(grid) == 0:
        return False
    return all(isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in grid)

def format_submission_entry_enhanced(result, task_id, debug=False):
    """Enhanced submission formatting with proper Kaggle structure."""
    if debug:
        print(f"  📋 Formatting result for {task_id}")
        print(f"      Raw result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    # Extract attempts from result
    attempt1_raw = result.get('attempt_1', [])
    attempt2_raw = result.get('attempt_2', [])
    
    if debug:
        print(f"      attempt_1 type: {type(attempt1_raw)}, length: {len(attempt1_raw) if hasattr(attempt1_raw, '__len__') else 'N/A'}")
        print(f"      attempt_2 type: {type(attempt2_raw)}, length: {len(attempt2_raw) if hasattr(attempt2_raw, '__len__') else 'N/A'}")
    
    # Ensure we have lists
    if not isinstance(attempt1_raw, list):
        attempt1_raw = [attempt1_raw] if attempt1_raw is not None else []
    if not isinstance(attempt2_raw, list):
        attempt2_raw = [attempt2_raw] if attempt2_raw is not None else []
    
    # Determine number of test cases
    num_test_cases = max(len(attempt1_raw), len(attempt2_raw), 1)
    
    if debug:
        print(f"      Number of test cases detected: {num_test_cases}")
    
    # Create submission entry list
    submission_entry = []
    
    for i in range(num_test_cases):
        # Get grid for this test case
        grid1 = attempt1_raw[i] if i < len(attempt1_raw) else None
        grid2 = attempt2_raw[i] if i < len(attempt2_raw) else None
        
        # Validate and fix grids
        if not validate_grid(grid1):
            grid1 = [[0, 0], [0, 0]]  # Fallback
            if debug:
                print(f"      Test case {i}: Using fallback for attempt_1")
        
        if not validate_grid(grid2):
            grid2 = [[0, 0], [0, 0]]  # Fallback
            if debug:
                print(f"      Test case {i}: Using fallback for attempt_2")
        
        # Create entry for this test case
        entry = {
            "attempt_1": grid1,
            "attempt_2": grid2
        }
        submission_entry.append(entry)
        
        if debug:
            print(f"      Test case {i}: attempt_1 shape {np.array(grid1).shape}, attempt_2 shape {np.array(grid2).shape}")
    
    if debug:
        print(f"      Final submission entry has {len(submission_entry)} test cases")
    
    return submission_entry

def solve_task_safe(solver, task, task_id, timeout=300, debug=False):
    """Safely solve a task with enhanced error handling and submission formatting."""
    import time
    import gc
    
    start_time = time.time()
    
    try:
        # Solve the task
        result = solver.solve_task(task)
        
        # Format for proper submission structure
        submission_entry = format_submission_entry_enhanced(result, task_id, debug=debug)
        
        elapsed = time.time() - start_time
        print(f"    ✅ Solved in {elapsed:.1f}s")
        
        return submission_entry
        
    except Exception as e:
        print(f"    ❌ Error: {str(e)[:100]}...")
        
        # Fallback solution with proper format
        fallback_entry = [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]
        return fallback_entry
    
    finally:
        gc.collect()

def run_kaggle_evaluation_enhanced(test_challenges_path: Optional[str] = None, debug_mode=False):
    """Run complete Kaggle evaluation with enhanced PUMA solver and fixed submission format."""
    
    # Setup solver
    print("🚀 Setting up Enhanced PUMA solver for Kaggle...")
    setup_puma_solver()
    solver = get_solver(memory_optimized=True)
    
    # Load test challenges
    if test_challenges_path:
        test_path = Path(test_challenges_path)
        if not test_path.exists():
            test_path = _resolve_data_file(test_path.name)
    else:
        test_path = _resolve_data_file('arc-agi_test_challenges.json')

    with open(test_path, 'r') as f:
        test_challenges = json.load(f)
    
    submission = {}
    total_tasks = len(test_challenges)
    print(f"📊 Starting enhanced evaluation on {total_tasks} tasks from {test_path}...")
    
    # Debug mode processes fewer tasks with detailed output
    if debug_mode:
        print("🧪 Debug mode: Processing first 3 tasks with detailed output")
        test_challenges = dict(list(test_challenges.items())[:3])
        total_tasks = len(test_challenges)
    
    for i, (task_id, task) in enumerate(test_challenges.items()):
        print(f"[{i+1}/{total_tasks}] Task {task_id}")
        
        # Skip if already in checkpoint
        if hasattr(solver, 'submission_results') and task_id in solver.submission_results:
            # Convert existing result to proper format
            existing_result = solver.submission_results[task_id]
            submission[task_id] = format_submission_entry_enhanced(existing_result, task_id, debug=debug_mode and i == 0)
            print("    📋 Loaded from checkpoint")
            continue
        
        # Solve task with enhanced formatting
        submission_entry = solve_task_safe(solver, task, task_id, debug=debug_mode and i == 0)
        submission[task_id] = submission_entry
        
        # Store in solver for checkpointing
        if hasattr(solver, 'add_submission_result'):
            # Convert back to solver format for checkpointing
            solver_result = {
                'attempt_1': [tc['attempt_1'] for tc in submission_entry],
                'attempt_2': [tc['attempt_2'] for tc in submission_entry]
            }
            solver.add_submission_result(task_id, solver_result)
        
        # Progress update every 10 tasks
        if (i + 1) % 10 == 0:
            print(f"    💾 Checkpoint saved ({i+1}/{total_tasks} completed)")
    
    # Final save
    if hasattr(solver, 'save_checkpoint'):
        solver.save_checkpoint(force=True)
    
    # Validate submission format
    print("\n🔍 Validating submission format...")
    validation_passed = validate_submission_format(submission)
    
    if validation_passed:
        print("✅ Submission format validation passed!")
    else:
        print("⚠️  Submission format validation found issues")
    
    # Save submission
    with open('submission.json', 'w') as f:
        json.dump(submission, f)
    
    print(f"🎯 Enhanced evaluation complete! Submission saved with {len(submission)} tasks.")
    print(f"📊 Format validation: {'PASSED' if validation_passed else 'FAILED'}")
    
    return submission

def validate_submission_format(submission):
    """Validate submission format matches Kaggle requirements."""
    print("🔍 Validating submission format...")
    
    if not isinstance(submission, dict):
        print("❌ Submission must be a dictionary")
        return False
    
    issues = []
    
    # Check first few tasks
    sample_tasks = list(submission.keys())[:5]
    
    for task_id in sample_tasks:
        task_entry = submission[task_id]
        
        # Check if task entry is a list
        if not isinstance(task_entry, list):
            issues.append(f"Task {task_id}: Entry must be a list, got {type(task_entry)}")
            continue
        
        if len(task_entry) == 0:
            issues.append(f"Task {task_id}: Entry list cannot be empty")
            continue
        
        # Check each test case
        for i, test_case in enumerate(task_entry):
            if not isinstance(test_case, dict):
                issues.append(f"Task {task_id} test {i}: Must be dict, got {type(test_case)}")
                continue
            
            if 'attempt_1' not in test_case or 'attempt_2' not in test_case:
                issues.append(f"Task {task_id} test {i}: Missing attempt_1 or attempt_2")
                continue
            
            for attempt_key in ['attempt_1', 'attempt_2']:
                attempt = test_case[attempt_key]
                if not validate_grid(attempt):
                    issues.append(f"Task {task_id} test {i} {attempt_key}: Invalid grid format")
    
    if issues:
        print(f"⚠️  Found {len(issues)} validation issues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
        return False
    else:
        print(f"✅ Validation passed for {len(sample_tasks)} sample tasks")
        return True

# Easy import functions
__all__ = ['setup_puma_solver', 'get_solver', 'solve_task_safe', 'run_kaggle_evaluation_enhanced', 'validate_submission_format']