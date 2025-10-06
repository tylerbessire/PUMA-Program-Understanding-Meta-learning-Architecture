#!/usr/bin/env python3
"""
Test KAGGLE Setup

Verifies that kaggle_setup.py works correctly in local environment.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that kaggle_setup imports work."""
    print("=" * 70)
    print("TEST 1: Imports")
    print("=" * 70)

    try:
        from kaggle_setup import (
            get_solver,
            solve_task_safe,
            format_submission_entry,
            validate_submission,
            is_kaggle_environment,
            get_puma_root,
            get_kaggle_root,
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_environment_detection():
    """Test environment detection."""
    print("\n" + "=" * 70)
    print("TEST 2: Environment Detection")
    print("=" * 70)

    from kaggle_setup import is_kaggle_environment, get_puma_root, get_kaggle_root

    is_kaggle = is_kaggle_environment()
    print(f"Is Kaggle: {is_kaggle}")

    if is_kaggle:
        print("⚠️  Running in Kaggle environment")
        return True

    try:
        puma_root = get_puma_root()
        print(f"✅ PUMA root: {puma_root}")

        kaggle_root = get_kaggle_root()
        print(f"✅ KAGGLE root: {kaggle_root}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_arc_solver_setup():
    """Test arc_solver setup."""
    print("\n" + "=" * 70)
    print("TEST 3: Arc Solver Setup")
    print("=" * 70)

    try:
        from kaggle_setup import setup_arc_solver_imports

        arc_solver_path = setup_arc_solver_imports()
        print(f"✅ Arc solver path: {arc_solver_path}")

        # Test import
        from arc_solver.solver import ARCSolver
        print("✅ ARCSolver import successful")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_solver_creation():
    """Test solver creation."""
    print("\n" + "=" * 70)
    print("TEST 4: Solver Creation")
    print("=" * 70)

    try:
        from kaggle_setup import get_solver

        print("Creating solver...")
        solver = get_solver(memory_optimized=True)
        print(f"✅ Solver created: {type(solver).__name__}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_submission_formatting():
    """Test submission formatting."""
    print("\n" + "=" * 70)
    print("TEST 5: Submission Formatting")
    print("=" * 70)

    from kaggle_setup import format_submission_entry, validate_submission

    # Create dummy task and result
    task = {
        'train': [],
        'test': [
            {'input': [[0, 1], [2, 3]]}
        ]
    }

    result = {
        'attempt_1': [[[0, 1], [2, 3]]],
        'attempt_2': [[[4, 5], [6, 7]]]
    }

    try:
        formatted = format_submission_entry(task, result)
        print(f"✅ Formatted result: {formatted}")

        # Test validation
        submission = {'test_task': formatted}
        validation_result = validate_submission(submission)
        is_valid = validation_result[0] if isinstance(validation_result, tuple) else validation_result

        if is_valid:
            print("✅ Validation passed")
            return True
        else:
            print("❌ Validation failed")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_existence():
    """Test that required files exist."""
    print("\n" + "=" * 70)
    print("TEST 6: Required Files")
    print("=" * 70)

    kaggle_root = Path(__file__).parent

    required_files = [
        'kaggle_setup.py',
        'sync_puma.py',
        'validate_submission.py',
        'README.md',
    ]

    all_exist = True
    for filename in required_files:
        filepath = kaggle_root / filename
        if filepath.exists():
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - MISSING")
            all_exist = False

    return all_exist


def run_all_tests():
    """Run all tests."""
    print("KAGGLE SETUP TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        ("Imports", test_imports),
        ("Environment Detection", test_environment_detection),
        ("Arc Solver Setup", test_arc_solver_setup),
        ("Solver Creation", test_solver_creation),
        ("Submission Formatting", test_submission_formatting),
        ("Required Files", test_file_existence),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, p in results if p)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)