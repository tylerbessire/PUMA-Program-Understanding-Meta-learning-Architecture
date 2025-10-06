#!/usr/bin/env python3
"""
PUMA to KAGGLE Sync Script

Synchronizes essential files from PUMA to KAGGLE for competition submission.
Ensures KAGGLE has latest solver code, models, and data.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple
import json


def get_puma_root() -> Path:
    """Get PUMA root directory."""
    script_dir = Path(__file__).resolve().parent
    puma_root = script_dir.parent / "PUMA"
    if not puma_root.exists():
        raise FileNotFoundError(f"PUMA directory not found at {puma_root}")
    return puma_root


def get_kaggle_root() -> Path:
    """Get KAGGLE root directory."""
    return Path(__file__).resolve().parent


def sync_arc_solver(dry_run: bool = False) -> List[str]:
    """
    Sync arc_solver from PUMA to KAGGLE.

    Creates a clean copy in datasets-kaggle/puma_rft_2_0/arc_solver
    for packaging and upload to Kaggle.
    """
    puma_root = get_puma_root()
    kaggle_root = get_kaggle_root()

    source = puma_root / "arc_solver"
    target = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / "arc_solver"

    if not source.exists():
        raise FileNotFoundError(f"Source arc_solver not found: {source}")

    actions = []

    if target.exists():
        actions.append(f"REMOVE: {target}")
        if not dry_run:
            shutil.rmtree(target)

    actions.append(f"COPY: {source} -> {target}")
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store'))

    return actions


def sync_memory_files(dry_run: bool = False) -> List[str]:
    """Sync memory and checkpoint files."""
    puma_root = get_puma_root()
    kaggle_root = get_kaggle_root()

    actions = []

    memory_files = [
        "fast_comprehensive_memory.json",
        "comprehensive_memory.json",
        "continuous_memory.json",
        "checkpoint.json"
    ]

    for filename in memory_files:
        source = puma_root / filename
        if source.exists():
            target = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / filename
            actions.append(f"COPY: {filename}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    return actions


def sync_models(dry_run: bool = False) -> List[str]:
    """Sync model files."""
    puma_root = get_puma_root()
    kaggle_root = get_kaggle_root()

    source_models = puma_root / "models"
    target_models = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / "models"

    if not source_models.exists():
        return ["SKIP: models directory not found"]

    actions = []

    if target_models.exists():
        actions.append(f"REMOVE: {target_models}")
        if not dry_run:
            shutil.rmtree(target_models)

    actions.append(f"COPY: models/ -> {target_models}")
    if not dry_run:
        target_models.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_models, target_models, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))

    return actions


def sync_data(dry_run: bool = False) -> List[str]:
    """Sync data files (test challenges, etc)."""
    puma_root = get_puma_root()
    kaggle_root = get_kaggle_root()

    actions = []

    data_files = [
        "arc-agi_test_challenges.json",
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json"
    ]

    source_data = puma_root / "data"
    target_data = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / "data"

    if not source_data.exists():
        return ["SKIP: data directory not found"]

    for filename in data_files:
        source = source_data / filename
        if source.exists():
            target = target_data / filename
            actions.append(f"COPY: data/{filename}")
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    return actions


def create_dataset_metadata(dry_run: bool = False) -> List[str]:
    """Create Kaggle dataset metadata."""
    kaggle_root = get_kaggle_root()

    metadata = {
        "title": "PUMA RFT 2.0 - Enhanced ARC Solver",
        "id": "tylerbessire/puma-rft-2-0",
        "licenses": [{"name": "MIT"}],
        "keywords": ["arc", "abstract reasoning", "puma", "rft"]
    }

    target = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / "dataset-metadata.json"

    actions = [f"CREATE: dataset-metadata.json"]
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w') as f:
            json.dump(metadata, f, indent=2)

    return actions


def create_readme(dry_run: bool = False) -> List[str]:
    """Create README for Kaggle dataset."""
    kaggle_root = get_kaggle_root()

    readme_content = """# PUMA RFT 2.0 - Enhanced ARC Solver

This dataset contains the complete PUMA (Pattern Understanding and Manipulation Architecture)
solver for the ARC (Abstraction and Reasoning Corpus) challenge.

## Contents

- `arc_solver/` - Complete solver package with all modules
- `models/` - Pre-trained guidance models
- `data/` - Test challenges and evaluation data
- `kaggle_setup.py` - Setup utilities for Kaggle environment
- `fast_comprehensive_memory.json` - Pre-built episodic memory

## Usage in Kaggle Notebook

```python
# Import setup utilities
from kaggle_setup import get_solver, run_kaggle_evaluation

# Get configured solver
solver = get_solver(memory_optimized=True)

# Run full evaluation
submission = run_kaggle_evaluation()
```

## Features

- Dynamic shape detection and governance
- Human-grade spatial reasoning with RFT
- Comprehensive memory and episodic retrieval
- Neural guidance and hypothesis generation
- Robust error handling and checkpointing

## Version

Version: 2.0
Updated: 2025-09-30
"""

    target = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / "README.md"

    actions = [f"CREATE: README.md"]
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, 'w') as f:
            f.write(readme_content)

    return actions


def verify_kaggle_setup(dry_run: bool = False) -> List[str]:
    """Verify kaggle_setup.py is in the dataset."""
    kaggle_root = get_kaggle_root()

    source = kaggle_root / "kaggle_setup.py"
    target = kaggle_root / "datasets-kaggle" / "puma_rft_2_0" / "kaggle_setup.py"

    if not source.exists():
        return ["ERROR: kaggle_setup.py not found in KAGGLE root"]

    actions = [f"COPY: kaggle_setup.py"]
    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return actions


def full_sync(dry_run: bool = False, verbose: bool = True) -> Tuple[int, int]:
    """
    Perform full sync from PUMA to KAGGLE.

    Returns: (success_count, error_count)
    """

    if verbose:
        print("=" * 70)
        print("PUMA → KAGGLE SYNC")
        print("=" * 70)
        if dry_run:
            print("DRY RUN MODE - No changes will be made")
            print("-" * 70)

    operations = [
        ("Syncing arc_solver", sync_arc_solver),
        ("Syncing memory files", sync_memory_files),
        ("Syncing models", sync_models),
        ("Syncing data", sync_data),
        ("Verifying kaggle_setup.py", verify_kaggle_setup),
        ("Creating dataset metadata", create_dataset_metadata),
        ("Creating README", create_readme),
    ]

    success_count = 0
    error_count = 0

    for description, operation in operations:
        if verbose:
            print(f"\n{description}...")

        try:
            actions = operation(dry_run=dry_run)
            for action in actions:
                if verbose:
                    print(f"  {action}")
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            error_count += 1

    if verbose:
        print("\n" + "=" * 70)
        print(f"SYNC COMPLETE")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")
        if dry_run:
            print(f"  (Dry run - no changes made)")
        print("=" * 70)

    return success_count, error_count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync PUMA solver to KAGGLE for competition submission"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be synced without making changes"
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help="Minimal output"
    )

    args = parser.parse_args()

    try:
        success, errors = full_sync(
            dry_run=args.dry_run,
            verbose=not args.quiet
        )

        sys.exit(0 if errors == 0 else 1)

    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()