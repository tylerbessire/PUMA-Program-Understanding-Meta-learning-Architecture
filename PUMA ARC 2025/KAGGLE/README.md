# KAGGLE Workspace - PUMA ARC Solver

This directory contains the **KAGGLE workspace** for the PUMA ARC solver. It manages datasets, notebooks, and submission files for the Kaggle ARC competition.

## 🎯 Purpose

The KAGGLE folder serves as a **secondary workspace** that:
- Prepares datasets for Kaggle upload
- Stores Kaggle notebooks for competition submission
- Manages submission files and validation
- **SHARES** the arc_solver from the main PUMA directory

## 📁 Directory Structure

```
KAGGLE/
├── kaggle_setup.py              # 🔧 Unified setup script (MAIN ENTRY POINT)
├── sync_puma.py                 # 📦 Syncs PUMA → KAGGLE
├── validate_submission.py       # ✅ Validates submission.json format
├── README.md                    # 📖 This file
│
├── datasets-kaggle/             # 📊 Datasets for Kaggle upload
│   └── puma_rft_2_0/           # Main dataset package
│       ├── arc_solver/         # (Synced from PUMA)
│       ├── kaggle_setup.py     # (Copy of root setup)
│       ├── models/             # Pre-trained models
│       ├── data/               # Test challenges
│       ├── fast_comprehensive_memory.json
│       ├── dataset-metadata.json
│       └── README.md
│
├── notebooks/                   # 📓 Kaggle notebooks
│   ├── PUMA-Latest-9-28-2025.ipynb
│   ├── PUMA-Full-Submission.ipynb
│   └── ...
│
├── submission_kaggle/           # 🎯 Generated submission files
│   ├── submission-latest.json
│   └── submission-old.json
│
├── models_kaggle/               # 🧠 Models (synced from PUMA)
├── data-kaggle/                 # 📂 Data files
└── logs/                        # 📝 Execution logs
```

## 🔄 How PUMA and KAGGLE Share arc_solver

### The Problem
- PUMA has the **source of truth** arc_solver at `/Users/tylerbessire/PUMA ARC 2025/PUMA/arc_solver/`
- KAGGLE needs to access it for local development and package it for Kaggle upload
- Multiple copies were causing import issues and sync problems

### The Solution
The new `kaggle_setup.py` handles both environments intelligently:

#### Local Development (Mac)
```python
from kaggle_setup import get_solver

# Automatically uses PUMA/arc_solver via sys.path
# No copies, no symlinks, just direct import
solver = get_solver()
```

**How it works:**
1. Detects local environment (not Kaggle)
2. Adds `PUMA/` to `sys.path`
3. Imports directly from `PUMA/arc_solver/`
4. Copies only data files (memory, models) to KAGGLE

#### Kaggle Environment
```python
from kaggle_setup import get_solver

# Automatically extracts from uploaded dataset
solver = get_solver()
```

**How it works:**
1. Detects Kaggle environment (`/kaggle/working` exists)
2. Looks for uploaded dataset in `/kaggle/input/`
3. Extracts arc_solver to `/kaggle/working/`
4. Sets up imports and paths

## 🚀 Workflow

### 1. Development (Local)

Work in PUMA directory as normal:
```bash
cd "/Users/tylerbessire/PUMA ARC 2025/PUMA"
# Edit arc_solver files
python -m arc_solver.solver
```

### 2. Sync to KAGGLE

When ready to prepare for Kaggle submission:
```bash
cd "/Users/tylerbessire/PUMA ARC 2025/KAGGLE"
python sync_puma.py
```

This syncs:
- `PUMA/arc_solver/` → `KAGGLE/datasets-kaggle/puma_rft_2_0/arc_solver/`
- Memory files, models, data files
- Creates dataset metadata and README

**Dry run first:**
```bash
python sync_puma.py --dry-run
```

### 3. Test Locally

Test notebooks using the synced setup:
```bash
cd "/Users/tylerbessire/PUMA ARC 2025/KAGGLE"
jupyter notebook notebooks/PUMA-Latest-9-28-2025.ipynb
```

The notebook will:
- Use `kaggle_setup.py`
- Import from `PUMA/arc_solver/` (since local)
- Generate `submission.json`

### 4. Validate Submission

```bash
python validate_submission.py submission_kaggle/submission-latest.json --stats
```

Expected output:
```
✅ VALIDATION PASSED - Submission format is correct!
Tasks checked: 240
Tasks with errors: 0
```

### 5. Package for Kaggle

Create a zip of the dataset:
```bash
cd datasets-kaggle
zip -r puma_rft_2_0.zip puma_rft_2_0/ -x "*.pyc" -x "*__pycache__*" -x ".DS_Store"
```

### 6. Upload to Kaggle

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click "New Dataset"
3. Upload `puma_rft_2_0.zip`
4. Title: "PUMA RFT 2.0 - Enhanced ARC Solver"
5. Make it public

### 7. Create Kaggle Notebook

Use the notebook template:
```python
# Import from uploaded dataset
from kaggle_setup import get_solver, run_kaggle_evaluation

# Run evaluation (auto-detects Kaggle environment)
submission = run_kaggle_evaluation()
```

## 📝 Key Files Explained

### kaggle_setup.py
**Location:** `/Users/tylerbessire/PUMA ARC 2025/KAGGLE/kaggle_setup.py`

**Purpose:** Unified setup script that handles both local and Kaggle environments.

**Key Functions:**
- `setup_arc_solver_imports()` - Sets up imports for current environment
- `get_solver(**kwargs)` - Returns configured ARCSolver instance
- `solve_task_safe(solver, task)` - Safely solves a task with error handling
- `format_submission_entry(task, result)` - Formats result to Kaggle format
- `validate_submission(submission)` - Validates submission format
- `run_kaggle_evaluation()` - Runs complete evaluation

**Example Usage:**
```python
from kaggle_setup import get_solver, run_kaggle_evaluation

# Quick evaluation
submission = run_kaggle_evaluation()

# Or manual control
solver = get_solver(memory_optimized=True)
result = solver.solve_task(task)
```

### sync_puma.py
**Location:** `/Users/tylerbessire/PUMA ARC 2025/KAGGLE/sync_puma.py`

**Purpose:** Syncs PUMA to KAGGLE for dataset preparation.

**Usage:**
```bash
# Dry run
python sync_puma.py --dry-run

# Full sync
python sync_puma.py

# Quiet mode
python sync_puma.py --quiet
```

**What it syncs:**
- arc_solver directory (removes __pycache__, .pyc files)
- Memory files (fast_comprehensive_memory.json, etc.)
- Models directory
- Data files (test challenges)
- kaggle_setup.py
- Creates dataset-metadata.json and README.md

### validate_submission.py
**Location:** `/Users/tylerbessire/PUMA ARC 2025/KAGGLE/validate_submission.py`

**Purpose:** Validates submission.json format matches Kaggle requirements.

**Usage:**
```bash
# Validate with stats
python validate_submission.py submission.json --stats

# Quiet mode (exit code only)
python validate_submission.py submission.json --quiet
```

**Checks:**
- Submission is a dictionary
- Each task has a list of test cases
- Each test case has attempt_1 and attempt_2
- Each attempt is a valid grid (2D list of integers 0-9)
- All rows in grid have same length

## 🔍 Troubleshooting

### Import Error: "No module named 'arc_solver'"

**Cause:** kaggle_setup.py not used correctly

**Solution:**
```python
# ❌ DON'T DO THIS
from arc_solver.solver import ARCSolver

# ✅ DO THIS
from kaggle_setup import get_solver
solver = get_solver()
```

### "PUMA directory not found"

**Cause:** Script can't find PUMA directory

**Solution:** Ensure directory structure:
```
PUMA ARC 2025/
├── PUMA/
│   └── arc_solver/
└── KAGGLE/
    └── kaggle_setup.py
```

### kaggle_setup.py always missing

**Cause:** Old notebooks looking in wrong location

**Solution:** kaggle_setup.py is now in KAGGLE root. Update notebook:
```python
import sys
from pathlib import Path

# Add KAGGLE root to path
kaggle_root = Path.cwd().parent  # Adjust as needed
sys.path.insert(0, str(kaggle_root))

from kaggle_setup import get_solver
```

### Submission format validation fails

**Cause:** Result not formatted correctly

**Solution:** Always use format_submission_entry:
```python
from kaggle_setup import format_submission_entry

result = solver.solve_task(task)
formatted = format_submission_entry(task, result)
submission[task_id] = formatted
```

## 📊 Submission Format

The correct Kaggle format for each task:
```json
{
  "task_id": [
    {
      "attempt_1": [[0, 1], [2, 3]],
      "attempt_2": [[0, 1], [2, 3]]
    },
    {
      "attempt_1": [[4, 5], [6, 7]],
      "attempt_2": [[4, 5], [6, 7]]
    }
  ]
}
```

**Key points:**
- Task value is a **list** (not dict)
- Each list item is a **dict** with attempt_1 and attempt_2
- Each attempt is a **2D list** (grid)
- All cells are **integers** 0-9

## 🎯 Quick Reference

### Local Development
```bash
# 1. Work in PUMA
cd "/Users/tylerbessire/PUMA ARC 2025/PUMA"
# Edit code...

# 2. Sync to KAGGLE
cd "../KAGGLE"
python sync_puma.py

# 3. Test notebook
jupyter notebook notebooks/PUMA-Latest-9-28-2025.ipynb

# 4. Validate submission
python validate_submission.py submission.json --stats
```

### Kaggle Submission
```bash
# 1. Package dataset
cd datasets-kaggle
zip -r puma_rft_2_0.zip puma_rft_2_0/

# 2. Upload to Kaggle
# (via web interface)

# 3. Create notebook with:
from kaggle_setup import run_kaggle_evaluation
submission = run_kaggle_evaluation()
```

## 📚 Additional Resources

- **PUMA Main README:** `/Users/tylerbessire/PUMA ARC 2025/PUMA/README.md`
- **Kaggle Submission Guide:** `/Users/tylerbessire/PUMA ARC 2025/PUMA/KAGGLE_SUBMISSION.md`
- **Dataset README:** `datasets-kaggle/puma_rft_2_0/README.md`

## 🔧 Configuration

### Environment Variables

The setup respects these environment variables:
- `ARC_ENABLE_LOGGING` - Enable/disable logging (default: false in Kaggle)
- `KAGGLE_KERNEL_RUN_TYPE` - Auto-detected by Kaggle

### Checkpoint Path

Default: `kaggle_checkpoint.json` in working directory

Custom checkpoint:
```python
solver = get_solver(checkpoint_path="/custom/path/checkpoint.json")
```

## ✅ Verification Checklist

Before submitting to Kaggle:
- [ ] Sync completed: `python sync_puma.py`
- [ ] Validation passed: `python validate_submission.py submission.json`
- [ ] Dataset zipped and uploaded to Kaggle
- [ ] Notebook imports `kaggle_setup` correctly
- [ ] Test run completed locally
- [ ] Submission format validated

---

**Last Updated:** 2025-09-30
**Version:** 2.0
**Maintainer:** Tyler Bessire