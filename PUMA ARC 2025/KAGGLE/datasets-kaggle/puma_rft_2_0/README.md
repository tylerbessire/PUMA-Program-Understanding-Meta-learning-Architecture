# PUMA RFT 2.0 - Enhanced ARC Solver

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
