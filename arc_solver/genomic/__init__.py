"""
Genomic (DNA-style) solver for ARC tasks.

This package implements a sequence-based approach that converts grids to 
1D sequences via Hilbert curves, performs alignment-based analysis to
extract mutations, and projects the results back to 2D operations.
"""

from .solver import solve_task_genomic