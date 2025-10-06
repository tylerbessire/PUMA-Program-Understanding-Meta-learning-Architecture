"""ARC Solver Package.

This package exposes the high-level :class:`ARCSolver` alongside common
utilities for interacting with ARC datasets. The solver integrates neural
guidance, episodic retrieval and test-time training into a cohesive system.
"""

from .solver import ARCSolver
from .io_utils import load_rerun_json, save_submission
from .grid import Array

__all__ = ["ARCSolver", "load_rerun_json", "save_submission", "Array"]
