"""
Object-Agentic solver for ARC tasks.

This package implements an agent-based approach where individual agents
reason about objects in the grid and propose transformations, coordinated
by a referee using beam search and MDL principles.
"""

from .agentic_solver import solve_task_agentic
from .ops import Op, execute_program_on_grid