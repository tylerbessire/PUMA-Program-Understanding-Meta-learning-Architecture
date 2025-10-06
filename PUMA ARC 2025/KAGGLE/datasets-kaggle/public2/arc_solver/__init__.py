"""ARC Solver package entry point with lazy solver import."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .grid import Array
from .io_utils import load_rerun_json, save_submission

__all__ = ["ARCSolver", "load_rerun_json", "save_submission", "Array"]

if TYPE_CHECKING:  # pragma: no cover
    from .solver import ARCSolver


def __getattr__(name: str):
    if name == "ARCSolver":
        from .solver import ARCSolver

        return ARCSolver
    raise AttributeError(f"module 'arc_solver' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(__all__)
