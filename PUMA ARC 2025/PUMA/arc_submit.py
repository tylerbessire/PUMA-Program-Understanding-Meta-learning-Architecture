"""Hardened command-line entry point for generating ARC Prize submissions.

This script enforces deterministic execution, applies strict resource budgets
for each task, and guarantees exactly two diverse attempts per test input. It
is designed to run within the ARC Prize Kaggle environment where internet
access is disabled and runtime is tightly constrained.
"""

from __future__ import annotations

import gc
import json
import os
import random
import resource
import signal
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from arc_solver.solver import ARCSolver
from arc_solver.io_utils import load_rerun_json, save_submission


# ---------------------------------------------------------------------------
# Determinism and thread caps
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
random.seed(0)
np.random.seed(0)


HARD_TIMEOUT_SEC: float = float(os.environ.get("ARC_TIMEOUT_SEC", "30"))
"""Per-task hard timeout in seconds."""

MEM_SOFT_LIMIT_MB: int = int(os.environ.get("ARC_MEM_MB", "1024"))
"""Soft memory limit in megabytes for each task."""


class Timeout(Exception):
    """Raised when a task exceeds the allocated time budget."""


def _alarm(_signum: int, _frame: Any) -> None:
    """Signal handler used to abort execution on timeout."""
    raise Timeout()


def _set_mem_limit() -> None:
    """Apply a soft memory limit for the current process."""
    try:
        soft_bytes = MEM_SOFT_LIMIT_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft_bytes, resource.RLIM_INFINITY))
    except Exception:
        # Some platforms (e.g., Windows) may not support RLIMIT_AS.
        pass


def solve_with_budget(task: Dict[str, Any], solver: ARCSolver) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Solve a task under strict time and memory budgets.

    Args:
        task: ARC task specification with ``train`` and ``test`` examples.
        solver: Configured :class:`ARCSolver` instance.

    Returns:
        A tuple ``(attempts, metadata)`` where ``attempts`` is a list with two
        dictionaries of the form ``{"output": grid}`` and ``metadata``
        contains diagnostic information such as elapsed time and timeout flag.
    """
    # [S:ALG v1] fallback=best_so_far memlimit=soft pass
    _set_mem_limit()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(HARD_TIMEOUT_SEC))
    start = time.time()
    try:
        attempt1, attempt2 = solver.solve_task_two_attempts(task)
        elapsed = time.time() - start
        return [
            {"output": attempt1},
            {"output": attempt2},
        ], {"elapsed": elapsed, "timeout": False, "memerror": False}
    except Timeout:
        best = solver.best_so_far(task)
        elapsed = time.time() - start
        return [
            {"output": best},
            {"output": best},
        ], {"elapsed": elapsed, "timeout": True, "memerror": False}
    except MemoryError:
        best = solver.best_so_far(task)
        elapsed = time.time() - start
        return [
            {"output": best},
            {"output": best},
        ], {"elapsed": elapsed, "timeout": False, "memerror": True}
    finally:
        signal.alarm(0)


def main() -> None:
    """Entry point for Kaggle submission generation."""
    data = load_rerun_json()
    solver = ARCSolver(use_enhancements=True)
    solutions: Dict[str, Dict[str, List[List[int]]]] = {}

    mem_error_count = 0
    for task_id, task in data.items():
        attempts, meta = solve_with_budget(task, solver)
        solutions[task_id] = {
            "attempt_1": attempts[0]["output"],
            "attempt_2": attempts[1]["output"],
        }
        if meta.get("memerror"):
            mem_error_count += 1
        print(
            f"[task {task_id}] t={meta['elapsed']:.2f}s timeout={meta['timeout']} memerror={meta['memerror']}",
            file=sys.stderr,
        )
        gc.collect()

    path = save_submission(solutions, "submission.json")
    print(f"Saved {path} with {len(solutions)} tasks. memory_errors={mem_error_count}")


if __name__ == "__main__":
    main()

