import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
from arc_submit import solve_with_budget

class DummySolver:
    def solve_task_two_attempts(self, task):
        raise MemoryError("boom")

    def best_so_far(self, task):
        return [[0]]

def test_memory_error_fallback():
    attempts, meta = solve_with_budget({}, DummySolver())
    assert attempts[0]["output"] == [[0]]
    assert meta["memerror"] is True
    assert meta["timeout"] is False
