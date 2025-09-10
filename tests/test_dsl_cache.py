import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from arc_solver.grid import to_array
from arc_solver.dsl import apply_op, _sem_cache


def test_apply_op_uses_semantic_cache():
    _sem_cache.clear()
    grid = to_array([[1, 2], [3, 4]])
    params = {"k": 1}
    out1 = apply_op(grid, "rotate", params)
    cache_size = len(_sem_cache)
    out2 = apply_op(grid, "rotate", params)
    assert np.array_equal(out1, out2)
    assert len(_sem_cache) == cache_size
