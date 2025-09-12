# [S:TEST v1] beam_search unit and property tests pass
import numpy as np
from arc_solver.grid import to_array
from arc_solver.beam_search import beam_search
from arc_solver.mcts_search import mcts_search
from arc_solver.dsl import apply_program
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp


def test_beam_search_finds_rotation():
    inp = to_array([[1, 2], [3, 4]])
    out = np.rot90(inp, -1)
    progs, stats = beam_search([(inp, out)], beam_width=5, depth=2)
    assert any(np.array_equal(apply_program(inp, p), out) for p in progs)
    assert stats["nodes_expanded"] > 0
    assert len(progs) <= 5


@given(
    grid=hnp.arrays(dtype=np.int16, shape=(3, 3), elements=st.integers(0, 9)),
    k=st.integers(1, 3),
)
def test_beam_search_rotation_property(grid, k):
    out = np.rot90(grid, -k)
    progs, _ = beam_search([(grid, out)], beam_width=5, depth=1)
    assert any(p == [("rotate", {"k": k})] for p in progs)


def test_beam_search_no_solution():
    a = to_array([[0]])
    b = to_array([[1]])
    progs, _ = beam_search([(a, b)], beam_width=3, depth=1)
    assert progs == []


def test_mcts_search_finds_rotation():
    inp = to_array([[1, 2], [3, 4]])
    out = np.rot90(inp, -1)
    progs = mcts_search([(inp, out)], iterations=1000, max_depth=1, seed=0)
    assert any(np.array_equal(apply_program(inp, p), out) for p in progs)
