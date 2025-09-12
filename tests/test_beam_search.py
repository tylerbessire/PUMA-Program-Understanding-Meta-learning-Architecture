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

def test_beam_search_low_prior_operations_still_found():
    """Test that valid programs using low-prior operations are still found as solutions."""
    inp = to_array([[1, 0], [0, 0]])
    out = np.rot90(inp, -1)  # Correct transformation is rotation
    
    # Give rotation a very low prior score but keep it non-zero for ranking
    scores = {op: 1.0 for op in ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']}
    scores['rotate'] = 0.001  # Very low but non-zero prior
    
    progs, _ = beam_search([(inp, out)], beam_width=5, depth=1, op_scores=scores)
    
    # The correct rotation program should still be found despite low prior
    rotation_found = any(p == [("rotate", {"k": 1})] for p in progs)
    assert rotation_found, "Valid rotation program should be found despite low prior score"


def test_beam_search_zero_prior_operations_excluded_from_ranking():
    """Test that operations with zero priors are excluded from ranking but solutions still work."""
    inp = to_array([[1, 0], [0, 0]])
    out = np.rot90(inp, -1)  # Correct transformation is rotation
    
    # Give rotation zero prior - it should still find solution but not appear in ranking
    scores = {op: 1.0 for op in ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']}
    scores['rotate'] = 0.0  # Zero prior
    
    progs, _ = beam_search([(inp, out)], beam_width=5, depth=1, op_scores=scores)
    
    # The correct rotation program should still be found as a complete solution
    rotation_found = any(p == [("rotate", {"k": 1})] for p in progs)
    assert rotation_found, "Valid rotation program should be found even with zero prior"
