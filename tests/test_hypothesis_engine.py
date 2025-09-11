"""Tests for the HypothesisEngine."""

import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.hypothesis import HypothesisEngine
from arc_solver.grid import to_array


def test_rotation_hypothesis_generation():
    engine = HypothesisEngine()
    inp = to_array([[1, 0], [2, 0]])
    out = np.rot90(inp)
    hyps = engine.generate_hypotheses([(inp, out)])
    assert any(
        h.transformation_type == "rotation" and h.program_sketch[0][1]["k"] == 1
        for h in hyps
    )


def test_color_mapping_hypothesis_generation():
    engine = HypothesisEngine()
    inp = to_array([[1, 2], [1, 2]])
    out = to_array([[2, 3], [2, 3]])
    hyps = engine.generate_hypotheses([(inp, out)])
    assert any(h.transformation_type == "color_swap" for h in hyps)
    h = hyps[0]
    score = engine.test_hypothesis(h, [(inp, out)])
    assert score >= 0

