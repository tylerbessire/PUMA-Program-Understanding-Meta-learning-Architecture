from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from arc_solver.dsl import apply_program
from arc_solver.heuristics_complete import detect_basic_transformations


def test_translate_fill_value_alias_and_detection():
    inp = np.array([[1, 0], [0, 0]])
    expected = np.array([[0, 0], [0, 1]])

    # legacy alias still works
    legacy_prog = [("translate", {"dy": 1, "dx": 1, "fill_value": 0})]
    assert np.array_equal(apply_program(inp, legacy_prog), expected)

    # heuristics now emit canonical parameter name
    detected = detect_basic_transformations(inp, expected)
    assert [("translate", {"dy": 1, "dx": 1, "fill": 0})] in detected
