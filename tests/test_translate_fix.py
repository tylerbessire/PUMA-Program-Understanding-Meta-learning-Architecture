import json
from pathlib import Path
import sys

import numpy as np
from hypothesis import given, strategies as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from arc_solver.dsl import apply_program
from arc_solver.heuristics_complete import detect_basic_transformations
from arc_solver.neural.episodic import Episode


def test_translate_fill_value_alias_and_detection():
    inp = np.array([[1, 0], [0, 0]])
    expected = np.array([[0, 0], [0, 1]])

    # legacy alias still works
    legacy_prog = [("translate", {"dy": 1, "dx": 1, "fill_value": 0})]
    assert np.array_equal(apply_program(inp, legacy_prog), expected)

    # heuristics now emit canonical parameter name
    detected = detect_basic_transformations(inp, expected)
    assert [("translate", {"dy": 1, "dx": 1, "fill": 0})] in detected


@given(st.integers(-3, 3), st.integers(-3, 3), st.integers(0, 9))
def test_episode_translate_roundtrip(dy: int, dx: int, fill: int) -> None:
    """Episode serialisation normalises translate parameters to ints."""
    inp = np.array([[1]])
    out = apply_program(inp, [("translate", {"dy": dy, "dx": dx, "fill": fill})])
    episode = Episode(
        task_signature="sig",
        programs=[[('translate', {'dy': str(dy), 'dx': str(dx), 'fill_value': str(fill)})]],
        train_pairs=[(inp, out)],
    )
    data = json.loads(json.dumps(episode.to_dict()))
    loaded = Episode.from_dict(data)
    op, params = loaded.programs[0][0]
    assert op == 'translate'
    assert all(isinstance(params[k], int) for k in ('dy', 'dx', 'fill'))
    assert np.array_equal(apply_program(inp, [(op, params)]), out)
