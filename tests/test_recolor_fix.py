import json
from typing import Dict
import sys
from pathlib import Path

import numpy as np
from hypothesis import given, strategies as st

sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.dsl import apply_program
from arc_solver.heuristics_complete import detect_color_patterns
from arc_solver.neural.episodic import Episode


def test_detect_color_patterns_recolor_program() -> None:
    """Heuristic recolor programs use mapping parameter."""
    inp = to_array([[1, 0], [0, 0]])
    out = to_array([[2, 0], [0, 0]])
    programs = detect_color_patterns(inp, out)
    assert [("recolor", {"mapping": {1: 2}})] in programs
    assert np.array_equal(apply_program(inp, programs[0]), out)


@given(st.dictionaries(st.integers(min_value=1, max_value=9),
                       st.integers(min_value=0, max_value=9),
                       min_size=1, max_size=3).filter(lambda m: all(k != v for k, v in m.items())))
def test_episode_recolor_roundtrip(mapping: Dict[int, int]) -> None:
    """Episode serialization preserves integer recolor mappings."""
    src, dst = next(iter(mapping.items()))
    inp = to_array([[src]])
    out = to_array([[dst]])
    episode = Episode(task_signature="sig", programs=[[('recolor', {'mapping': mapping})]],
                      train_pairs=[(inp, out)])
    data = json.loads(json.dumps(episode.to_dict()))
    loaded = Episode.from_dict(data)
    prog = loaded.programs[0]
    assert prog[0][1]['mapping'] == {int(k): int(v) for k, v in mapping.items()}
    assert np.array_equal(apply_program(inp, prog), out)
