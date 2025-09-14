"""Tests for canonicalisation utilities.

[S:TEST v1] unit=4 property=2 pass
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path for direct test execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

from arc_solver.canonical import D4, canonicalize_colors, canonicalize_D4

Array = np.ndarray

# Strategy for generating small grids with colours 0-9
array_shapes = hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5)
colour_arrays = hnp.arrays(np.int16, array_shapes, elements=st.integers(min_value=0, max_value=9))


def test_canonicalize_colors_type_checks() -> None:
    """canonicalize_colors rejects non-arrays and non-integer dtypes."""
    with pytest.raises(TypeError):
        canonicalize_colors([1, 2, 3])
    with pytest.raises(TypeError):
        canonicalize_colors(np.array([[0.5]]))


@given(colour_arrays)
def test_canonicalize_colors_frequency_order(grid: Array) -> None:
    """Colours are relabelled in descending frequency order with contiguous labels."""
    canonical, mapping = canonicalize_colors(grid)
    assert sorted(np.unique(canonical).tolist()) == list(range(len(mapping)))
    vals, counts = np.unique(grid, return_counts=True)
    expected_order = [int(v) for v, _ in sorted(zip(vals, counts), key=lambda t: (-t[1], t[0]))]
    assert list(mapping.keys()) == expected_order


def test_canonicalize_D4_type_checks() -> None:
    """canonicalize_D4 rejects non-arrays and non-integer dtypes."""
    with pytest.raises(TypeError):
        canonicalize_D4([1, 2, 3])
    with pytest.raises(TypeError):
        canonicalize_D4(np.array([[0.5]]))


@given(colour_arrays)
def test_canonicalize_D4_invariance(grid: Array) -> None:
    """Canonical form is invariant under D4 transformations and idempotent."""
    canonical = canonicalize_D4(grid)
    for transform in D4:
        transformed = transform(grid)
        assert np.array_equal(canonicalize_D4(transformed), canonical)
    assert np.array_equal(canonicalize_D4(canonical), canonical)
