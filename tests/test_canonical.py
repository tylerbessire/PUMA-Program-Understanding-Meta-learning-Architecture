"""Tests for canonicalisation utilities.

[S:TEST v2] unit=6 property=3 pass
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

from arc_solver.canonical import D4, canonicalize_colors, canonicalize_D4, canonicalize_pair

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


def test_canonicalize_pair_type_checks() -> None:
    """canonicalize_pair rejects non-arrays and non-integer dtypes."""
    with pytest.raises(TypeError):
        canonicalize_pair([1], np.array([[1]]))
    with pytest.raises(TypeError):
        canonicalize_pair(np.array([[1.0]]), np.array([[1]]))


@given(colour_arrays, colour_arrays)
def test_canonicalize_pair_invariance(a: Array, b: Array) -> None:
    """Canonical pair invariant under joint D4 transforms and idempotent."""
    can_a, can_b = canonicalize_pair(a, b)
    for transform in D4:
        ta = transform(a)
        tb = transform(b)
        cta, ctb = canonicalize_pair(ta, tb)
        assert np.array_equal(cta, can_a)
        assert np.array_equal(ctb, can_b)
    ca2, cb2 = canonicalize_pair(can_a, can_b)
    assert np.array_equal(ca2, can_a)
    assert np.array_equal(cb2, can_b)
