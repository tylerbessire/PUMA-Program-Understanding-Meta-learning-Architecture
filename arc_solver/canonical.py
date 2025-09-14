"""Canonicalisation utilities for ARC grids.

This module provides functions to normalise grids under the D4 symmetry group
(rotations and reflections) and canonicalise colour labels. Canonicalisation
reduces the search space by treating symmetric grids as identical.
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

Array = np.ndarray

# Precompute the eight transformations in the D4 symmetry group.
D4: Tuple[callable, ...] = (
    lambda g: g,
    lambda g: np.rot90(g, 1),
    lambda g: np.rot90(g, 2),
    lambda g: np.rot90(g, 3),
    lambda g: np.flipud(g),
    lambda g: np.fliplr(g),
    lambda g: np.rot90(np.flipud(g), 1),
    lambda g: np.rot90(np.fliplr(g), 1),
)


def canonicalize_colors(grid: Array) -> Tuple[Array, Dict[int, int]]:
    """Relabel colours in ``grid`` in descending frequency order.

    Parameters
    ----------
    grid:
        Input array containing integer colour labels.

    Returns
    -------
    canonical:
        Array with colours mapped to ``0..n-1`` in frequency order.
    mapping:
        Dictionary mapping original colours to canonical labels.

    Raises
    ------
    TypeError
        If ``grid`` is not a ``numpy.ndarray`` or is not of integer type.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be a numpy array")
    if not np.issubdtype(grid.dtype, np.integer):
        raise TypeError("grid dtype must be integer")

    vals, counts = np.unique(grid, return_counts=True)
    order = [int(v) for v, _ in sorted(zip(vals, counts), key=lambda t: (-t[1], t[0]))]
    mapping = {c: i for i, c in enumerate(order)}
    vect_map = np.vectorize(mapping.get)
    canonical = vect_map(grid)
    return canonical.astype(np.int16), mapping


def canonicalize_D4(grid: Array) -> Array:
    """Return the lexicographically smallest grid under D4 symmetries.

    The grid is first transformed by each D4 element, then colour-canonicalised.
    The transformation with the smallest shape and byte representation is chosen
    as the canonical representative.

    Parameters
    ----------
    grid:
        Input array to canonicalise.

    Returns
    -------
    np.ndarray
        Canonicalised grid.

    Raises
    ------
    TypeError
        If ``grid`` is not a ``numpy.ndarray`` or is not of integer type.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("grid must be a numpy array")
    if not np.issubdtype(grid.dtype, np.integer):
        raise TypeError("grid dtype must be integer")

    best: Array | None = None
    best_key: Tuple[Tuple[int, int], bytes] | None = None
    for transform in D4:
        transformed = transform(grid)
        canonical, _ = canonicalize_colors(transformed)
        key = (canonical.shape, canonical.tobytes())
        if best_key is None or key < best_key:
            best, best_key = canonical, key
    if best is None:
        # This should not occur because D4 contains identity, but guard anyway.
        return grid.copy()
    return best


def canonicalize_pair(input_grid: Array, output_grid: Array) -> Tuple[Array, Array]:
    """Canonicalise a pair of grids under shared D4 symmetries and colours.

    The same D4 transform and colour relabelling are applied to both ``input_grid``
    and ``output_grid`` so that puzzle examples remain aligned.

    [S:ALG v2] pair-D4 canonicalisation pass

    Parameters
    ----------
    input_grid, output_grid:
        Arrays representing an ARC training pair.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Canonicalised input and output grids.

    Raises
    ------
    TypeError
        If either grid is not a ``numpy.ndarray`` of integer dtype.
    """

    if not isinstance(input_grid, np.ndarray) or not isinstance(output_grid, np.ndarray):
        raise TypeError("grids must be numpy arrays")
    if not np.issubdtype(input_grid.dtype, np.integer) or not np.issubdtype(output_grid.dtype, np.integer):
        raise TypeError("grid dtype must be integer")

    best_in: Array | None = None
    best_out: Array | None = None
    best_key: Tuple[Tuple[int, int], bytes, Tuple[int, int], bytes] | None = None
    for transform in D4:
        inp_t = transform(input_grid)
        out_t = transform(output_grid)
        vals, counts = np.unique(np.concatenate([inp_t.ravel(), out_t.ravel()]), return_counts=True)
        order = [int(v) for v, _ in sorted(zip(vals, counts), key=lambda t: (-t[1], t[0]))]
        mapping = {c: i for i, c in enumerate(order)}
        vect_map = np.vectorize(mapping.get)
        inp_c = vect_map(inp_t).astype(np.int16)
        out_c = vect_map(out_t).astype(np.int16)
        key = (inp_c.shape, inp_c.tobytes(), out_c.shape, out_c.tobytes())
        if best_key is None or key < best_key:
            best_in, best_out, best_key = inp_c, out_c, key
    if best_in is None or best_out is None:
        # This should not occur because D4 contains identity, but guard anyway.
        return input_grid.copy(), output_grid.copy()
    return best_in, best_out
