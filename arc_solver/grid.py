"""
Grid utilities for the ARC solver.

This module defines fundamental operations on grids (2D numpy arrays)
representing ARC tasks. Operations include rotation, flipping, translation,
padding, cropping, and color mapping. These helpers are used throughout the
solver to build and manipulate candidate solutions.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any


# Type alias for clarity. ARC grids are small 2D arrays of integers.
Array = np.ndarray

__all__ = [
    "Array",
    "to_array",
    "to_list",
    "same_shape",
    "rotate90",
    "flip",
    "transpose",
    "pad_to",
    "crop",
    "translate",
    "color_map",
    "histogram",
    "eq",
    "bg_color",
]


def to_array(grid: Any) -> Array:
    """Convert a nested Python list into a numpy array of dtype int16."""
    a = np.asarray(grid, dtype=np.uint8)

    if a.ndim == 1:
        a = a[None, :] 

    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]

    assert a.ndim == 2, f"ARC grid must be 2-D, got {a.ndim}D shape={a.shape}"
    return a.astype(np.int16)


def to_list(arr: Array) -> List[List[int]]:
    """Convert a numpy array back into a nested Python list of ints."""
    return arr.astype(int).tolist()


def same_shape(a: Array, b: Array) -> bool:
    """Return True if two arrays have identical shape."""
    return a.shape == b.shape


def rotate90(a: Array, k: int = 1) -> Array:
    """Rotate an array by k*90 degrees counter-clockwise."""
    return np.rot90(a, k % 4)


def flip(a: Array, axis: int) -> Array:
    """Flip an array along the given axis (0 for vertical, 1 for horizontal)."""
    return np.flip(a, axis=axis)


def transpose(a: Array) -> Array:
    """Return the transpose of the array, copying to ensure contiguous memory."""
    return a.T.copy()


def pad_to(a: Array, shape: Tuple[int, int], fill: int = 0) -> Array:
    """Pad or crop an array to a target shape, filling new cells with `fill`.

    Parameters
    ----------
    a : np.ndarray
        Input array to pad or crop.
    shape : (int, int)
        Desired (height, width) of the output.
    fill : int, optional
        Value used for padded regions, by default 0.

    Returns
    -------
    np.ndarray
        Array of the requested shape containing the original array in the top-left
        corner and filled with `fill` elsewhere.
    """
    h, w = shape
    out = np.full((h, w), fill, dtype=a.dtype)
    hh, ww = a.shape
    out[: min(hh, h), : min(ww, w)] = a[: min(hh, h), : min(ww, w)]
    return out


def crop(a: Array, top: int, left: int, height: int, width: int) -> Array:
    """Safely crop a subarray given top-left coordinates and dimensions."""
    return a[top : top + height, left : left + width].copy()


def translate(a: Array, dy: int, dx: int, fill: int = 0) -> Array:
    """Translate an array by (dy, dx)."""
    h, w = a.shape
    out = np.full_like(a, fill)
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = a[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def color_map(a: Array, mapping: Dict[int, int]) -> Array:
    """Map each color in the array according to a dictionary.

    Non-mentioned colors remain unchanged.
    """
    out = a.copy()
    for c_from, c_to in mapping.items():
        out[a == c_from] = c_to
    return out


def histogram(a: Array) -> Dict[int, int]:
    """Return a dictionary mapping color values to their counts in the array."""
    vals, counts = np.unique(a, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def eq(a: Array, b: Array) -> bool:
    """Check equality of two arrays (shape and element-wise).

    Safely handles non-array comparisons by falling back to Python's
    equality semantics when either operand is not a ``numpy.ndarray``.
    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b

# [S:ALG v1] eq-check=shape+elementwise fallthrough=python-eq pass


def bg_color(a: Array) -> int:
    """Return the most frequent color in the array (background heuristic)."""
    vals, counts = np.unique(a, return_counts=True)
    idx = int(np.argmax(counts))
    return int(vals[idx])
