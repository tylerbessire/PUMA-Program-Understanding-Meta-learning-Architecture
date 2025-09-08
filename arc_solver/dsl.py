"""
Domain-specific language (DSL) primitives for ARC program synthesis.

This module defines a small set of composable operations that act on grids.
Each operation is represented by an `Op` object with a name, a function, and
metadata about its parameters. Programs are sequences of these operations.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Dict, List, Tuple

from .grid import Array, rotate90, flip, transpose, translate, color_map, crop, pad_to, bg_color


class Op:
    """Represents a primitive transformation on a grid.

    Attributes
    ----------
    name : str
        Human-readable name of the operation.
    fn : Callable
        Function implementing the operation.
    arity : int
        Number of input grids (arity=1 for single-grid ops).
    param_names : List[str]
        Names of parameters accepted by the operation.
    """

    def __init__(self, name: str, fn: Callable[..., Array], arity: int, param_names: List[str]):
        self.name = name
        self.fn = fn
        self.arity = arity
        self.param_names = param_names

    def __call__(self, *args, **kwargs) -> Array:
        return self.fn(*args, **kwargs)


# Primitive operations (single-grid)
def op_identity(a: Array) -> Array:
    return a


def op_rotate(a: Array, k: int) -> Array:
    return rotate90(a, k)


def op_flip(a: Array, axis: int) -> Array:
    return flip(a, axis)


def op_transpose(a: Array) -> Array:
    return transpose(a)


def op_translate(a: Array, dy: int, dx: int) -> Array:
    return translate(a, dy, dx, fill=bg_color(a))


def op_recolor(a: Array, mapping: Dict[int, int]) -> Array:
    return color_map(a, mapping)


def op_crop_bbox(a: Array, top: int, left: int, height: int, width: int) -> Array:
    # ensure cropping stays inside bounds
    h, w = a.shape
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    height = max(1, min(height, h - top))
    width = max(1, min(width, w - left))
    return crop(a, top, left, height, width)


def op_pad(a: Array, out_h: int, out_w: int) -> Array:
    return pad_to(a, (out_h, out_w), fill=bg_color(a))


# Register operations in a dictionary for easy lookup
OPS: Dict[str, Op] = {
    "identity": Op("identity", op_identity, 1, []),
    "rotate": Op("rotate", op_rotate, 1, ["k"]),
    "flip": Op("flip", op_flip, 1, ["axis"]),
    "transpose": Op("transpose", op_transpose, 1, []),
    "translate": Op("translate", op_translate, 1, ["dy", "dx"]),
    "recolor": Op("recolor", op_recolor, 1, ["mapping"]),
    "crop": Op("crop", op_crop_bbox, 1, ["top", "left", "height", "width"]),
    "pad": Op("pad", op_pad, 1, ["out_h", "out_w"]),
}


def apply_program(a: Array, program: List[Tuple[str, Dict[str, Any]]]) -> Array:
    """Apply a sequence of operations (program) to the input array.

    Parameters
    ----------
    a : Array
        Input grid.
    program : List of (op_name, params)
        Sequence of operations with parameters. The operations are looked up in
        OPS.

    Returns
    -------
    Array
        Resulting grid after applying the program.
    """
    out = a
    for name, params in program:
        op = OPS[name]
        out = op(out, **params)
    return out