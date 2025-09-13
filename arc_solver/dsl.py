"""Domain-specific language (DSL) primitives for ARC program synthesis.

This module defines a set of composable operations that act on grids. Each
operation is represented by an :class:`Op` and registered in :data:`OPS`.
Programs are sequences of these operations applied to a grid.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np

from arc_solver.grid import (
    Array,
    rotate90,
    flip as flip_grid,
    transpose as transpose_grid,
    translate as translate_grid,
    color_map as color_map_grid,
    crop as crop_array,
    pad_to,
    bg_color,
)


class Op:
    """Represents a primitive transformation on a grid."""

    def __init__(self, name: str, fn: Callable[..., Array], arity: int, param_names: List[str]):
        self.name = name
        self.fn = fn
        self.arity = arity
        self.param_names = param_names

    def __call__(self, *args: Any, **kwargs: Any) -> Array:
        return self.fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Primitive operation implementations
# ---------------------------------------------------------------------------

def op_identity(a: Array) -> Array:
    """Return a copy of the input grid."""
    return a.copy()


def op_rotate(a: Array, k: int) -> Array:
    """Rotate grid by ``k`` quarter turns clockwise."""
    return rotate90(a, -k)


def op_flip(a: Array, axis: int) -> Array:
    """Flip grid along the specified axis (0=vertical, 1=horizontal)."""
    return flip_grid(a, axis)


def op_transpose(a: Array) -> Array:
    """Transpose the grid."""
    return transpose_grid(a)


def op_translate(a: Array, dy: int, dx: int, fill: Optional[int] = None, *, fill_value: Optional[int] = None) -> Array:
    """Translate the grid by ``(dy, dx)`` filling uncovered cells.

    Parameters
    ----------
    a:
        Input grid.
    dy, dx:
        Translation offsets. Positive values move content down/right.
    fill, fill_value:
        Optional fill value for uncovered cells. ``fill_value`` is an alias for
        backward compatibility. When both are ``None`` the background colour of
        ``a`` is used.
    """
    chosen = fill if fill is not None else fill_value
    fill_val = 0 if chosen is None else chosen
    return translate_grid(a, dy, dx, fill=fill_val)


def op_recolor(a: Array, mapping: Dict[int, int]) -> Array:
    """Recolour grid according to a mapping from old to new colours."""
    return color_map_grid(a, mapping)


def op_crop_bbox(a: Array, top: int, left: int, height: int, width: int) -> Array:
    """Crop a bounding box from the grid ensuring bounds are valid."""
    h, w = a.shape
    top = max(0, min(top, h - 1))
    left = max(0, min(left, w - 1))
    height = max(1, min(height, h - top))
    width = max(1, min(width, w - left))
    return crop_array(a, top, left, height, width)


def op_pad(a: Array, out_h: int, out_w: int) -> Array:
    """Pad grid to a specific height and width using background colour."""
    return pad_to(a, (out_h, out_w), fill=bg_color(a))


# Registry of primitive operations ---------------------------------------------------------
OPS: Dict[str, Op] = {
    "identity": Op("identity", op_identity, 1, []),
    "rotate": Op("rotate", op_rotate, 1, ["k"]),
    "flip": Op("flip", op_flip, 1, ["axis"]),
    "transpose": Op("transpose", op_transpose, 1, []),
    "translate": Op("translate", op_translate, 1, ["dy", "dx", "fill"]),
    "recolor": Op("recolor", op_recolor, 1, ["mapping"]),
    "crop": Op("crop", op_crop_bbox, 1, ["top", "left", "height", "width"]),
    "pad": Op("pad", op_pad, 1, ["out_h", "out_w"]),
}


# Semantic cache -------------------------------------------------------------------------
_sem_cache: Dict[Tuple[bytes, str, Tuple[Tuple[str, Any], ...]], Array] = {}


def _canonical_params(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``params`` with legacy aliases normalised and typed."""
    new_params = dict(params)
    if name == "recolor":
        mapping = new_params.get("mapping") or new_params.pop("color_map", {})
        if mapping:
            new_params["mapping"] = {int(k): int(v) for k, v in mapping.items()}
    elif name == "translate":
        if "fill" not in new_params and "fill_value" in new_params:
            new_params["fill"] = new_params.pop("fill_value")
        for key in ("dy", "dx", "fill"):
            if key in new_params and new_params[key] is not None:
                new_params[key] = int(new_params[key])
    return new_params



def _norm_params(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Normalise parameters to a hashable tuple."""
    items: List[Tuple[str, Any]] = []
    for k, v in sorted(params.items()):
        if isinstance(v, dict):
            items.append((k, tuple(sorted(v.items()))))
        else:
            items.append((k, v))
    return tuple(items)


def apply_op(a: Array, name: str, params: Dict[str, Any]) -> Array:
    """Apply a primitive operation with semantic caching."""
    params = _canonical_params(name, params)
    key = (a.tobytes(), name, _norm_params(params))
    cached = _sem_cache.get(key)
    if cached is not None:
        return cached
    op = OPS[name]
    out = op(a, **params)
    _sem_cache[key] = out
    return out


# User-facing convenience wrappers --------------------------------------------------------

def identity(a: Array) -> Array:
    """Return a copy of the input grid."""
    return op_identity(a)


def rotate(a: Array, k: int) -> Array:
    """Rotate grid by ``k`` quarter turns clockwise."""
    return op_rotate(a, k)


def flip(a: Array, axis: int) -> Array:
    """Flip grid along the specified axis."""
    return op_flip(a, axis)


def transpose(a: Array) -> Array:
    """Transpose the grid."""
    return op_transpose(a)


def translate(a: Array, dx: int, dy: int, fill_value: Optional[int] = None) -> Array:
    """Translate grid by ``(dy, dx)`` with optional fill value."""
    return op_translate(a, dy, dx, fill=fill_value)


def recolor(a: Array, color_map: Dict[int, int]) -> Array:
    """Recolour grid according to a mapping."""
    return op_recolor(a, color_map)


def crop(a: Array, top: int, bottom: int, left: int, right: int) -> Array:
    """Crop a region specified by inclusive-exclusive bounds.

    Args:
        top, bottom, left, right: Bounds following Python slice semantics where
            ``bottom`` and ``right`` are exclusive.
    """
    if bottom <= top or right <= left:
        raise ValueError("Invalid crop bounds")
    h, w = a.shape
    top = max(0, min(top, h))
    bottom = max(top, min(bottom, h))
    left = max(0, min(left, w))
    right = max(left, min(right, w))
    return a[top:bottom, left:right].copy()


def pad(a: Array, top: int, bottom: int, left: int, right: int, fill_value: int = 0) -> Array:
    """Pad grid with ``fill_value`` on each side."""
    if min(top, bottom, left, right) < 0:
        raise ValueError("Pad widths must be non-negative")
    h, w = a.shape
    out = np.full((h + top + bottom, w + left + right), fill_value, dtype=a.dtype)
    out[top:top + h, left:left + w] = a
    return out


# Program application --------------------------------------------------------------------

def apply_program(a: Array, program: List[Tuple[str, Dict[str, Any]]]) -> Array:
    """Apply a sequence of operations to the input grid."""
    out = a
    for idx, (name, params) in enumerate(program):
        try:
            out = apply_op(out, name, params)
        except Exception as exc:
            raise ValueError(
                f"Failed to apply operation '{name}' at position {idx} with params {params}"
            ) from exc
    return out


__all__ = [
    "Array",
    "Op",
    "OPS",
    "apply_program",
    "apply_op",
    "identity",
    "rotate",
    "flip",
    "transpose",
    "translate",
    "recolor",
    "crop",
    "pad",
]
