"""
Object extraction utilities for the ARC solver.

ARC tasks often involve reasoning about contiguous patches of color (objects).
This module provides simple connected component detection and symmetry
information for grids. Extracted objects include their color, bounding box and
mask to aid further reasoning.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any

from .grid import Array, bg_color


def neighbors4(y: int, x: int) -> List[Tuple[int, int]]:
    """Return the coordinates of 4-neighbourhood around (y,x)."""
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


def connected_components(a: Array) -> List[Dict[str, Any]]:
    """Find all 4-connected components in the grid, grouping by exact color.

    Each component dictionary contains:
      - color: the color value of the component
      - bbox: (top, left, height, width) of the bounding box
      - mask: a 2D array of shape (height, width) with the component values
      - pixels: list of (row, col) indices in original grid
    """
    h, w = a.shape
    visited = np.zeros_like(a, dtype=bool)
    comps: List[Dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if visited[y, x]:
                continue
            color = int(a[y, x])
            q = [(y, x)]
            pix: List[Tuple[int, int]] = []
            visited[y, x] = True
            while q:
                cy, cx = q.pop()
                pix.append((cy, cx))
                for ny, nx in neighbors4(cy, cx):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and a[ny, nx] == color:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            if pix:
                ys = [p[0] for p in pix]
                xs = [p[1] for p in pix]
                top, left = min(ys), min(xs)
                bottom, right = max(ys), max(xs)
                mask = np.zeros((bottom - top + 1, right - left + 1), dtype=np.int16)
                for py, px in pix:
                    mask[py - top, px - left] = color
                comps.append({
                    "color": color,
                    "bbox": (top, left, bottom - top + 1, right - left + 1),
                    "mask": mask,
                    "pixels": pix,
                })
    return comps


def infer_symmetries(a: Array) -> Dict[str, bool]:
    """Return a simple dictionary of possible symmetries in the grid.

    For speed, this function does not check each symmetry but sets flags to True.
    More precise symmetry detection can be incorporated later based on heuristics.
    """
    return {
        "rot90": True,
        "rot180": True,
        "rot270": True,
        "flip_h": True,
        "flip_v": True,
    }