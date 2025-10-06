"""
Extended grid utilities for the PUMA ARC solver system.

This module extends the base grid functionality with additional utilities
needed for agentic and genomic solvers, including color palette operations
and advanced grid manipulations.
"""

from ..grid import *
from typing import Dict, Set, Tuple
import numpy as np


def get_unique_colors(grid: Array) -> Set[int]:
    """Get set of all unique colors in a grid."""
    return set(int(c) for c in np.unique(grid))


def palette_permutation_distance(a: Array, b: Array) -> int:
    """
    Calculate minimum number of color swaps needed to transform palette of a to b.
    Returns -1 if transformation is impossible (different color counts).
    """
    from collections import Counter
    hist_a = Counter(a.flatten())
    hist_b = Counter(b.flatten())
    
    if sorted(hist_a.values()) != sorted(hist_b.values()):
        return -1
    
    # This is a simplified estimate - actual permutation distance is complex
    colors_a = set(hist_a.keys())
    colors_b = set(hist_b.keys())
    return len(colors_a.symmetric_difference(colors_b)) // 2


def canonical_palette(grid: Array) -> Dict[int, int]:
    """
    Return a mapping to canonicalize the color palette.
    Colors are relabeled in order of frequency (most frequent = 0).
    """
    from collections import Counter
    hist = Counter(grid.flatten())
    sorted_colors = sorted(hist.keys(), key=hist.get, reverse=True)
    return {old_color: new_color for new_color, old_color in enumerate(sorted_colors)}


def grid_hash(grid: Array) -> int:
    """Compute a hash of the grid for fast comparison."""
    return hash(grid.tobytes())


def is_monochrome(grid: Array) -> bool:
    """Check if grid contains only one color."""
    return len(np.unique(grid)) == 1


def get_color_boundaries(grid: Array) -> Dict[int, Tuple[int, int, int, int]]:
    """
    For each color, return its bounding box as (top, left, bottom, right).
    Returns empty dict if color not found.
    """
    boundaries = {}
    for color in np.unique(grid):
        positions = np.where(grid == color)
        if len(positions[0]) > 0:
            top, left = positions[0].min(), positions[1].min()
            bottom, right = positions[0].max(), positions[1].max()
            boundaries[int(color)] = (top, left, bottom, right)
    return boundaries