"""
Invariant checking utilities for ARC solver validation.

This module provides functions to check various invariants that should
hold between input and output grids in ARC tasks, helping to validate
and score potential solutions.
"""

from typing import List, Tuple, Dict, Set
import numpy as np
from collections import Counter

from ..grid import Array
from .objects import connected_components


def palette_equiv(a: Array, b: Array) -> bool:
    """Check if two grids have equivalent color palettes (same color frequencies)."""
    from collections import Counter
    return Counter(a.flatten()) == Counter(b.flatten())


def palette_permutation_equiv(a: Array, b: Array) -> bool:
    """
    Check if two grids are equivalent up to color permutation.
    Two grids are permutation-equivalent if one can be obtained from 
    the other by swapping color labels.
    """
    from collections import Counter
    hist_a = Counter(a.flatten())
    hist_b = Counter(b.flatten())
    
    # Same histogram of histogram values means permutation is possible
    freq_a = sorted(hist_a.values())
    freq_b = sorted(hist_b.values())
    return freq_a == freq_b


def object_count_delta(a: Array, b: Array) -> int:
    """Compute the change in number of objects from grid a to grid b."""
    objects_a = connected_components(a)
    objects_b = connected_components(b)
    return len(objects_b) - len(objects_a)


def object_count_invariant(a: Array, b: Array, allowed_deltas: Set[int] = {-1, 0, 1}) -> bool:
    """Check if object count change is within allowed range."""
    delta = object_count_delta(a, b)
    return delta in allowed_deltas


def shape_invariant(a: Array, b: Array) -> bool:
    """Check if two grids have the same shape."""
    return a.shape == b.shape


def background_color_invariant(a: Array, b: Array) -> bool:
    """Check if background color is preserved between grids."""
    from ..grid import bg_color
    return bg_color(a) == bg_color(b)


def symmetry_preserved(a: Array, b: Array) -> Dict[str, bool]:
    """
    Check which symmetries are preserved between input and output.
    Returns dict indicating which symmetries hold for both grids.
    """
    def has_symmetries(grid: Array) -> Dict[str, bool]:
        h, w = grid.shape
        
        # Vertical symmetry (left-right reflection)
        vert_sym = np.array_equal(grid, np.fliplr(grid))
        
        # Horizontal symmetry (top-bottom reflection)
        horiz_sym = np.array_equal(grid, np.flipud(grid))
        
        # 180-degree rotational symmetry
        rot180_sym = np.array_equal(grid, np.rot90(grid, 2))
        
        # 90-degree rotational symmetry (only possible for square grids)
        rot90_sym = False
        if h == w:
            rot90_sym = np.array_equal(grid, np.rot90(grid, 1))
        
        return {
            'vertical': vert_sym,
            'horizontal': horiz_sym,
            'rotation_180': rot180_sym,
            'rotation_90': rot90_sym
        }
    
    sym_a = has_symmetries(a)
    sym_b = has_symmetries(b)
    
    return {
        'vertical': sym_a['vertical'] and sym_b['vertical'],
        'horizontal': sym_a['horizontal'] and sym_b['horizontal'],
        'rotation_180': sym_a['rotation_180'] and sym_b['rotation_180'],
        'rotation_90': sym_a['rotation_90'] and sym_b['rotation_90']
    }


def color_count_preserved(a: Array, b: Array) -> bool:
    """Check if the number of distinct colors is preserved."""
    return len(np.unique(a)) == len(np.unique(b))


def bounding_box_invariant(a: Array, b: Array, tolerance: int = 0) -> bool:
    """
    Check if non-background regions have similar bounding boxes.
    Tolerance allows for small changes in bounding box size.
    """
    from ..grid import bg_color
    
    def non_bg_bbox(grid: Array) -> Tuple[int, int, int, int]:
        bg = bg_color(grid)
        positions = np.where(grid != bg)
        if len(positions[0]) == 0:
            return (0, 0, 0, 0)
        top, left = positions[0].min(), positions[1].min()
        bottom, right = positions[0].max(), positions[1].max()
        return (top, left, bottom - top + 1, right - left + 1)
    
    bbox_a = non_bg_bbox(a)
    bbox_b = non_bg_bbox(b)
    
    if tolerance == 0:
        return bbox_a == bbox_b
    
    # Check if bounding boxes are within tolerance
    return (abs(bbox_a[0] - bbox_b[0]) <= tolerance and
            abs(bbox_a[1] - bbox_b[1]) <= tolerance and
            abs(bbox_a[2] - bbox_b[2]) <= tolerance and
            abs(bbox_a[3] - bbox_b[3]) <= tolerance)


def connectivity_preserved(a: Array, b: Array) -> bool:
    """
    Check if objects maintain their internal connectivity.
    This is a simplified check - more sophisticated versions could
    check specific connectivity patterns.
    """
    objects_a = connected_components(a)
    objects_b = connected_components(b)
    
    # Simple check: same number of objects
    if len(objects_a) != len(objects_b):
        return False
    
    # Check if object sizes are similar (allows for small changes)
    sizes_a = sorted([obj['area'] for obj in objects_a])
    sizes_b = sorted([obj['area'] for obj in objects_b])
    
    return sizes_a == sizes_b


def evaluate_invariants(input_grid: Array, output_grid: Array) -> Dict[str, bool]:
    """
    Evaluate all invariants and return a report.
    This is useful for debugging and understanding what changed.
    """
    return {
        'palette_equiv': palette_equiv(input_grid, output_grid),
        'palette_permutation': palette_permutation_equiv(input_grid, output_grid),
        'shape_preserved': shape_invariant(input_grid, output_grid),
        'background_preserved': background_color_invariant(input_grid, output_grid),
        'object_count_stable': object_count_invariant(input_grid, output_grid),
        'color_count_preserved': color_count_preserved(input_grid, output_grid),
        'bounding_box_stable': bounding_box_invariant(input_grid, output_grid),
        'connectivity_preserved': connectivity_preserved(input_grid, output_grid),
    }


def invariant_score(input_grid: Array, output_grid: Array, weights: Dict[str, float] = None) -> float:
    """
    Compute a weighted score based on how many invariants are satisfied.
    Higher scores indicate better preservation of expected properties.
    """
    if weights is None:
        weights = {
            'palette_equiv': 1.0,
            'palette_permutation': 0.8,
            'shape_preserved': 1.0,
            'background_preserved': 0.6,
            'object_count_stable': 0.8,
            'color_count_preserved': 0.6,
            'bounding_box_stable': 0.4,
            'connectivity_preserved': 0.7,
        }
    
    invariants = evaluate_invariants(input_grid, output_grid)
    total_weight = sum(weights.values())
    
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(weights.get(inv, 0) * (1 if satisfied else 0)
                      for inv, satisfied in invariants.items())
    
    return weighted_sum / total_weight