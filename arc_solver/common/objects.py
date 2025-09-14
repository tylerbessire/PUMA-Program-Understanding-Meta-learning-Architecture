"""
Extended object detection and analysis utilities.

This module extends the base object functionality with additional features
for agentic and genomic solvers, including symmetry detection and object
relationship analysis.
"""

from ..objects import connected_components as base_connected_components
from ..grid import Array
from typing import List, Dict, Any, Tuple, Set
import numpy as np


def connected_components(grid: Array) -> List[Dict[str, Any]]:
    """Enhanced connected components with additional metadata."""
    base_comps = base_connected_components(grid)
    
    # Add enhanced metadata
    for comp in base_comps:
        comp['area'] = len(comp['pixels'])
        comp['centroid'] = _compute_centroid(comp['pixels'])
        comp['symmetries'] = _detect_object_symmetries(comp['mask'])
        comp['is_rectangular'] = _is_rectangular_shape(comp['mask'], comp['color'])
        comp['perimeter'] = _compute_perimeter(comp['mask'], comp['color'])
    
    return base_comps


def _compute_centroid(pixels: List[Tuple[int, int]]) -> Tuple[float, float]:
    """Compute centroid of a list of pixel coordinates."""
    if not pixels:
        return (0.0, 0.0)
    y_coords = [p[0] for p in pixels]
    x_coords = [p[1] for p in pixels]
    return (sum(y_coords) / len(y_coords), sum(x_coords) / len(x_coords))


def _detect_object_symmetries(mask: Array) -> Dict[str, bool]:
    """Detect symmetries in an object mask."""
    h, w = mask.shape
    
    # Vertical symmetry (left-right)
    vert_sym = True
    for y in range(h):
        for x in range(w // 2):
            if mask[y, x] != mask[y, w - 1 - x]:
                vert_sym = False
                break
        if not vert_sym:
            break
    
    # Horizontal symmetry (top-bottom)
    horiz_sym = True
    for y in range(h // 2):
        for x in range(w):
            if mask[y, x] != mask[h - 1 - y, x]:
                horiz_sym = False
                break
        if not horiz_sym:
            break
    
    # Diagonal symmetries (simplified check)
    diag_sym = h == w  # Only square objects can have diagonal symmetry
    if diag_sym:
        for y in range(h):
            for x in range(w):
                if mask[y, x] != mask[x, y]:
                    diag_sym = False
                    break
            if not diag_sym:
                break
    
    return {
        'vertical': vert_sym,
        'horizontal': horiz_sym,
        'diagonal': diag_sym,
        'point': vert_sym and horiz_sym
    }


def _is_rectangular_shape(mask: Array, color: int) -> bool:
    """Check if the object has a rectangular shape."""
    # Count non-background pixels
    object_pixels = np.sum(mask == color)
    h, w = mask.shape
    
    # Perfect rectangle would fill the entire bounding box
    return object_pixels == h * w


def _compute_perimeter(mask: Array, color: int) -> int:
    """Compute perimeter of an object (number of edge pixels)."""
    h, w = mask.shape
    perimeter = 0
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] == color:
                # Check if this pixel is on the edge
                is_edge = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (ny < 0 or ny >= h or nx < 0 or nx >= w or 
                        mask[ny, nx] != color):
                        is_edge = True
                        break
                if is_edge:
                    perimeter += 1
    
    return perimeter


def find_object_relationships(objects: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find spatial relationships between objects.
    Returns dict with relationship types as keys and pairs of object indices as values.
    """
    relationships = {
        'adjacent': [],
        'contained': [],
        'aligned_horizontal': [],
        'aligned_vertical': [],
        'same_color': [],
        'same_size': []
    }
    
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            obj1, obj2 = objects[i], objects[j]
            
            # Same color
            if obj1['color'] == obj2['color']:
                relationships['same_color'].append((i, j))
            
            # Same size
            if obj1['area'] == obj2['area']:
                relationships['same_size'].append((i, j))
            
            # Check alignment
            c1_y, c1_x = obj1['centroid']
            c2_y, c2_x = obj2['centroid']
            
            if abs(c1_y - c2_y) < 1.0:  # Same row (approximately)
                relationships['aligned_horizontal'].append((i, j))
            
            if abs(c1_x - c2_x) < 1.0:  # Same column (approximately)
                relationships['aligned_vertical'].append((i, j))
            
            # Check adjacency (simplified)
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            if _bboxes_adjacent(bbox1, bbox2):
                relationships['adjacent'].append((i, j))
            
            # Check containment
            if _bbox_contains(bbox1, bbox2):
                relationships['contained'].append((i, j))
            elif _bbox_contains(bbox2, bbox1):
                relationships['contained'].append((j, i))
    
    return relationships


def _bboxes_adjacent(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes are adjacent."""
    t1, l1, h1, w1 = bbox1
    t2, l2, h2, w2 = bbox2
    
    b1, r1 = t1 + h1, l1 + w1
    b2, r2 = t2 + h2, l2 + w2
    
    # Check if they touch but don't overlap
    vertical_touch = (r1 == l2 or r2 == l1) and not (b1 <= t2 or b2 <= t1)
    horizontal_touch = (b1 == t2 or b2 == t1) and not (r1 <= l2 or r2 <= l1)
    
    return vertical_touch or horizontal_touch


def _bbox_contains(outer: Tuple[int, int, int, int], inner: Tuple[int, int, int, int]) -> bool:
    """Check if outer bbox completely contains inner bbox."""
    t1, l1, h1, w1 = outer
    t2, l2, h2, w2 = inner
    
    return (t1 <= t2 and l1 <= l2 and 
            t1 + h1 >= t2 + h2 and l1 + w1 >= l2 + w2)


def sort_objects_by_reading_order(objects: List[Dict[str, Any]]) -> List[int]:
    """
    Sort objects by reading order (top-to-bottom, left-to-right).
    Returns list of indices in sorted order.
    """
    def reading_order_key(idx: int) -> Tuple[int, int]:
        obj = objects[idx]
        centroid_y, centroid_x = obj['centroid']
        return (int(centroid_y), int(centroid_x))
    
    return sorted(range(len(objects)), key=reading_order_key)