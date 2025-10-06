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
        'same_size': [],
        'same_shape': [],
        'mirrored': [],
        'concentric': [],
        'overlapping': [],
        'diagonal_aligned': [],
        'similar_symmetry': []
    }
    
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            obj1, obj2 = objects[i], objects[j]
            
            # Basic relationships
            if obj1['color'] == obj2['color']:
                relationships['same_color'].append((i, j))
            
            if obj1['area'] == obj2['area']:
                relationships['same_size'].append((i, j))
            
            # Shape similarity (simplified)
            if _shapes_similar(obj1['mask'], obj2['mask']):
                relationships['same_shape'].append((i, j))
            
            # Symmetry similarity
            if _symmetries_similar(obj1['symmetries'], obj2['symmetries']):
                relationships['similar_symmetry'].append((i, j))
            
            # Spatial relationships
            c1_y, c1_x = obj1['centroid']
            c2_y, c2_x = obj2['centroid']
            
            # Alignment checks with tolerance
            if abs(c1_y - c2_y) < 1.5:  # Same row (with tolerance)
                relationships['aligned_horizontal'].append((i, j))
            
            if abs(c1_x - c2_x) < 1.5:  # Same column (with tolerance)
                relationships['aligned_vertical'].append((i, j))
            
            # Diagonal alignment
            if abs(abs(c1_y - c2_y) - abs(c1_x - c2_x)) < 1.0:
                relationships['diagonal_aligned'].append((i, j))
            
            # Advanced spatial relationships
            bbox1 = obj1['bbox']
            bbox2 = obj2['bbox']
            
            if _bboxes_adjacent(bbox1, bbox2):
                relationships['adjacent'].append((i, j))
            
            if _bboxes_overlapping(bbox1, bbox2):
                relationships['overlapping'].append((i, j))
            
            if _bbox_contains(bbox1, bbox2):
                relationships['contained'].append((i, j))
            elif _bbox_contains(bbox2, bbox1):
                relationships['contained'].append((j, i))
            
            # Concentric objects (one inside another with similar centers)
            if _objects_concentric(obj1, obj2):
                relationships['concentric'].append((i, j))
            
            # Mirrored objects
            if _objects_mirrored(obj1, obj2):
                relationships['mirrored'].append((i, j))
    
    return relationships


def _shapes_similar(mask1: Array, mask2: Array, threshold: float = 0.8) -> bool:
    """Check if two object masks have similar shapes."""
    if mask1.shape != mask2.shape:
        return False
    
    # Normalize masks to binary
    binary1 = (mask1 > 0).astype(int)
    binary2 = (mask2 > 0).astype(int)
    
    # Calculate overlap
    intersection = np.sum(binary1 & binary2)
    union = np.sum(binary1 | binary2)
    
    if union == 0:
        return True
    
    iou = intersection / union
    return iou >= threshold


def _symmetries_similar(sym1: Dict[str, bool], sym2: Dict[str, bool]) -> bool:
    """Check if two objects have similar symmetry properties."""
    common_symmetries = 0
    total_symmetries = 0
    
    for key in sym1:
        if key in sym2:
            total_symmetries += 1
            if sym1[key] == sym2[key]:
                common_symmetries += 1
    
    return common_symmetries / max(1, total_symmetries) >= 0.7


def _bboxes_overlapping(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """Check if two bounding boxes overlap."""
    t1, l1, h1, w1 = bbox1
    t2, l2, h2, w2 = bbox2
    
    b1, r1 = t1 + h1, l1 + w1
    b2, r2 = t2 + h2, l2 + w2
    
    # Check if they don't overlap
    if r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1:
        return False
    
    return True


def _objects_concentric(obj1: Dict[str, Any], obj2: Dict[str, Any], tolerance: float = 2.0) -> bool:
    """Check if two objects are concentric (similar centers, different sizes)."""
    c1_y, c1_x = obj1['centroid']
    c2_y, c2_x = obj2['centroid']
    
    # Check if centroids are close
    distance = ((c1_y - c2_y) ** 2 + (c1_x - c2_x) ** 2) ** 0.5
    if distance > tolerance:
        return False
    
    # Check if one contains the other
    bbox1 = obj1['bbox']
    bbox2 = obj2['bbox']
    
    return _bbox_contains(bbox1, bbox2) or _bbox_contains(bbox2, bbox1)


def _objects_mirrored(obj1: Dict[str, Any], obj2: Dict[str, Any]) -> bool:
    """Check if two objects are mirror images of each other."""
    # This is a simplified check - could be enhanced
    if obj1['area'] != obj2['area']:
        return False
    
    # Check if they have similar shapes but are positioned symmetrically
    c1_y, c1_x = obj1['centroid']
    c2_y, c2_x = obj2['centroid']
    
    # If they're aligned horizontally, check for vertical mirroring
    if abs(c1_y - c2_y) < 1.0:
        return True
    
    # If they're aligned vertically, check for horizontal mirroring
    if abs(c1_x - c2_x) < 1.0:
        return True
    
    return False


def detect_object_patterns(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect high-level patterns in object arrangements."""
    if not objects:
        return {}
    
    patterns = {
        'grid_arrangement': False,
        'linear_sequence': False,
        'size_progression': False,
        'color_alternation': False,
        'symmetrical_layout': False,
        'clusters': []
    }
    
    # Grid arrangement detection
    if len(objects) >= 4:
        centroids = [(obj['centroid'][0], obj['centroid'][1]) for obj in objects]
        patterns['grid_arrangement'] = _detect_grid_pattern(centroids)
    
    # Linear sequence detection
    if len(objects) >= 3:
        patterns['linear_sequence'] = _detect_linear_sequence(objects)
    
    # Size progression
    sizes = [obj['area'] for obj in objects]
    if len(set(sizes)) > 2:  # Multiple different sizes
        sorted_sizes = sorted(sizes)
        # Check if sizes form a progression
        if len(sorted_sizes) >= 3:
            differences = [sorted_sizes[i+1] - sorted_sizes[i] for i in range(len(sorted_sizes)-1)]
            patterns['size_progression'] = len(set(differences)) <= 2  # Arithmetic or geometric-like
    
    # Color alternation
    if len(objects) >= 4:
        colors = [obj['color'] for obj in objects]
        patterns['color_alternation'] = _detect_color_alternation(colors)
    
    # Symmetrical layout
    patterns['symmetrical_layout'] = _detect_symmetrical_layout(objects)
    
    # Clustering
    patterns['clusters'] = _detect_object_clusters(objects)
    
    return patterns


def _detect_grid_pattern(centroids: List[Tuple[float, float]], tolerance: float = 1.5) -> bool:
    """Detect if objects are arranged in a grid pattern."""
    if len(centroids) < 4:
        return False
    
    # Group by Y coordinate (rows)
    rows = {}
    for y, x in centroids:
        row_key = round(y / tolerance) * tolerance
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append(x)
    
    # Group by X coordinate (columns)  
    cols = {}
    for y, x in centroids:
        col_key = round(x / tolerance) * tolerance
        if col_key not in cols:
            cols[col_key] = []
        cols[col_key].append(y)
    
    # Check if we have regular rows and columns
    return len(rows) >= 2 and len(cols) >= 2


def _detect_linear_sequence(objects: List[Dict[str, Any]], tolerance: float = 1.5) -> bool:
    """Detect if objects form a linear sequence."""
    if len(objects) < 3:
        return False
    
    centroids = [obj['centroid'] for obj in objects]
    
    # Check horizontal alignment
    y_coords = [c[0] for c in centroids]
    if max(y_coords) - min(y_coords) < tolerance:
        return True
    
    # Check vertical alignment
    x_coords = [c[1] for c in centroids]
    if max(x_coords) - min(x_coords) < tolerance:
        return True
    
    # Check diagonal alignment
    # This is simplified - could be more sophisticated
    return False


def _detect_color_alternation(colors: List[int]) -> bool:
    """Detect if colors follow an alternating pattern."""
    if len(colors) < 4:
        return False
    
    # Check for simple AB pattern
    pattern_len = 2
    for start in range(pattern_len):
        pattern = colors[start:start + pattern_len]
        matches = True
        for i in range(start + pattern_len, len(colors), pattern_len):
            if i + pattern_len <= len(colors):
                if colors[i:i + pattern_len] != pattern:
                    matches = False
                    break
        if matches:
            return True
    
    return False


def _detect_symmetrical_layout(objects: List[Dict[str, Any]]) -> bool:
    """Detect if objects are arranged symmetrically."""
    if len(objects) < 2:
        return False
    
    centroids = [obj['centroid'] for obj in objects]
    
    # Check for horizontal symmetry
    center_y = sum(c[0] for c in centroids) / len(centroids)
    
    matched_pairs = 0
    tolerance = 1.5
    
    for i, (y1, x1) in enumerate(centroids):
        for j, (y2, x2) in enumerate(centroids):
            if i >= j:
                continue
            
            # Check if they're symmetric about center_y
            expected_y2 = 2 * center_y - y1
            if abs(y2 - expected_y2) < tolerance and abs(x1 - x2) < tolerance:
                matched_pairs += 1
    
    return matched_pairs >= len(centroids) // 4


def _detect_object_clusters(objects: List[Dict[str, Any]], max_distance: float = 3.0) -> List[List[int]]:
    """Detect clusters of nearby objects."""
    if len(objects) < 2:
        return []
    
    # Simple clustering based on distance
    clusters = []
    used = set()
    
    for i, obj1 in enumerate(objects):
        if i in used:
            continue
        
        cluster = [i]
        used.add(i)
        c1_y, c1_x = obj1['centroid']
        
        for j, obj2 in enumerate(objects):
            if j in used or j == i:
                continue
            
            c2_y, c2_x = obj2['centroid']
            distance = ((c1_y - c2_y) ** 2 + (c1_x - c2_x) ** 2) ** 0.5
            
            if distance <= max_distance:
                cluster.append(j)
                used.add(j)
        
        if len(cluster) > 1:
            clusters.append(cluster)
    
    return clusters


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