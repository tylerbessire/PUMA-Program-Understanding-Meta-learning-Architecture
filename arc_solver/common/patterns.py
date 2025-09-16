"""
Multi-scale pattern detection for ARC grids.

This module provides sophisticated pattern recognition capabilities
that operate at different scales and abstraction levels.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from ..grid import Array
from .objects import connected_components, detect_object_patterns


def detect_multiscale_patterns(grid: Array) -> Dict[str, Any]:
    """
    Detect patterns at multiple scales: pixel, object, and global levels.
    
    Args:
        grid: Input grid to analyze
    
    Returns:
        Dictionary with detected patterns at different scales
    """
    patterns = {
        'pixel_level': detect_pixel_patterns(grid),
        'object_level': detect_object_level_patterns(grid),
        'global_level': detect_global_patterns(grid),
        'repetition_patterns': detect_repetition_patterns(grid),
        'symmetry_patterns': detect_symmetry_patterns(grid)
    }
    
    return patterns


def detect_pixel_patterns(grid: Array) -> Dict[str, Any]:
    """Detect patterns at the pixel level."""
    h, w = grid.shape
    patterns = {
        'stripes': {'horizontal': False, 'vertical': False, 'diagonal': False},
        'checkerboard': False,
        'gradient': False,
        'border_pattern': False,
        'sparse_pattern': False
    }
    
    # Stripe detection
    patterns['stripes']['horizontal'] = _detect_horizontal_stripes(grid)
    patterns['stripes']['vertical'] = _detect_vertical_stripes(grid)
    patterns['stripes']['diagonal'] = _detect_diagonal_stripes(grid)
    
    # Checkerboard pattern
    patterns['checkerboard'] = _detect_checkerboard(grid)
    
    # Gradient pattern
    patterns['gradient'] = _detect_gradient(grid)
    
    # Border pattern
    patterns['border_pattern'] = _detect_border_pattern(grid)
    
    # Sparse pattern (mostly background with few non-background pixels)
    from collections import Counter
    color_counts = Counter(grid.flatten())
    total_pixels = h * w
    max_count = max(color_counts.values())
    patterns['sparse_pattern'] = (max_count / total_pixels) > 0.8
    
    return patterns


def detect_object_level_patterns(grid: Array) -> Dict[str, Any]:
    """Detect patterns at the object level."""
    objects = connected_components(grid)
    
    if not objects:
        return {'num_objects': 0}
    
    # Use existing object pattern detection
    object_patterns = detect_object_patterns(objects)
    
    # Add additional object-level analysis
    patterns = {
        'num_objects': len(objects),
        'size_distribution': _analyze_size_distribution(objects),
        'color_distribution': _analyze_color_distribution(objects),
        'shape_complexity': _analyze_shape_complexity(objects),
        **object_patterns
    }
    
    return patterns


def detect_global_patterns(grid: Array) -> Dict[str, Any]:
    """Detect patterns at the global grid level."""
    h, w = grid.shape
    patterns = {
        'aspect_ratio': w / h,
        'size_category': 'small' if h * w < 50 else 'medium' if h * w < 200 else 'large',
        'is_square': h == w,
        'quadrant_similarity': _detect_quadrant_similarity(grid),
        'central_focus': _detect_central_focus(grid),
        'corner_patterns': _detect_corner_patterns(grid)
    }
    
    return patterns


def detect_repetition_patterns(grid: Array) -> Dict[str, Any]:
    """Detect repeating patterns and tiling."""
    h, w = grid.shape
    patterns = {
        'tiling': None,
        'periodic_rows': False,
        'periodic_cols': False,
        'fractal_like': False
    }
    
    # Find the smallest repeating unit
    smallest_tile = _find_smallest_tile(grid)
    if smallest_tile is not None:
        tile_h, tile_w = smallest_tile
        patterns['tiling'] = {
            'tile_size': (tile_h, tile_w),
            'repetitions': (h // tile_h, w // tile_w)
        }
    
    # Check for periodic rows/columns
    patterns['periodic_rows'] = _detect_periodic_rows(grid)
    patterns['periodic_cols'] = _detect_periodic_cols(grid)
    
    # Simple fractal-like pattern detection
    patterns['fractal_like'] = _detect_fractal_like(grid)
    
    return patterns


def detect_symmetry_patterns(grid: Array) -> Dict[str, Any]:
    """Detect various types of symmetry."""
    patterns = {
        'horizontal_symmetry': np.array_equal(grid, np.flipud(grid)),
        'vertical_symmetry': np.array_equal(grid, np.fliplr(grid)),
        'rotational_90': False,
        'rotational_180': False,
        'point_symmetry': False,
        'partial_symmetries': []
    }
    
    h, w = grid.shape
    
    # Rotational symmetries (only for square grids)
    if h == w:
        patterns['rotational_90'] = np.array_equal(grid, np.rot90(grid, 1))
        patterns['rotational_180'] = np.array_equal(grid, np.rot90(grid, 2))
        patterns['point_symmetry'] = np.array_equal(grid, np.rot90(grid, 2))
    else:
        patterns['rotational_180'] = np.array_equal(grid, np.rot90(grid, 2))
    
    # Partial symmetries
    patterns['partial_symmetries'] = _detect_partial_symmetries(grid)
    
    return patterns


def _detect_horizontal_stripes(grid: Array) -> bool:
    """Detect horizontal stripe patterns."""
    h, w = grid.shape
    if h < 2:
        return False
    
    # Check if alternating rows have the same pattern
    for row_offset in [1, 2]:
        if row_offset >= h:
            continue
        
        stripe_pattern = True
        for y in range(0, h - row_offset, row_offset):
            if not np.array_equal(grid[y], grid[y + row_offset]):
                stripe_pattern = False
                break
        
        if stripe_pattern:
            return True
    
    return False


def _detect_vertical_stripes(grid: Array) -> bool:
    """Detect vertical stripe patterns."""
    h, w = grid.shape
    if w < 2:
        return False
    
    # Check if alternating columns have the same pattern
    for col_offset in [1, 2]:
        if col_offset >= w:
            continue
        
        stripe_pattern = True
        for x in range(0, w - col_offset, col_offset):
            if not np.array_equal(grid[:, x], grid[:, x + col_offset]):
                stripe_pattern = False
                break
        
        if stripe_pattern:
            return True
    
    return False


def _detect_diagonal_stripes(grid: Array) -> bool:
    """Detect diagonal stripe patterns."""
    h, w = grid.shape
    if h < 3 or w < 3:
        return False
    
    # This is a simplified diagonal detection
    # Check main diagonals for patterns
    main_diag = [grid[i, i] for i in range(min(h, w))]
    anti_diag = [grid[i, w - 1 - i] for i in range(min(h, w))]
    
    # Check for alternating patterns
    if len(main_diag) >= 4:
        alternating = all(main_diag[i] == main_diag[i + 2] for i in range(len(main_diag) - 2))
        if alternating:
            return True
    
    return False


def _detect_checkerboard(grid: Array) -> bool:
    """Detect checkerboard patterns."""
    h, w = grid.shape
    if h < 2 or w < 2:
        return False
    
    # Check if (i+j) % 2 determines the color
    colors_even = set()
    colors_odd = set()
    
    for y in range(h):
        for x in range(w):
            if (y + x) % 2 == 0:
                colors_even.add(grid[y, x])
            else:
                colors_odd.add(grid[y, x])
    
    # For a true checkerboard, we should have 1-2 colors in each set
    return len(colors_even) <= 2 and len(colors_odd) <= 2 and colors_even != colors_odd


def _detect_gradient(grid: Array) -> bool:
    """Detect gradient patterns."""
    h, w = grid.shape
    if h < 3 or w < 3:
        return False
    
    # Check for horizontal gradient
    horizontal_grad = True
    for y in range(h):
        row = grid[y, :]
        if not _is_monotonic(row):
            horizontal_grad = False
            break
    
    if horizontal_grad:
        return True
    
    # Check for vertical gradient
    vertical_grad = True
    for x in range(w):
        col = grid[:, x]
        if not _is_monotonic(col):
            vertical_grad = False
            break
    
    return vertical_grad


def _is_monotonic(sequence: np.ndarray) -> bool:
    """Check if a sequence is monotonic (increasing or decreasing)."""
    if len(sequence) < 2:
        return True
    
    increasing = all(sequence[i] <= sequence[i + 1] for i in range(len(sequence) - 1))
    decreasing = all(sequence[i] >= sequence[i + 1] for i in range(len(sequence) - 1))
    
    return increasing or decreasing


def _detect_border_pattern(grid: Array) -> bool:
    """Detect if there's a distinct border pattern."""
    h, w = grid.shape
    if h < 3 or w < 3:
        return False
    
    # Extract border
    border_pixels = []
    # Top and bottom rows
    border_pixels.extend(grid[0, :].tolist())
    border_pixels.extend(grid[h - 1, :].tolist())
    # Left and right columns (excluding corners already counted)
    border_pixels.extend(grid[1:h - 1, 0].tolist())
    border_pixels.extend(grid[1:h - 1, w - 1].tolist())
    
    # Extract interior
    if h > 2 and w > 2:
        interior_pixels = grid[1:h - 1, 1:w - 1].flatten().tolist()
        
        # Check if border and interior have different color distributions
        from collections import Counter
        border_colors = set(border_pixels)
        interior_colors = set(interior_pixels)
        
        return len(border_colors.intersection(interior_colors)) < max(len(border_colors), len(interior_colors))
    
    return False


def _analyze_size_distribution(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of object sizes."""
    if not objects:
        return {}
    
    sizes = [obj['area'] for obj in objects]
    from collections import Counter
    size_counts = Counter(sizes)
    
    return {
        'unique_sizes': len(size_counts),
        'size_range': (min(sizes), max(sizes)),
        'most_common_size': size_counts.most_common(1)[0],
        'uniform_sizes': len(size_counts) == 1
    }


def _analyze_color_distribution(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the distribution of object colors."""
    if not objects:
        return {}
    
    colors = [obj['color'] for obj in objects]
    from collections import Counter
    color_counts = Counter(colors)
    
    return {
        'unique_colors': len(color_counts),
        'most_common_color': color_counts.most_common(1)[0],
        'uniform_colors': len(color_counts) == 1
    }


def _analyze_shape_complexity(objects: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze the complexity of object shapes."""
    if not objects:
        return {}
    
    complexities = []
    for obj in objects:
        # Simple complexity measure: perimeter^2 / area ratio
        area = obj['area']
        perimeter = obj['perimeter']
        
        if area > 0:
            complexity = (perimeter * perimeter) / (4 * np.pi * area)  # Isoperimetric ratio
            complexities.append(complexity)
    
    if complexities:
        return {
            'avg_complexity': np.mean(complexities),
            'max_complexity': max(complexities),
            'complexity_variance': np.var(complexities)
        }
    
    return {}


def _detect_quadrant_similarity(grid: Array) -> Dict[str, bool]:
    """Detect similarity between quadrants of the grid."""
    h, w = grid.shape
    if h < 4 or w < 4:
        return {}
    
    # Divide into quadrants
    mid_h, mid_w = h // 2, w // 2
    
    q1 = grid[:mid_h, :mid_w]  # Top-left
    q2 = grid[:mid_h, mid_w:mid_w + mid_h]  # Top-right
    q3 = grid[mid_h:mid_h + mid_h, :mid_w]  # Bottom-left  
    q4 = grid[mid_h:mid_h + mid_h, mid_w:mid_w + mid_h]  # Bottom-right
    
    return {
        'q1_q2_similar': np.array_equal(q1, q2),
        'q1_q3_similar': np.array_equal(q1, q3),
        'q1_q4_similar': np.array_equal(q1, q4),
        'q2_q4_similar': np.array_equal(q2, q4),
        'q3_q4_similar': np.array_equal(q3, q4),
        'all_quadrants_same': (np.array_equal(q1, q2) and 
                              np.array_equal(q1, q3) and 
                              np.array_equal(q1, q4))
    }


def _detect_central_focus(grid: Array) -> bool:
    """Detect if the pattern has a central focus."""
    h, w = grid.shape
    center_y, center_x = h // 2, w // 2
    
    # Simple check: is the center different from corners?
    if h >= 3 and w >= 3:
        center_color = grid[center_y, center_x]
        corner_colors = [
            grid[0, 0], grid[0, w - 1], 
            grid[h - 1, 0], grid[h - 1, w - 1]
        ]
        
        return center_color not in corner_colors
    
    return False


def _detect_corner_patterns(grid: Array) -> Dict[str, Any]:
    """Detect patterns in the corners."""
    h, w = grid.shape
    if h < 2 or w < 2:
        return {}
    
    corners = {
        'top_left': grid[0, 0],
        'top_right': grid[0, w - 1],
        'bottom_left': grid[h - 1, 0],
        'bottom_right': grid[h - 1, w - 1]
    }
    
    return {
        'all_corners_same': len(set(corners.values())) == 1,
        'opposite_corners_same': (corners['top_left'] == corners['bottom_right'] and
                                corners['top_right'] == corners['bottom_left']),
        'corner_colors': corners
    }


def _find_smallest_tile(grid: Array) -> Optional[Tuple[int, int]]:
    """Find the smallest repeating tile in the grid."""
    h, w = grid.shape
    
    # Try different tile sizes
    for tile_h in range(1, h // 2 + 1):
        for tile_w in range(1, w // 2 + 1):
            if h % tile_h == 0 and w % tile_w == 0:
                # Check if this tile size creates a valid tiling
                tile = grid[:tile_h, :tile_w]
                is_valid_tiling = True
                
                for y in range(0, h, tile_h):
                    for x in range(0, w, tile_w):
                        if not np.array_equal(grid[y:y + tile_h, x:x + tile_w], tile):
                            is_valid_tiling = False
                            break
                    if not is_valid_tiling:
                        break
                
                if is_valid_tiling:
                    return (tile_h, tile_w)
    
    return None


def _detect_periodic_rows(grid: Array) -> bool:
    """Detect if rows follow a periodic pattern."""
    h, w = grid.shape
    if h < 4:
        return False
    
    # Check for period 2
    period_2 = all(np.array_equal(grid[i], grid[i + 2]) 
                  for i in range(h - 2))
    if period_2:
        return True
    
    # Check for period 3
    if h >= 6:
        period_3 = all(np.array_equal(grid[i], grid[i + 3]) 
                      for i in range(h - 3))
        if period_3:
            return True
    
    return False


def _detect_periodic_cols(grid: Array) -> bool:
    """Detect if columns follow a periodic pattern."""
    h, w = grid.shape
    if w < 4:
        return False
    
    # Check for period 2
    period_2 = all(np.array_equal(grid[:, i], grid[:, i + 2]) 
                  for i in range(w - 2))
    if period_2:
        return True
    
    # Check for period 3
    if w >= 6:
        period_3 = all(np.array_equal(grid[:, i], grid[:, i + 3]) 
                      for i in range(w - 3))
        if period_3:
            return True
    
    return False


def _detect_fractal_like(grid: Array) -> bool:
    """Detect fractal-like self-similarity patterns."""
    h, w = grid.shape
    
    # Simple fractal detection: check if subdivisions look similar to the whole
    if h >= 6 and w >= 6 and h % 3 == 0 and w % 3 == 0:
        sub_h, sub_w = h // 3, w // 3
        
        # Extract center subdivision
        center = grid[sub_h:2 * sub_h, sub_w:2 * sub_w]
        
        # Check if it has similar structure to the whole (very simplified)
        # This is a placeholder for more sophisticated fractal analysis
        center_pattern = detect_pixel_patterns(center)
        grid_pattern = detect_pixel_patterns(grid)
        
        # Compare some basic pattern features
        similar_features = 0
        total_features = 0
        
        for key in ['checkerboard', 'gradient', 'border_pattern']:
            total_features += 1
            if center_pattern.get(key) == grid_pattern.get(key):
                similar_features += 1
        
        return similar_features / total_features >= 0.7 if total_features > 0 else False
    
    return False


def _detect_partial_symmetries(grid: Array) -> List[Dict[str, Any]]:
    """Detect partial symmetries in specific regions."""
    h, w = grid.shape
    partial_symmetries = []
    
    # Check symmetry in different regions
    if h >= 4 and w >= 4:
        # Central region
        margin = min(h // 4, w // 4, 2)
        central_region = grid[margin:h - margin, margin:w - margin]
        
        if central_region.shape[0] > 0 and central_region.shape[1] > 0:
            if np.array_equal(central_region, np.fliplr(central_region)):
                partial_symmetries.append({
                    'type': 'vertical_symmetry',
                    'region': 'central',
                    'region_bounds': (margin, margin, h - margin, w - margin)
                })
            
            if np.array_equal(central_region, np.flipud(central_region)):
                partial_symmetries.append({
                    'type': 'horizontal_symmetry',
                    'region': 'central',
                    'region_bounds': (margin, margin, h - margin, w - margin)
                })
    
    return partial_symmetries