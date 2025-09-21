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


def op_tile(a: Array, factor_h: int, factor_w: int) -> Array:
    """Tile the grid by the given factors."""
    return np.tile(a, (factor_h, factor_w))


def op_find_color_region(a: Array, color: int) -> Array:
    """Extract the bounding box of all cells with the specified color."""
    mask = (a == color)
    rows, cols = np.where(mask)
    if len(rows) == 0:
        # No cells with this color found - return 1x1 grid with the color
        return np.array([[color]], dtype=a.dtype)
    
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    return a[top:bottom, left:right].copy()


def op_extract_marked_region(a: Array, marker_color: int = 8) -> Array:
    """Extract region that is marked/surrounded by marker_color."""
    h, w = a.shape
    
    # Find all cells with marker color
    marker_mask = (a == marker_color)
    if not marker_mask.any():
        return a  # No markers found
    
    # Find bounding box of marker region
    marker_rows, marker_cols = np.where(marker_mask)
    top, bottom = marker_rows.min(), marker_rows.max() + 1
    left, right = marker_cols.min(), marker_cols.max() + 1
    
    # Extract the region containing the markers
    region = a[top:bottom, left:right].copy()
    
    # Strategy 1: Look for a rectangular block of the same marker color
    # and extract the equivalent region from the opposite side of the grid
    region_h, region_w = region.shape
    
    # If the marker region is on the left side of the grid, extract from right side
    if left < w // 2:
        # Try to extract equivalent region from right side
        mirror_left = w - right
        mirror_right = w - left
        if mirror_right <= w and mirror_left >= 0:
            mirror_region = a[top:bottom, mirror_left:mirror_right].copy()
            # If the mirror region doesn't contain the marker color, use it
            if not (mirror_region == marker_color).any():
                return mirror_region
    
    # If marker region is on the right side, extract from left side  
    elif left >= w // 2:
        mirror_right = left
        mirror_left = left - region_w
        if mirror_left >= 0 and mirror_right <= w:
            mirror_region = a[top:bottom, mirror_left:mirror_right].copy()
            if not (mirror_region == marker_color).any():
                return mirror_region
    
    # Strategy 2: Extract non-marker content within the bounding box
    non_marker_mask = (region != marker_color)
    if non_marker_mask.any():
        non_marker_rows, non_marker_cols = np.where(non_marker_mask)
        inner_top, inner_bottom = non_marker_rows.min(), non_marker_rows.max() + 1
        inner_left, inner_right = non_marker_cols.min(), non_marker_cols.max() + 1
        inner_region = region[inner_top:inner_bottom, inner_left:inner_right].copy()
        
        # Only return inner region if it's significantly smaller than the marker region
        if inner_region.size < region.size * 0.8:
            return inner_region
    
    # Strategy 3: Remove marker border and extract core
    if region_h > 2 and region_w > 2:
        # Remove border of markers if they form a frame
        core_region = region[1:-1, 1:-1].copy()
        if not (core_region == marker_color).all():
            return core_region
    
    # Fallback: return the region as-is
    return region


def op_smart_crop_auto(a: Array) -> Array:
    """Automatically detect and crop interesting region based on patterns."""
    h, w = a.shape
    
    # Strategy 1: Look for rectangular regions of a distinct color (like color 8)
    for marker_color in [8, 9, 7]:  # Try common marker colors
        if (a == marker_color).any():
            try:
                return op_extract_marked_region(a, marker_color)
            except:
                continue
    
    # Strategy 2: Find the most "interesting" rectangular region
    # Look for regions with diverse colors (not just background)
    bg = bg_color(a)
    
    best_region = a
    best_score = 0
    
    # Try different crop sizes and positions
    for crop_h in range(3, min(15, h)):
        for crop_w in range(3, min(15, w)):
            for top in range(h - crop_h + 1):
                for left in range(w - crop_w + 1):
                    region = a[top:top+crop_h, left:left+crop_w]
                    
                    # Score based on color diversity and non-background content
                    unique_colors = len(np.unique(region))
                    non_bg_ratio = np.sum(region != bg) / region.size
                    score = unique_colors * non_bg_ratio
                    
                    if score > best_score:
                        best_score = score
                        best_region = region
    
    return best_region


def op_extract_symmetric_region(a: Array) -> Array:
    """Extract interesting region by looking for symmetric patterns."""
    h, w = a.shape
    
    # Look for vertical symmetry - if left and right sides are mirrors, 
    # extract the middle region that breaks the symmetry
    if w >= 6:
        mid = w // 2
        left_side = a[:, :mid]
        right_side = a[:, -mid:]
        
        # Check if sides are horizontally mirrored
        if np.array_equal(left_side, np.fliplr(right_side)):
            # Extract the middle region
            middle_region = a[:, mid-1:mid+2] if mid > 0 else a[:, mid:mid+1]
            return middle_region
    
    # Look for horizontal symmetry
    if h >= 6:
        mid = h // 2
        top_side = a[:mid, :]
        bottom_side = a[-mid:, :]
        
        if np.array_equal(top_side, np.flipud(bottom_side)):
            # Extract middle region
            middle_region = a[mid-1:mid+2, :] if mid > 0 else a[mid:mid+1, :]
            return middle_region
    
    # Look for quadrant patterns
    if h >= 4 and w >= 4:
        mid_h, mid_w = h // 2, w // 2
        top_left = a[:mid_h, :mid_w]
        top_right = a[:mid_h, -mid_w:]
        bottom_left = a[-mid_h:, :mid_w] 
        bottom_right = a[-mid_h:, -mid_w:]
        
        # If three quadrants are similar, extract the different one
        quadrants = [top_left, top_right, bottom_left, bottom_right]
        for i, quad in enumerate(quadrants):
            others = [q for j, q in enumerate(quadrants) if j != i]
            if all(np.array_equal(quad, other) for other in others):
                # This quadrant is different, but we want to return one of the identical ones
                return others[0]
    
    # Fallback to the entire array
    return a


def op_extract_pattern_region(a: Array, marker_color: int = 8) -> Array:
    """Advanced pattern extraction using multiple strategies to find the hidden content."""
    h, w = a.shape
    
    # Find marker region
    marker_mask = (a == marker_color)
    if not marker_mask.any():
        return a
    
    marker_rows, marker_cols = np.where(marker_mask)
    top, bottom = marker_rows.min(), marker_rows.max() + 1
    left, right = marker_cols.min(), marker_cols.max() + 1
    region_h, region_w = bottom - top, right - left
    
    # Strategy: The pattern might be "underneath" the markers in a logical sense
    # Look for the pattern that would be there if markers weren't overlaid
    
    # Create a version without markers and try to extract the same region
    strategies = []
    
    # Strategy 1: Try to reconstruct by looking at symmetric pattern
    # If this is tile-based, find the corresponding region in other tiles
    for tile_h in [10, 15]:
        for tile_w in [10, 15]:
            if h % tile_h == 0 and w % tile_w == 0:
                tiles_v, tiles_h = h // tile_h, w // tile_w
                marker_tile_row = top // tile_h
                marker_tile_col = left // tile_w
                
                in_tile_top = top % tile_h
                in_tile_left = left % tile_w
                in_tile_bottom = in_tile_top + region_h
                in_tile_right = in_tile_left + region_w
                
                if (in_tile_bottom <= tile_h and in_tile_right <= tile_w):
                    # Look for the same region in other tiles
                    for tr in range(tiles_v):
                        for tc in range(tiles_h):
                            if tr == marker_tile_row and tc == marker_tile_col:
                                continue
                            
                            tile_start_row = tr * tile_h
                            tile_start_col = tc * tile_w
                            
                            candidate = a[tile_start_row + in_tile_top:tile_start_row + in_tile_bottom,
                                        tile_start_col + in_tile_left:tile_start_col + in_tile_right]
                            
                            if not (candidate == marker_color).any() and len(np.unique(candidate)) > 1:
                                strategies.append((f'tile_{tr}_{tc}', candidate.copy()))
    
    # Strategy 2: Pattern reconstruction based on grid structure
    # Look for vertical/horizontal symmetries and extract the "clean" version
    
    # Try vertical sections
    section_width = region_w
    for start_col in range(0, w - section_width + 1, section_width):
        if start_col == left:  # Skip the marker column
            continue
        candidate = a[top:bottom, start_col:start_col + section_width]
        if not (candidate == marker_color).any() and len(np.unique(candidate)) > 1:
            strategies.append((f'vertical_section_{start_col}', candidate.copy()))
    
    # Try horizontal sections
    section_height = region_h
    for start_row in range(0, h - section_height + 1, section_height):
        if start_row == top:  # Skip the marker row
            continue
        candidate = a[start_row:start_row + section_height, left:right]
        if not (candidate == marker_color).any() and len(np.unique(candidate)) > 1:
            strategies.append((f'horizontal_section_{start_row}', candidate.copy()))
    
    # If we have strategies, return the most "diverse" one (most unique values)
    if strategies:
        best_strategy = max(strategies, key=lambda x: len(np.unique(x[1])))
        return best_strategy[1]
    
    # Fallback: return marker region
    return a[top:bottom, left:right]


def op_extract_content_region(a: Array) -> Array:
    """Extract the most content-dense rectangular region."""
    h, w = a.shape
    
    if h <= 2 or w <= 2:
        return a
    
    # Find background color (most frequent)
    bg_color = np.argmax(np.bincount(a.flatten()))
    
    # Create a density map - count non-background pixels in sliding windows
    best_region = a
    best_density = 0
    
    # Try different region sizes (prefer smaller regions with high content density)
    for region_h in range(3, min(15, h + 1)):
        for region_w in range(3, min(15, w + 1)):
            for top in range(h - region_h + 1):
                for left in range(w - region_w + 1):
                    region = a[top:top+region_h, left:left+region_w]
                    
                    # Calculate content density (non-background pixels)
                    non_bg_pixels = np.sum(region != bg_color)
                    total_pixels = region.size
                    density = non_bg_pixels / total_pixels
                    
                    # Prefer high density and diverse colors
                    color_diversity = len(np.unique(region))
                    score = density * color_diversity
                    
                    if score > best_density:
                        best_density = score
                        best_region = region
    
    return best_region


def op_extract_bounded_region(a: Array, boundary_color: int = 8) -> Array:
    """Extract region bounded by a specific color (default: 8)."""
    h, w = a.shape
    
    # Find all positions of the boundary color
    boundary_positions = np.where(a == boundary_color)
    
    if len(boundary_positions[0]) == 0:
        # No boundary found, fallback to content region
        return op_extract_content_region(a)
    
    # Find bounding box of boundary
    min_row, max_row = boundary_positions[0].min(), boundary_positions[0].max()
    min_col, max_col = boundary_positions[1].min(), boundary_positions[1].max()
    
    # Extract region inside the boundary
    if min_row < max_row and min_col < max_col:
        inner_region = a[min_row+1:max_row, min_col+1:max_col]
        if inner_region.size > 0:
            return inner_region
    
    # Fallback: extract the bounded region itself
    return a[min_row:max_row+1, min_col:max_col+1]


def op_extract_largest_rect(a: Array, ignore_color: Optional[int] = None) -> Array:
    """Extract the largest rectangle of non-ignore color."""
    h, w = a.shape
    
    # Auto-detect ignore color if not specified (most frequent)
    if ignore_color is None:
        ignore_color = np.argmax(np.bincount(a.flatten()))
    
    # Create binary mask (1 for content, 0 for ignored)
    mask = (a != ignore_color).astype(int)
    
    # Find the largest rectangle in the mask using histogram method
    max_area = 0
    best_region = a
    
    # For each row, find the largest rectangle in histogram
    heights = np.zeros(w, dtype=int)
    
    for row in range(h):
        # Update heights for current row
        for col in range(w):
            if mask[row, col] == 0:
                heights[col] = 0
            else:
                heights[col] += 1
        
        # Find largest rectangle in current histogram
        area, left, right, height = _largest_rectangle_in_histogram(heights)
        
        if area > max_area:
            max_area = area
            # Calculate the actual region
            top = row - height + 1
            bottom = row + 1
            best_region = a[top:bottom, left:right]
    
    return best_region if max_area > 0 else a


def _largest_rectangle_in_histogram(heights):
    """Find the largest rectangle in a histogram."""
    stack = []
    max_area = 0
    best_left, best_right, best_height = 0, 0, 0
    
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            area = height * (i - index)
            if area > max_area:
                max_area = area
                best_left, best_right, best_height = index, i, height
            start = index
        stack.append((start, h))
    
    # Process remaining stack
    for index, height in stack:
        area = height * (len(heights) - index)
        if area > max_area:
            max_area = area
            best_left, best_right, best_height = index, len(heights), height
    
    return max_area, best_left, best_right, best_height


def op_extract_central_pattern(a: Array) -> Array:
    """Extract central pattern by removing uniform border regions."""
    h, w = a.shape
    
    if h <= 4 or w <= 4:
        return a
    
    # Find the tightest crop that removes uniform borders
    # Start from outside and work inward
    
    top, bottom, left, right = 0, h, 0, w
    
    # Remove uniform top rows
    for row in range(h // 3):  # Don't remove more than 1/3
        if len(set(a[row, :])) <= 2:  # Nearly uniform
            top = row + 1
        else:
            break
    
    # Remove uniform bottom rows
    for row in range(h - 1, max(top, h * 2 // 3) - 1, -1):
        if len(set(a[row, :])) <= 2:
            bottom = row
        else:
            break
    
    # Remove uniform left columns
    for col in range(w // 3):
        if len(set(a[top:bottom, col])) <= 2:
            left = col + 1
        else:
            break
    
    # Remove uniform right columns
    for col in range(w - 1, max(left, w * 2 // 3) - 1, -1):
        if len(set(a[top:bottom, col])) <= 2:
            right = col
        else:
            break
    
    # Extract the central pattern
    if top < bottom and left < right:
        return a[top:bottom, left:right]
    
    return a


def op_extract_pattern_blocks(a: Array, block_size: int = 4) -> Array:
    """Extract rectangular blocks based on internal pattern analysis."""
    h, w = a.shape
    
    if h < block_size or w < block_size:
        return a
    
    # Try to find repeating patterns or structured regions
    best_region = a
    best_score = -1
    
    # Test different block sizes around the given size
    for test_h in range(max(3, block_size-2), min(15, block_size+3)):
        for test_w in range(max(3, block_size-2), min(15, block_size+3)):
            for top in range(0, h - test_h + 1, 2):  # Skip by 2 for efficiency
                for left in range(0, w - test_w + 1, 2):
                    region = a[top:top+test_h, left:left+test_w]
                    
                    # Score based on:
                    # 1. Pattern regularity (how structured the region looks)
                    # 2. Color distribution balance
                    # 3. Avoid too much background
                    
                    colors, counts = np.unique(region, return_counts=True)
                    
                    # Avoid regions dominated by a single color
                    max_color_ratio = np.max(counts) / region.size
                    if max_color_ratio > 0.7:
                        continue
                    
                    # Prefer regions with 3-6 distinct colors (good for patterns)
                    color_diversity = len(colors)
                    if color_diversity < 2:
                        continue
                        
                    diversity_score = min(color_diversity / 5.0, 1.0)
                    balance_score = 1.0 - max_color_ratio
                    
                    # Bonus for having expected dimensions (rough heuristic)
                    size_bonus = 0
                    if 3 <= test_h <= 10 and 3 <= test_w <= 10:
                        size_bonus = 0.2
                    
                    score = diversity_score * balance_score + size_bonus
                    
                    if score > best_score:
                        best_score = score
                        best_region = region
    
    return best_region


def op_extract_distinct_regions(a: Array) -> Array:
    """Extract regions that stand out from background patterns."""
    h, w = a.shape
    
    if h <= 6 or w <= 6:
        return a
    
    # Identify background by looking at border pixels
    border_pixels = np.concatenate([
        a[0, :], a[-1, :], a[:, 0], a[:, -1]
    ])
    bg_candidates = np.unique(border_pixels)
    
    best_region = a
    best_score = 0
    
    # Try different region extractions
    for region_h in range(4, min(12, h)):
        for region_w in range(4, min(12, w)):
            for top in range(h - region_h + 1):
                for left in range(w - region_w + 1):
                    region = a[top:top+region_h, left:left+region_w]
                    
                    # Score based on how different this region is from background
                    non_bg_pixels = 0
                    total_pixels = region.size
                    
                    for bg_candidate in bg_candidates:
                        non_bg_pixels = max(non_bg_pixels, 
                                          np.sum(region != bg_candidate))
                    
                    distinctiveness = non_bg_pixels / total_pixels
                    color_count = len(np.unique(region))
                    
                    # Prefer regions that are distinctive and have multiple colors
                    score = distinctiveness * (color_count / 6.0)
                    
                    if score > best_score:
                        best_score = score
                        best_region = region
    
    return best_region


def op_human_spatial_reasoning(a: Array, hypothesis_name: str = "", 
                              hypothesis_id: int = 0, confidence: float = 1.0,
                              verification_score: float = 1.0) -> Array:
    """Apply human-grade spatial reasoning to solve the transformation."""
    # This is a placeholder that will be handled specially by the enhanced search
    # The actual reasoning is done in the HumanGradeReasoner class
    return a  # Will be replaced by the actual hypothesis result


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
    "tile": Op("tile", op_tile, 1, ["factor_h", "factor_w"]),
    "find_color_region": Op("find_color_region", op_find_color_region, 1, ["color"]),
    "extract_marked_region": Op("extract_marked_region", op_extract_marked_region, 1, ["marker_color"]),
    "smart_crop_auto": Op("smart_crop_auto", op_smart_crop_auto, 1, []),
    "extract_symmetric_region": Op("extract_symmetric_region", op_extract_symmetric_region, 1, []),
    "extract_pattern_region": Op("extract_pattern_region", op_extract_pattern_region, 1, ["marker_color"]),
    "extract_content_region": Op("extract_content_region", op_extract_content_region, 1, []),
    "extract_bounded_region": Op("extract_bounded_region", op_extract_bounded_region, 1, ["boundary_color"]),
    "extract_largest_rect": Op("extract_largest_rect", op_extract_largest_rect, 1, ["ignore_color"]),
    "extract_central_pattern": Op("extract_central_pattern", op_extract_central_pattern, 1, []),
    "extract_pattern_blocks": Op("extract_pattern_blocks", op_extract_pattern_blocks, 1, ["block_size"]),
    "extract_distinct_regions": Op("extract_distinct_regions", op_extract_distinct_regions, 1, []),
    "human_spatial_reasoning": Op("human_spatial_reasoning", op_human_spatial_reasoning, 1, 
                                  ["hypothesis_name", "hypothesis_id", "confidence", "verification_score"]),
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
    # Check if this is a human reasoning operation with metadata
    if params.get('_source') == 'human_reasoner':
        # Get the stored hypothesis and apply it directly
        hypothesis = params.get('_hypothesis_obj')
        if hypothesis:
            return hypothesis.construction_rule(a)
        else:
            # Fallback - treat as identity
            return a
    
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
