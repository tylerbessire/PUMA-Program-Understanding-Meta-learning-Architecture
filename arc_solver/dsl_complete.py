"""
COMPLETE Enhanced DSL with ALL operations for ARC solving.

This module implements a comprehensive domain-specific language with ALL
transformation primitives needed for ARC tasks, including pattern completion,
object manipulation, conditional logic, and advanced spatial reasoning.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from collections import defaultdict
from scipy import ndimage
import itertools

Array = np.ndarray


# ================== CORE GRID OPERATIONS ==================

def identity(grid: Array) -> Array:
    """Return unchanged grid."""
    return grid.copy()


def rotate(grid: Array, k: int = 1) -> Array:
    """Rotate grid k*90 degrees clockwise."""
    return np.rot90(grid, -k % 4)


def flip(grid: Array, axis: int = 0) -> Array:
    """Flip grid along axis (0=vertical, 1=horizontal)."""
    return np.flip(grid, axis=axis)


def transpose(grid: Array) -> Array:
    """Transpose grid (swap rows and columns)."""
    return grid.T


def translate(grid: Array, dx: int = 0, dy: int = 0, fill_value: int = 0) -> Array:
    """Translate grid by (dx, dy) with wraparound or filling."""
    H, W = grid.shape
    result = np.full_like(grid, fill_value)
    
    # Source bounds
    src_y_start = max(0, -dy)
    src_y_end = min(H, H - dy)
    src_x_start = max(0, -dx) 
    src_x_end = min(W, W - dx)
    
    # Destination bounds
    dst_y_start = max(0, dy)
    dst_y_end = min(H, H + dy)
    dst_x_start = max(0, dx)
    dst_x_end = min(W, W + dx)
    
    # Copy valid region
    result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        grid[src_y_start:src_y_end, src_x_start:src_x_end]
    
    return result


def crop(grid: Array, top: int, bottom: int, left: int, right: int) -> Array:
    """Crop grid to specified bounds."""
    return grid[top:bottom, left:right].copy()


def pad(grid: Array, top: int = 0, bottom: int = 0, left: int = 0, right: int = 0, 
        fill_value: int = 0) -> Array:
    """Pad grid with fill_value."""
    return np.pad(grid, ((top, bottom), (left, right)), constant_values=fill_value)


def resize(grid: Array, new_height: int, new_width: int, method: str = 'nearest') -> Array:
    """Resize grid to new dimensions."""
    if method == 'nearest':
        # Simple nearest neighbor scaling
        y_scale = new_height / grid.shape[0]
        x_scale = new_width / grid.shape[1]
        
        result = np.zeros((new_height, new_width), dtype=grid.dtype)
        for i in range(new_height):
            for j in range(new_width):
                src_i = int(i / y_scale)
                src_j = int(j / x_scale)
                result[i, j] = grid[src_i, src_j]
        return result
    elif method == 'replicate':
        # Replicate each cell
        return np.repeat(np.repeat(grid, new_height // grid.shape[0], axis=0), 
                        new_width // grid.shape[1], axis=1)
    else:
        return grid.copy()


# ================== COLOR OPERATIONS ==================

def recolor(grid: Array, color_map: Dict[int, int]) -> Array:
    """Recolor grid according to color mapping."""
    result = grid.copy()
    for old_color, new_color in color_map.items():
        result[grid == old_color] = new_color
    return result


def recolor_by_position(grid: Array, position_map: Dict[Tuple[int, int], int]) -> Array:
    """Recolor specific positions."""
    result = grid.copy()
    for (y, x), color in position_map.items():
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            result[y, x] = color
    return result


def swap_colors(grid: Array, color1: int, color2: int) -> Array:
    """Swap two colors in the grid."""
    result = grid.copy()
    mask1 = grid == color1
    mask2 = grid == color2
    result[mask1] = color2
    result[mask2] = color1
    return result


def dominant_color_recolor(grid: Array, target_color: int) -> Array:
    """Recolor most frequent non-background color to target."""
    counts = np.bincount(grid.flatten())
    # Assume 0 is background, find most common non-zero
    non_zero_counts = counts[1:] if len(counts) > 1 else []
    if len(non_zero_counts) > 0:
        dominant = np.argmax(non_zero_counts) + 1
        return recolor(grid, {dominant: target_color})
    return grid.copy()


def gradient_recolor(grid: Array, start_color: int, end_color: int, direction: str = 'horizontal') -> Array:
    """Apply color gradient across grid."""
    result = grid.copy()
    H, W = grid.shape
    
    if direction == 'horizontal':
        for x in range(W):
            ratio = x / max(1, W - 1)
            new_color = int(start_color + ratio * (end_color - start_color))
            mask = grid[:, x] != 0  # Don't recolor background
            result[mask, x] = new_color
    elif direction == 'vertical':
        for y in range(H):
            ratio = y / max(1, H - 1)
            new_color = int(start_color + ratio * (end_color - start_color))
            mask = grid[y, :] != 0
            result[y, mask] = new_color
    
    return result


# ================== OBJECT DETECTION AND MANIPULATION ==================

def get_connected_components(grid: Array, connectivity: int = 4) -> List[Tuple[Set[Tuple[int, int]], int]]:
    """Get connected components with their colors."""
    components = []
    visited = np.zeros_like(grid, dtype=bool)
    H, W = grid.shape
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if connectivity == 8:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    
    def flood_fill(start_y, start_x, color):
        stack = [(start_y, start_x)]
        component = set()
        
        while stack:
            y, x = stack.pop()
            if (y, x) in component or visited[y, x]:
                continue
                
            visited[y, x] = True
            component.add((y, x))
            
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if (0 <= ny < H and 0 <= nx < W and 
                    not visited[ny, nx] and grid[ny, nx] == color):
                    stack.append((ny, nx))
        
        return component
    
    for y in range(H):
        for x in range(W):
            if not visited[y, x] and grid[y, x] != 0:  # Skip background
                component = flood_fill(y, x, grid[y, x])
                if component:
                    components.append((component, grid[y, x]))
    
    return components


def extract_largest_object(grid: Array) -> Array:
    """Extract the largest connected object."""
    components = get_connected_components(grid)
    if not components:
        return grid.copy()
    
    largest = max(components, key=lambda x: len(x[0]))
    result = np.zeros_like(grid)
    
    for y, x in largest[0]:
        result[y, x] = largest[1]
    
    return result


def move_object(grid: Array, from_color: int, dx: int, dy: int) -> Array:
    """Move all pixels of a specific color."""
    result = grid.copy()
    mask = grid == from_color
    result[mask] = 0  # Clear original positions
    
    # Move to new positions
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            if mask[y, x]:
                new_y, new_x = y + dy, x + dx
                if 0 <= new_y < H and 0 <= new_x < W:
                    result[new_y, new_x] = from_color
    
    return result


def scale_object(grid: Array, target_color: int, scale_factor: int) -> Array:
    """Scale up an object by replicating pixels."""
    if scale_factor <= 1:
        return grid.copy()
    
    H, W = grid.shape
    new_H, new_W = H * scale_factor, W * scale_factor
    result = np.zeros((new_H, new_W), dtype=grid.dtype)
    
    for y in range(H):
        for x in range(W):
            if grid[y, x] == target_color:
                # Fill scale_factor x scale_factor block
                for dy in range(scale_factor):
                    for dx in range(scale_factor):
                        result[y * scale_factor + dy, x * scale_factor + dx] = target_color
            elif grid[y, x] != 0:
                # Keep other objects at original positions
                result[y, x] = grid[y, x]
    
    return result


def remove_object(grid: Array, target_color: int) -> Array:
    """Remove all pixels of target color."""
    result = grid.copy()
    result[grid == target_color] = 0
    return result


def duplicate_object(grid: Array, target_color: int, offset_y: int, offset_x: int) -> Array:
    """Duplicate an object at a new position."""
    result = grid.copy()
    H, W = grid.shape
    
    for y in range(H):
        for x in range(W):
            if grid[y, x] == target_color:
                new_y, new_x = y + offset_y, x + offset_x
                if 0 <= new_y < H and 0 <= new_x < W:
                    result[new_y, new_x] = target_color
    
    return result


# ================== PATTERN OPERATIONS ==================

def fill_rectangle(grid: Array, top: int, left: int, height: int, width: int, 
                  color: int) -> Array:
    """Fill rectangular region with color."""
    result = grid.copy()
    bottom = min(top + height, grid.shape[0])
    right = min(left + width, grid.shape[1])
    result[top:bottom, left:right] = color
    return result


def draw_line(grid: Array, start_y: int, start_x: int, end_y: int, end_x: int, 
              color: int) -> Array:
    """Draw line between two points."""
    result = grid.copy()
    
    # Bresenham's line algorithm
    dx = abs(end_x - start_x)
    dy = abs(end_y - start_y)
    x_step = 1 if start_x < end_x else -1
    y_step = 1 if start_y < end_y else -1
    
    x, y = start_x, start_y
    error = dx - dy
    
    while True:
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            result[y, x] = color
        
        if x == end_x and y == end_y:
            break
            
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x += x_step
        if e2 < dx:
            error += dx
            y += y_step
    
    return result


def complete_symmetry(grid: Array, axis: str = 'vertical') -> Array:
    """Complete symmetric patterns."""
    result = grid.copy()
    H, W = grid.shape
    
    if axis == 'vertical':
        # Mirror left half to right half
        mid = W // 2
        for y in range(H):
            for x in range(mid):
                if grid[y, x] != 0:  # Don't overwrite with background
                    mirror_x = W - 1 - x
                    if mirror_x < W:
                        result[y, mirror_x] = grid[y, x]
    elif axis == 'horizontal':
        # Mirror top half to bottom half
        mid = H // 2
        for y in range(mid):
            for x in range(W):
                if grid[y, x] != 0:
                    mirror_y = H - 1 - y
                    if mirror_y < H:
                        result[mirror_y, x] = grid[y, x]
    
    return result


def create_grid_pattern(grid: Array, pattern_height: int, pattern_width: int) -> Array:
    """Tile a pattern across the grid."""
    H, W = grid.shape
    
    # Extract pattern from top-left
    pattern = grid[:pattern_height, :pattern_width].copy()
    result = np.zeros_like(grid)
    
    # Tile pattern
    for y in range(0, H, pattern_height):
        for x in range(0, W, pattern_width):
            end_y = min(y + pattern_height, H)
            end_x = min(x + pattern_width, W)
            pattern_end_y = end_y - y
            pattern_end_x = end_x - x
            result[y:end_y, x:end_x] = pattern[:pattern_end_y, :pattern_end_x]
    
    return result


def flood_fill(grid: Array, start_y: int, start_x: int, new_color: int) -> Array:
    """Flood fill from starting position."""
    result = grid.copy()
    if not (0 <= start_y < grid.shape[0] and 0 <= start_x < grid.shape[1]):
        return result
    
    original_color = grid[start_y, start_x]
    if original_color == new_color:
        return result
    
    stack = [(start_y, start_x)]
    H, W = grid.shape
    
    while stack:
        y, x = stack.pop()
        if (not (0 <= y < H and 0 <= x < W) or 
            result[y, x] != original_color):
            continue
        
        result[y, x] = new_color
        
        # Add neighbors
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            stack.append((y + dy, x + dx))
    
    return result


def apply_gravity(grid: Array, direction: str = 'down') -> Array:
    """Apply gravity in specified direction."""
    result = np.zeros_like(grid)
    H, W = grid.shape
    
    if direction == 'down':
        for x in range(W):
            objects = []
            for y in range(H):
                if grid[y, x] != 0:
                    objects.append(grid[y, x])
            
            # Place objects at bottom
            for i, obj in enumerate(objects):
                result[H - 1 - i, x] = obj
                
    elif direction == 'up':
        for x in range(W):
            objects = []
            for y in range(H):
                if grid[y, x] != 0:
                    objects.append(grid[y, x])
            
            # Place objects at top
            for i, obj in enumerate(objects):
                result[i, x] = obj
                
    elif direction == 'left':
        for y in range(H):
            objects = []
            for x in range(W):
                if grid[y, x] != 0:
                    objects.append(grid[y, x])
            
            # Place objects at left
            for i, obj in enumerate(objects):
                result[y, i] = obj
                
    elif direction == 'right':
        for y in range(H):
            objects = []
            for x in range(W):
                if grid[y, x] != 0:
                    objects.append(grid[y, x])
            
            # Place objects at right
            for i, obj in enumerate(objects):
                result[y, W - 1 - i] = obj
    
    return result


# ================== CONDITIONAL OPERATIONS ==================

def conditional_recolor(grid: Array, condition_color: int, target_color: int, 
                       new_color: int) -> Array:
    """Recolor target_color to new_color only if condition_color is present."""
    if np.any(grid == condition_color):
        return recolor(grid, {target_color: new_color})
    return grid.copy()


def conditional_transform(grid: Array, condition: Callable[[Array], bool], 
                         transform: Callable[[Array], Array]) -> Array:
    """Apply transform only if condition is met."""
    if condition(grid):
        return transform(grid)
    return grid.copy()


def replace_pattern(grid: Array, pattern: Array, replacement: Array) -> Array:
    """Replace all occurrences of pattern with replacement."""
    result = grid.copy()
    pH, pW = pattern.shape
    rH, rW = replacement.shape
    H, W = grid.shape
    
    for y in range(H - pH + 1):
        for x in range(W - pW + 1):
            if np.array_equal(grid[y:y+pH, x:x+pW], pattern):
                # Replace with new pattern
                end_y = min(y + rH, H)
                end_x = min(x + rW, W)
                result[y:end_y, x:end_x] = replacement[:end_y-y, :end_x-x]
    
    return result


# ================== ADVANCED OPERATIONS ==================

def connect_objects(grid: Array, color1: int, color2: int, line_color: int) -> Array:
    """Draw line connecting nearest points of two colored objects."""
    result = grid.copy()
    
    # Find positions of each color
    pos1 = np.argwhere(grid == color1)
    pos2 = np.argwhere(grid == color2)
    
    if len(pos1) == 0 or len(pos2) == 0:
        return result
    
    # Find closest pair
    min_dist = float('inf')
    best_pair = None
    
    for p1 in pos1:
        for p2 in pos2:
            dist = np.sum((p1 - p2) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_pair = (p1, p2)
    
    if best_pair:
        p1, p2 = best_pair
        result = draw_line(result, p1[0], p1[1], p2[0], p2[1], line_color)
    
    return result


def outline_objects(grid: Array, outline_color: int) -> Array:
    """Add outline around all non-background objects."""
    result = grid.copy()
    H, W = grid.shape
    
    for y in range(H):
        for x in range(W):
            if grid[y, x] != 0:  # Non-background
                # Check if it's on the border of the object
                is_border = False
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if (ny < 0 or ny >= H or nx < 0 or nx >= W or 
                        grid[ny, nx] == 0):
                        is_border = True
                        break
                
                if is_border:
                    # Add outline around this pixel
                    for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < H and 0 <= nx < W and 
                            grid[ny, nx] == 0):
                            result[ny, nx] = outline_color
    
    return result


def mirror_complete(grid: Array) -> Array:
    """Complete partial mirror patterns."""
    result = grid.copy()
    H, W = grid.shape
    
    # Try vertical mirror
    left_half = grid[:, :W//2]
    right_half = grid[:, W//2:]
    
    if W % 2 == 0:
        # Even width - direct mirror
        right_mirror = np.fliplr(left_half)
        # Fill missing parts
        mask = right_half == 0
        result[:, W//2:] = np.where(mask, right_mirror, right_half)
    
    # Try horizontal mirror  
    top_half = grid[:H//2, :]
    bottom_half = grid[H//2:, :]
    
    if H % 2 == 0:
        bottom_mirror = np.flipud(top_half)
        mask = bottom_half == 0
        result[H//2:, :] = np.where(mask, bottom_mirror, bottom_half)
    
    return result


def sort_colors_by_position(grid: Array, direction: str = 'horizontal') -> Array:
    """Sort colors by their position."""
    result = grid.copy()
    H, W = grid.shape
    
    if direction == 'horizontal':
        for y in range(H):
            row = grid[y, :].copy()
            non_zero = row[row != 0]
            if len(non_zero) > 0:
                non_zero.sort()
                result[y, :] = 0
                result[y, :len(non_zero)] = non_zero
    elif direction == 'vertical':
        for x in range(W):
            col = grid[:, x].copy()
            non_zero = col[col != 0]
            if len(non_zero) > 0:
                non_zero.sort()
                result[:, x] = 0
                result[:len(non_zero), x] = non_zero
    
    return result


def repeat_pattern(grid: Array, repetitions: int, direction: str = 'right') -> Array:
    """Repeat the entire grid pattern multiple times."""
    H, W = grid.shape
    
    if direction == 'right':
        result = np.zeros((H, W * repetitions), dtype=grid.dtype)
        for i in range(repetitions):
            result[:, i*W:(i+1)*W] = grid
    elif direction == 'down':
        result = np.zeros((H * repetitions, W), dtype=grid.dtype)
        for i in range(repetitions):
            result[i*H:(i+1)*H, :] = grid
    else:
        result = grid.copy()
    
    return result


# ================== OPERATION REGISTRY ==================

OPS = {
    # Basic transformations
    'identity': identity,
    'rotate': rotate,
    'flip': flip,
    'transpose': transpose,
    'translate': translate,
    'crop': crop,
    'pad': pad,
    'resize': resize,
    
    # Color operations
    'recolor': recolor,
    'recolor_by_position': recolor_by_position,
    'swap_colors': swap_colors,
    'dominant_color_recolor': dominant_color_recolor,
    'gradient_recolor': gradient_recolor,
    
    # Object operations
    'extract_largest_object': extract_largest_object,
    'move_object': move_object,
    'scale_object': scale_object,
    'remove_object': remove_object,
    'duplicate_object': duplicate_object,
    
    # Pattern operations
    'fill_rectangle': fill_rectangle,
    'draw_line': draw_line,
    'complete_symmetry': complete_symmetry,
    'create_grid_pattern': create_grid_pattern,
    'flood_fill': flood_fill,
    'apply_gravity': apply_gravity,
    
    # Conditional operations
    'conditional_recolor': conditional_recolor,
    'conditional_transform': conditional_transform,
    'replace_pattern': replace_pattern,
    
    # Advanced operations
    'connect_objects': connect_objects,
    'outline_objects': outline_objects,
    'mirror_complete': mirror_complete,
    'sort_colors_by_position': sort_colors_by_position,
    'repeat_pattern': repeat_pattern,
}


def apply_program(grid: Array, program: List[Tuple[str, Dict[str, Any]]]) -> Array:
    """Apply a sequence of operations to a grid."""
    result = grid.copy()
    
    for op_name, params in program:
        if op_name in OPS:
            try:
                result = OPS[op_name](result, **params)
            except Exception as e:
                # If operation fails, continue with current result
                pass
        elif op_name == 'identity':
            # Handle identity specially
            pass
    
    return result


def get_operation_signatures() -> Dict[str, List[str]]:
    """Get parameter signatures for all operations."""
    signatures = {
        'identity': [],
        'rotate': ['k'],
        'flip': ['axis'],
        'transpose': [],
        'translate': ['dx', 'dy', 'fill_value'],
        'crop': ['top', 'bottom', 'left', 'right'],
        'pad': ['top', 'bottom', 'left', 'right', 'fill_value'],
        'resize': ['new_height', 'new_width', 'method'],
        'recolor': ['color_map'],
        'recolor_by_position': ['position_map'],
        'swap_colors': ['color1', 'color2'],
        'dominant_color_recolor': ['target_color'],
        'gradient_recolor': ['start_color', 'end_color', 'direction'],
        'extract_largest_object': [],
        'move_object': ['from_color', 'dx', 'dy'],
        'scale_object': ['target_color', 'scale_factor'],
        'remove_object': ['target_color'],
        'duplicate_object': ['target_color', 'offset_y', 'offset_x'],
        'fill_rectangle': ['top', 'left', 'height', 'width', 'color'],
        'draw_line': ['start_y', 'start_x', 'end_y', 'end_x', 'color'],
        'complete_symmetry': ['axis'],
        'create_grid_pattern': ['pattern_height', 'pattern_width'],
        'flood_fill': ['start_y', 'start_x', 'new_color'],
        'apply_gravity': ['direction'],
        'conditional_recolor': ['condition_color', 'target_color', 'new_color'],
        'replace_pattern': ['pattern', 'replacement'],
        'connect_objects': ['color1', 'color2', 'line_color'],
        'outline_objects': ['outline_color'],
        'mirror_complete': [],
        'sort_colors_by_position': ['direction'],
        'repeat_pattern': ['repetitions', 'direction'],
    }
    return signatures
