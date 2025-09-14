"""
Operation DSL for the Object-Agentic solver.

This module defines the core operations that agents can propose,
along with utilities for executing and composing these operations.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from ..grid import Array, to_array, rotate90, flip, translate, color_map
from ..common.objects import connected_components


@dataclass(frozen=True)
class Op:
    """Represents a single operation that can be applied to objects or grids."""
    kind: str                # Operation type
    params: Tuple            # Operation parameters
    target_object: Optional[int] = None  # Which object this applies to (None = global)

    def __str__(self):
        target_str = f"[obj{self.target_object}]" if self.target_object is not None else "[global]"
        return f"{self.kind}{self.params}{target_str}"


# Operation costs for MDL calculation
COST = {
    "identity": 0,
    "recolor": 1,
    "translate": 1,
    "reflect": 1,
    "rotate": 1,
    "align": 2,
    "duplicate": 3,
    "palette_permute": 1,
    "resize": 2,
    "flood_fill": 2,
}


def apply_identity(grid: Array, obj_idx: Optional[int] = None) -> Array:
    """Identity operation - returns grid unchanged."""
    return grid.copy()


def apply_recolor(grid: Array, old_color: int, new_color: int, 
                 obj_idx: Optional[int] = None) -> Array:
    """Recolor operation - changes all pixels of old_color to new_color."""
    result = grid.copy()
    
    if obj_idx is not None:
        # Apply only to specific object
        objects = connected_components(grid)
        if 0 <= obj_idx < len(objects):
            obj = objects[obj_idx]
            for y, x in obj['pixels']:
                if grid[y, x] == old_color:
                    result[y, x] = new_color
    else:
        # Apply globally
        result[grid == old_color] = new_color
    
    return result


def apply_translate(grid: Array, dy: int, dx: int, 
                   obj_idx: Optional[int] = None, fill: int = 0) -> Array:
    """Translate operation - moves content by (dy, dx)."""
    if obj_idx is not None:
        # Translate specific object
        objects = connected_components(grid)
        if not (0 <= obj_idx < len(objects)):
            return grid
        
        result = grid.copy()
        obj = objects[obj_idx]
        
        # Clear original object pixels
        for y, x in obj['pixels']:
            result[y, x] = fill
        
        # Place object at new location
        h, w = grid.shape
        for y, x in obj['pixels']:
            new_y, new_x = y + dy, x + dx
            if 0 <= new_y < h and 0 <= new_x < w:
                result[new_y, new_x] = grid[y, x]
        
        return result
    else:
        # Translate entire grid
        return translate(grid, dy, dx, fill)


def apply_reflect(grid: Array, axis: str, obj_idx: Optional[int] = None) -> Array:
    """Reflect operation - mirrors along specified axis ('h' or 'v')."""
    if obj_idx is not None:
        # Reflect specific object
        objects = connected_components(grid)
        if not (0 <= obj_idx < len(objects)):
            return grid
        
        result = grid.copy()
        obj = objects[obj_idx]
        
        # Clear original object
        bg_color = 0  # Assume background is 0
        for y, x in obj['pixels']:
            result[y, x] = bg_color
        
        # Reflect object around its center
        cy, cx = obj['centroid']
        for y, x in obj['pixels']:
            if axis == 'h':  # horizontal flip (mirror left-right)
                new_x = int(2 * cx - x)
                new_y = y
            else:  # vertical flip (mirror top-bottom)  
                new_y = int(2 * cy - y)
                new_x = x
            
            h, w = grid.shape
            if 0 <= new_y < h and 0 <= new_x < w:
                result[new_y, new_x] = grid[y, x]
        
        return result
    else:
        # Reflect entire grid
        if axis == 'h':
            return flip(grid, axis=1)  # horizontal flip
        else:
            return flip(grid, axis=0)  # vertical flip


def apply_rotate(grid: Array, k: int, obj_idx: Optional[int] = None) -> Array:
    """Rotate operation - rotates by k*90 degrees counter-clockwise."""
    if obj_idx is not None:
        # Object rotation is complex - simplified version
        # For now, just return the grid (could be enhanced)
        return grid
    else:
        # Rotate entire grid
        return rotate90(grid, k)


def apply_align(grid: Array, direction: str, obj_idx: Optional[int] = None) -> Array:
    """
    Align operation - aligns objects to grid edges or centers.
    Direction can be 'left', 'right', 'top', 'bottom', 'center'.
    """
    if obj_idx is None:
        return grid  # Global alignment not well-defined
    
    objects = connected_components(grid)
    if not (0 <= obj_idx < len(objects)):
        return grid
    
    result = grid.copy()
    obj = objects[obj_idx]
    h, w = grid.shape
    
    # Clear original object
    bg_color = 0
    for y, x in obj['pixels']:
        result[y, x] = bg_color
    
    # Compute alignment translation
    top, left, height, width = obj['bbox']
    
    if direction == 'left':
        dy, dx = 0, -left
    elif direction == 'right':
        dy, dx = 0, w - (left + width)
    elif direction == 'top':
        dy, dx = -top, 0
    elif direction == 'bottom':
        dy, dx = h - (top + height), 0
    elif direction == 'center':
        dy, dx = (h - height) // 2 - top, (w - width) // 2 - left
    else:
        dy, dx = 0, 0
    
    # Place object at aligned position
    for y, x in obj['pixels']:
        new_y, new_x = y + dy, x + dx
        if 0 <= new_y < h and 0 <= new_x < w:
            result[new_y, new_x] = grid[y, x]
    
    return result


def apply_duplicate(grid: Array, k: int, pattern: str, 
                   obj_idx: Optional[int] = None) -> Array:
    """
    Duplicate operation - creates k copies of an object in a pattern.
    Pattern can be 'horizontal', 'vertical', or 'grid'.
    """
    if obj_idx is None:
        return grid  # Global duplication not well-defined
        
    objects = connected_components(grid)
    if not (0 <= obj_idx < len(objects)):
        return grid
    
    result = grid.copy()
    obj = objects[obj_idx]
    h, w = grid.shape
    
    # Get object dimensions
    top, left, height, width = obj['bbox']
    
    for copy_idx in range(k):
        if pattern == 'horizontal':
            dy, dx = 0, (copy_idx + 1) * width
        elif pattern == 'vertical':
            dy, dx = (copy_idx + 1) * height, 0
        else:  # grid pattern
            cols = int(np.ceil(np.sqrt(k + 1)))
            row = (copy_idx + 1) // cols
            col = (copy_idx + 1) % cols
            dy, dx = row * height, col * width
        
        # Place copy
        for y, x in obj['pixels']:
            new_y, new_x = y + dy, x + dx
            if 0 <= new_y < h and 0 <= new_x < w:
                result[new_y, new_x] = grid[y, x]
    
    return result


def apply_palette_permute(grid: Array, permutation: Dict[int, int]) -> Array:
    """Apply a color palette permutation to the entire grid."""
    return color_map(grid, permutation)


def execute_operation(grid: Array, op: Op) -> Optional[Array]:
    """Execute a single operation on a grid."""
    try:
        if op.kind == "identity":
            return apply_identity(grid, op.target_object)
        elif op.kind == "recolor":
            old_color, new_color = op.params
            return apply_recolor(grid, old_color, new_color, op.target_object)
        elif op.kind == "translate":
            dy, dx = op.params
            return apply_translate(grid, dy, dx, op.target_object)
        elif op.kind == "reflect":
            axis, = op.params
            return apply_reflect(grid, axis, op.target_object)
        elif op.kind == "rotate":
            k, = op.params
            return apply_rotate(grid, k, op.target_object)
        elif op.kind == "align":
            direction, = op.params
            return apply_align(grid, direction, op.target_object)
        elif op.kind == "duplicate":
            k, pattern = op.params
            return apply_duplicate(grid, k, pattern, op.target_object)
        elif op.kind == "palette_permute":
            permutation, = op.params
            return apply_palette_permute(grid, permutation)
        else:
            # Unknown operation
            return None
    except Exception:
        # Operation failed
        return None


def execute_program_on_grid(program: List[Op], grid: Array) -> Optional[Array]:
    """Execute a sequence of operations on a grid."""
    result = grid
    
    for op in program:
        result = execute_operation(result, op)
        if result is None:
            return None  # Program failed
    
    return result


def propose_ops_for_object(obj: Dict[str, Any], context: Dict[str, Any]) -> List[Op]:
    """
    Propose a list of operations for a specific object.
    Context contains information about the grid, other objects, etc.
    """
    obj_idx = context.get('obj_idx', 0)
    grid = context.get('grid')
    all_objects = context.get('all_objects', [])
    
    proposals = []
    
    # Always propose identity
    proposals.append(Op("identity", (), obj_idx))
    
    # Propose recoloring to common colors
    common_colors = context.get('common_colors', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    current_color = obj['color']
    for new_color in common_colors:
        if new_color != current_color:
            proposals.append(Op("recolor", (current_color, new_color), obj_idx))
    
    # Propose small translations
    for dy in [-3, -2, -1, 0, 1, 2, 3]:
        for dx in [-3, -2, -1, 0, 1, 2, 3]:
            if dy != 0 or dx != 0:
                proposals.append(Op("translate", (dy, dx), obj_idx))
    
    # Propose reflections
    proposals.append(Op("reflect", ("h",), obj_idx))
    proposals.append(Op("reflect", ("v",), obj_idx))
    
    # Propose alignment
    for direction in ["left", "right", "top", "bottom", "center"]:
        proposals.append(Op("align", (direction,), obj_idx))
    
    # Propose duplication (small numbers)
    for k in [1, 2, 3]:
        for pattern in ["horizontal", "vertical"]:
            proposals.append(Op("duplicate", (k, pattern), obj_idx))
    
    # Limit proposals to prevent explosion
    return proposals[:32]


def propose_global_ops(grid: Array, objects: List[Dict[str, Any]]) -> List[Op]:
    """Propose global operations that affect the entire grid."""
    proposals = []
    
    # Identity
    proposals.append(Op("identity", ()))
    
    # Global transformations
    proposals.append(Op("reflect", ("h",)))
    proposals.append(Op("reflect", ("v",)))
    
    for k in [1, 2, 3]:
        proposals.append(Op("rotate", (k,)))
    
    # Palette permutations (simplified)
    colors = list(set(grid.flatten()))
    if len(colors) == 2:
        # Binary swap
        perm = {colors[0]: colors[1], colors[1]: colors[0]}
        proposals.append(Op("palette_permute", (perm,)))
    
    return proposals