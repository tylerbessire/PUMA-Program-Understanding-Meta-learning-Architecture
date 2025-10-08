import inspect
import logging
from collections import Counter
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import label, find_objects as find_objects_scipy

from .grid import (
    color_map as color_map_func,
    crop as crop_func,
    to_array,
    to_list,
    translate as translate_func,
    flip as flip_func,
)
from .object_reasoning import ObjectReasoner


logger = logging.getLogger("arc_dsl")
logger.addHandler(logging.NullHandler())

METRICS: Counter = Counter()

# [S:DSL v2] features=paint,copy,map_objects validation=unit pass


class DSLInvariantError(RuntimeError):
    """Raised when a DSL operation detects an unrecoverable invariant violation."""

class BoundingBox:
    """Axis-aligned bounding box with convenience helpers."""

    def __init__(self, top: int, left: int, bottom: int, right: int, grid_shape: Tuple[int, int]):
        self.top = int(top)
        self.left = int(left)
        self.bottom = int(bottom)
        self.right = int(right)
        self._grid_shape = tuple(int(v) for v in grid_shape)

    def __iter__(self):
        yield self.top
        yield self.left
        yield self.bottom
        yield self.right

    def __getitem__(self, index: int) -> int:
        return (self.top, self.left, self.bottom, self.right)[index]

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.top, self.left, self.bottom, self.right)

    def expand(self, margin: int = 1) -> "BoundingBox":
        """Return a new bounding box expanded by ``margin`` pixels in every direction."""
        height, width = self._grid_shape
        new_top = max(0, self.top - margin)
        new_left = max(0, self.left - margin)
        new_bottom = min(height - 1, self.bottom + margin)
        new_right = min(width - 1, self.right + margin)
        return BoundingBox(new_top, new_left, new_bottom, new_right, self._grid_shape)

    def fill(self, color: int, *, background_color: int = 0) -> "Grid":
        """Return a grid with the bounding box area painted ``color``."""
        height, width = self._grid_shape
        new_grid = np.full((height, width), background_color, dtype=int)
        new_grid[self.top : self.bottom + 1, self.left : self.right + 1] = color
        return Grid(new_grid)

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1


class Object:
    def __init__(self, pixels, color, grid_shape: Tuple[int, int]):
        self.pixels = pixels
        self.color = int(color)
        self.height = int(np.max(pixels[:, 0]) - np.min(pixels[:, 0]) + 1)
        self.width = int(np.max(pixels[:, 1]) - np.min(pixels[:, 1]) + 1)
        self.y = int(np.min(pixels[:, 0]))
        self.x = int(np.min(pixels[:, 1]))
        self._grid_shape = tuple(int(v) for v in grid_shape)

    def __repr__(self):
        return f"Object(color={self.color}, shape=({self.height}, {self.width}), top_left=({self.y}, {self.x}))"

    def bounding_box(self) -> BoundingBox:
        return BoundingBox(self.y, self.x, self.y + self.height - 1, self.x + self.width - 1, self._grid_shape)

    def bbox(self) -> BoundingBox:
        return self.bounding_box()

    def translate(self, offset):
        """Translate an object by an offset (dy, dx)."""
        new_pixels = self.pixels.copy()
        new_pixels[:, 0] += offset[0]  # dy
        new_pixels[:, 1] += offset[1]  # dx
        return Object(new_pixels, self.color, self._grid_shape)

    def copy_to_grid(
        self,
        height: int,
        width: int,
        target_y: int = 0,
        target_x: int = 0,
        *,
        background_color: int = 0,
        color_transform: Optional[dict[int, int]] = None,
    ) -> "Grid":
        """Copy the object into a new grid of the provided shape."""
        new_grid = np.full((height, width), background_color, dtype=int)
        transform = color_transform or {}
        for r, c in self.pixels:
            dest_r = r - self.y + target_y
            dest_c = c - self.x + target_x
            if 0 <= dest_r < height and 0 <= dest_c < width:
                original = self.color
                new_grid[dest_r, dest_c] = transform.get(original, original)
        return Grid(new_grid)

    def touching(self, other: "Object", adjacent_only: bool = True) -> bool:
        """Return True if this object touches ``other``."""
        offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)] if adjacent_only else [
            (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if not (dr == 0 and dc == 0)
        ]
        other_pixels = {(r, c) for r, c in other.pixels}
        for r, c in self.pixels:
            for dr, dc in offsets:
                if (r + dr, c + dc) in other_pixels:
                    return True
        return False

    def __getitem__(self, key):
        if key == 'colors':
            return [self.color]
        if key == 'original_color':
            return self.color
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Object has no key '{key}'")

# --- Global DSL Functions ---
def compose(*funcs):
    def composed_func(arg):
        res = arg
        for f in funcs:
            res = f(res)
        return res
    return composed_func

def compose_grids(grid1, grid2, background_color=0):
    """Combines two grids by overlaying grid2 on top of grid1."""
    if not isinstance(grid1, Grid):
        grid1 = Grid(grid1)
    if not isinstance(grid2, Grid):
        grid2 = Grid(grid2)

    # Start with a copy of the first grid
    new_grid_arr = grid1._grid.copy()

    # Ensure grids are the same size, if not, this might need a resizing strategy
    # For now, we assume they are the same size as per typical use cases.
    if grid1.shape != grid2.shape:
        # Simple strategy: crop or pad the second grid to match the first.
        # This is a basic approach and might need refinement based on actual task needs.
        temp_grid2 = np.full(grid1.shape, background_color, dtype=int)
        h = min(grid1.shape[0], grid2.shape[0])
        w = min(grid1.shape[1], grid2.shape[1])
        temp_grid2[:h, :w] = grid2._grid[:h, :w]
        grid2_arr = temp_grid2
    else:
        grid2_arr = grid2._grid

    # Overlay grid2 on grid1. Where grid2 has non-background pixels, they are copied over.
    mask = grid2_arr != background_color
    new_grid_arr[mask] = grid2_arr[mask]

    return Grid(new_grid_arr)

def map_color(mapping):
    """Returns a function that applies a color mapping to a grid."""
    def func(grid):
        if not isinstance(grid, Grid):
            grid = Grid(grid)
        return grid.map_colors(mapping)
    return func

def find_objects(grid, color=None, ignore_color=0, min_size=1, **kwargs):
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    
    binary_grid = grid._grid != ignore_color
    if color is not None:
        binary_grid = grid._grid == color

    labeled_grid, num_labels = label(binary_grid)
    if num_labels == 0:
        return []
    
    objects = []
    slices = find_objects_scipy(labeled_grid)
    for i in range(num_labels):
        pixels = np.argwhere(labeled_grid == (i + 1))
        if len(pixels) < min_size:
            continue
        obj_color = grid._grid[pixels[0][0], pixels[0][1]]
        objects.append(Object(pixels, obj_color, grid.shape))
        
    return objects

def get_objects(grid, *args, **kwargs):
    return find_objects(grid, *args, **kwargs)

def bounding_box(obj):
    """Get bounding box coordinates (min_x, min_y, max_x, max_y) for an object or list of objects."""
    if isinstance(obj, list):
        if not obj:
            return (0, 0, 0, 0)
        # For multiple objects, find the bounding box that contains all of them
        min_x = min(o.x for o in obj)
        min_y = min(o.y for o in obj)
        max_x = max(o.x + o.width - 1 for o in obj)
        max_y = max(o.y + o.height - 1 for o in obj)
        return (min_x, min_y, max_x, max_y)
    else:
        # Single object
        return (obj.x, obj.y, obj.x + obj.width - 1, obj.y + obj.height - 1)

def map_objects(
    grid,
    *args,
    filter_func=None,
    f=None,
    predicate=None,
    color=None,
    colors=None,
    objects=None,
    background_color: int = 0,
    target_color: Optional[int] = None,
    also_color: bool = False,
    right: bool = False,
    right_color: Optional[int] = None,
    **kwargs,
):
    """Apply a transform to objects discovered in ``grid`` with flexible filtering."""

    color_filter = kwargs.pop("filter_color", None)
    # Allow and ignore other common kwargs to prevent crashes
    kwargs.pop("filter", None)
    kwargs.pop("filter_size", None)
    kwargs.pop("map_fn", None)
    kwargs.pop("color_map", None)

    if kwargs:
        raise DSLInvariantError(f"Unsupported map_objects kwargs: {sorted(kwargs.keys())}")

    base_grid = grid if isinstance(grid, Grid) else Grid(grid)
    arg_list = list(args)
    transform = None
    selection = None
    explicit_objects = objects

    if arg_list:
        first = arg_list.pop(0)
        if callable(first) and f is not None:
            predicate = first
            transform = f
        elif not callable(first):
            explicit_objects = first
        else:
            transform = first
        if arg_list:
            if transform is None and callable(arg_list[0]):
                transform = arg_list.pop(0)
            elif selection is None:
                selection = arg_list.pop(0)
        if arg_list:
            raise DSLInvariantError("Too many positional arguments for map_objects")

    if transform is None and f is not None:
        transform = f
    if transform is None:
        raise DSLInvariantError("map_objects requires a transform callable")

    def expand_candidates(source):
        if source is None:
            return []
        if isinstance(source, Object):
            return [source]
        if isinstance(source, BoundingBox):
            return [source]
        if isinstance(source, Grid):
            return find_objects(source)
        if isinstance(source, np.ndarray):
            return find_objects(Grid(source))
        if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
            items = []
            for item in source:
                if isinstance(item, tuple) and len(item) == 4:
                    items.append(BoundingBox(item[0], item[1], item[2], item[3], base_grid.grid.shape))
                else:
                    items.extend(expand_candidates(item))
            return items
        raise DSLInvariantError(f"Unsupported object specification: {type(source)}")

    if explicit_objects is not None:
        candidates = expand_candidates(explicit_objects)
    elif selection is not None:
        candidates = expand_candidates(selection)
    else:
        selected_color = None
        if color is not None:
            selected_color = color
        elif colors is not None:
            selected_color = list(colors)
        candidates = find_objects(base_grid, color=selected_color)

    def passes_filters(obj):
        current = obj
        obj_color = current.color if isinstance(current, Object) else None
        if isinstance(current, BoundingBox):
            obj_color = obj_color or base_grid.grid[current.top : current.bottom + 1, current.left : current.right + 1].max()
        if filter_func and not filter_func(current):
            return False
        if predicate and not predicate(current):
            return False
        if color is not None and obj_color is not None and obj_color != color:
            return False
        if colors is not None and obj_color is not None and obj_color not in set(int(c) for c in colors):
            return False
        if color_filter is not None:
            if callable(color_filter):
                if obj_color is None or not color_filter(obj_color):
                    return False
            else:
                allowed = {color_filter} if isinstance(color_filter, int) else set(color_filter)
                if obj_color is None or obj_color not in allowed:
                    return False
        return True

    def overlay(result_item, anchor):
        if result_item is None:
            return
        if isinstance(result_item, (list, tuple)):
            for sub in result_item:
                overlay(sub, anchor)
            return
        if isinstance(result_item, dict) and "color" in result_item:
            color_value = result_item["color"]
            paint_object_area(anchor, color_value)
            return
        if isinstance(result_item, (int, np.integer)):
            paint_object_area(anchor, int(result_item))
            return
        if isinstance(result_item, BoundingBox):
            paint_object_area(result_item, target_color if target_color is not None else anchor_color(anchor))
            return

        overlay_grid = result_item
        if isinstance(result_item, Object):
            overlay_grid = result_item.copy_to_grid(result_item._grid_shape[0], result_item._grid_shape[1])
        if isinstance(overlay_grid, Grid):
            arr = overlay_grid.grid
        else:
            arr = np.array(overlay_grid)

        if arr.shape == base_grid.grid.shape:
            mask = arr != background_color
            base_array[mask] = arr[mask]
            return

        top, left = anchor_position(anchor)
        h, w = arr.shape
        end_row = min(base_array.shape[0], top + h)
        end_col = min(base_array.shape[1], left + w)
        arr_h = end_row - top
        arr_w = end_col - left
        if arr_h <= 0 or arr_w <= 0:
            return
        region = base_array[top:end_row, left:end_col]
        mask = arr[:arr_h, :arr_w] != background_color
        region[mask] = arr[:arr_h, :arr_w][mask]
        base_array[top:end_row, left:end_col] = region

    def anchor_position(anchor):
        if isinstance(anchor, Object):
            return anchor.y, anchor.x
        if isinstance(anchor, BoundingBox):
            return anchor.top, anchor.left
        if isinstance(anchor, tuple) and len(anchor) == 4:
            bbox = BoundingBox(anchor[0], anchor[1], anchor[2], anchor[3], base_grid.grid.shape)
            return bbox.top, bbox.left
        return (0, 0)

    def anchor_color(anchor):
        if isinstance(anchor, Object):
            return anchor.color
        if isinstance(anchor, BoundingBox):
            sample = base_grid.grid[anchor.top : anchor.bottom + 1, anchor.left : anchor.right + 1]
            unique, counts = np.unique(sample, return_counts=True)
            return int(unique[np.argmax(counts)]) if unique.size else background_color
        return background_color

    def paint_object_area(anchor, new_color):
        if isinstance(anchor, Object):
            for r, c in anchor.pixels:
                if 0 <= r < base_array.shape[0] and 0 <= c < base_array.shape[1]:
                    base_array[r, c] = new_color
        else:
            bbox = anchor if isinstance(anchor, BoundingBox) else BoundingBox(anchor[0], anchor[1], anchor[2], anchor[3], base_grid.grid.shape)
            top, left = bbox.top, bbox.left
            end_row = min(base_array.shape[0], bbox.bottom + 1)
            end_col = min(base_array.shape[1], bbox.right + 1)
            base_array[top:end_row, left:end_col] = new_color

    base_array = base_grid.grid.copy()

    for obj in candidates:
        anchor = obj
        if isinstance(obj, tuple) and len(obj) == 4:
            anchor = BoundingBox(obj[0], obj[1], obj[2], obj[3], base_grid.grid.shape)
        if not passes_filters(anchor):
            continue
        result = transform(anchor)
        if target_color is not None:
            paint_object_area(anchor, target_color)
            if not also_color:
                result = None
        overlay(result, anchor)
        if right and right_color is not None:
            top, left = anchor_position(anchor)
            height = anchor.height if isinstance(anchor, BoundingBox) else anchor.height
            right_col = left + (anchor.width if isinstance(anchor, BoundingBox) else anchor.width)
            end_row = min(base_array.shape[0], top + height)
            if 0 <= right_col < base_array.shape[1]:
                base_array[top:end_row, right_col] = right_color

    METRICS["map_objects_calls"] += 1
    logger.debug(
        "map_objects",
        extra={
            "event": "map_objects",
            "num_candidates": len(candidates),
        },
    )
    return Grid(base_array)

def paint(
    target,
    color: Optional[int] = None,
    mask=None,
    *,
    coordinates: Optional[Iterable[Tuple[int, int]]] = None,
    background_color: int = 0,
) -> "Grid":
    """General-purpose paint helper supporting mask matrices and predicates."""

    if isinstance(target, Object):
        grid = target.copy_to_grid(target.height, target.width, background_color=background_color)
        base_array = grid.grid.copy()
    else:
        grid = target if isinstance(target, Grid) else Grid(target)
        base_array = grid.grid.copy()

    def resolve_color(default_color: Optional[int], mask_value) -> int:
        if isinstance(mask_value, (np.integer, int)):
            if mask_value <= 0 and default_color is not None:
                return default_color
            if mask_value <= 0 and default_color is None:
                return background_color
            if mask_value == 1 and default_color is not None:
                return int(default_color)
            return int(mask_value)
        return int(default_color if default_color is not None else mask_value)

    def apply_at(row: int, col: int, value_override=None):
        if 0 <= row < base_array.shape[0] and 0 <= col < base_array.shape[1]:
            if value_override is None:
                new_color = color if color is not None else base_array[row, col]
            else:
                new_color = resolve_color(color, value_override)
            base_array[row, col] = new_color

    if mask is None and coordinates is None:
        if color is None:
            return Grid(base_array)
        base_array[:, :] = color
    else:
        if mask is not None:
            if callable(mask):
                sig = inspect.signature(mask)
                for r in range(base_array.shape[0]):
                    for c in range(base_array.shape[1]):
                        args = (r, c)
                        if len(sig.parameters) >= 3:
                            args += (int(grid.grid[r, c]),)
                        if len(sig.parameters) == 1:
                            args = (grid.grid[r, c],)
                        if mask(*args):
                            apply_at(r, c)
            else:
                mask_array = mask
                if isinstance(mask, Grid):
                    mask_array = mask.grid
                mask_array = np.array(mask_array)
                if mask_array.shape != base_array.shape:
                    raise DSLInvariantError("Mask dimensions must match grid shape")
                for r in range(mask_array.shape[0]):
                    for c in range(mask_array.shape[1]):
                        if mask_array[r, c] is None:
                            continue
                        if isinstance(mask_array[r, c], (int, np.integer)) and mask_array[r, c] <= 0:
                            continue
                        apply_at(r, c, mask_array[r, c])
        if coordinates is not None:
            for coord in coordinates:
                if coord is None:
                    continue
                if len(coord) != 2:
                    raise DSLInvariantError("Coordinates must be (row, col) tuples")
                apply_at(
                    int(coord[0]),
                    int(coord[1]),
                    value_override=color if color is not None else None,
                )

    METRICS["paint_calls"] += 1
    logger.debug(
        "paint",
        extra={
            "event": "paint",
            "color": color,
            "has_mask": mask is not None,
            "has_coordinates": coordinates is not None,
        },
    )
    return Grid(base_array)

def reflect(grid, axis, offset=0):
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    if axis == 'horizontal':
        return Grid(flip_func(grid.grid, 0))
    elif axis == 'vertical':
        return Grid(flip_func(grid.grid, 1))
    elif axis == 'both':
        return Grid(flip_func(flip_func(grid.grid, 0), 1))
    return grid

def copy(
    target=None,
    *,
    dy: int = 0,
    dx: int = 0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    source_y: Optional[int] = None,
    source_x: Optional[int] = None,
    target_y: Optional[int] = None,
    target_x: Optional[int] = None,
    target_grid=None,
    color_transform: Optional[dict[int, int]] = None,
    background_color: int = 0,
    **kwargs,
) -> "Grid":
    """Flexible copy utility for grids and objects."""
    if target is None:
        return Grid(np.array([[]]))

    kwargs.pop("source", None)

    def resolve_target_grid(shape):
        if target_grid is None:
            return np.full(shape, background_color, dtype=int)
        if isinstance(target_grid, Grid):
            return target_grid.grid.copy()
        if isinstance(target_grid, (tuple, list)) and len(target_grid) == 2:
            return np.full((int(target_grid[0]), int(target_grid[1])), background_color, dtype=int)
        array = np.array(target_grid)
        if array.ndim != 2:
            raise DSLInvariantError("target_grid must be 2D")
        return array.copy()

    if isinstance(target, Object):
        dest_height = int(height if height is not None else target.height)
        dest_width = int(width if width is not None else target.width)
        dest = resolve_target_grid((dest_height, dest_width))
        offset_y = max(0, -dy)
        offset_x = max(0, -dx)
        for r, c in target.pixels:
            rel_r = r - target.y + offset_y
            rel_c = c - target.x + offset_x
            if 0 <= rel_r < dest_height and 0 <= rel_c < dest_width:
                color = target.color
                if color_transform and color in color_transform:
                    color = color_transform[color]
                dest[rel_r, rel_c] = color
        result = Grid(dest)
    else:
        grid = target if isinstance(target, Grid) else Grid(target)
        sy = int(source_y if source_y is not None else 0)
        sx = int(source_x if source_x is not None else 0)
        h = int(height if height is not None else grid.height - sy)
        w = int(width if width is not None else grid.width - sx)
        sy = max(0, sy)
        sx = max(0, sx)
        h = max(0, min(h, grid.height - sy))
        w = max(0, min(w, grid.width - sx))
        source_slice = grid.grid[sy : sy + h, sx : sx + w]

        if target_grid is None:
            dest = resolve_target_grid(grid.grid.shape)
        else:
            dest = resolve_target_grid((h, w))

        ty = int(target_y) if target_y is not None else sy + dy
        tx = int(target_x) if target_x is not None else sx + dx
        for r in range(h):
            for c in range(w):
                dest_r = ty + r
                dest_c = tx + c
                if 0 <= dest_r < dest.shape[0] and 0 <= dest_c < dest.shape[1]:
                    value = source_slice[r, c]
                    if color_transform and value in color_transform:
                        value = color_transform[value]
                    dest[dest_r, dest_c] = value
        result = Grid(dest)

    METRICS["copy_calls"] += 1
    logger.debug(
        "copy",
        extra={
            "event": "copy",
            "is_object": isinstance(target, Object),
            "dy": dy,
            "dx": dx,
        },
    )
    return result

def copy_grid(grid):
    result = grid.copy()
    logger.debug("copy_grid", extra={"event": "copy_grid", "input_type": type(grid).__name__})
    return result

def deepcopy(obj):
    return obj.copy()

def filter_color(grid, color):
    """Keep only specified color(s), setting all others to 0."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)

    # Handle both single color and list of colors
    if isinstance(color, list):
        colors = set(color)
    else:
        colors = {color}

    new_grid = np.where(np.isin(grid.grid, list(colors)), grid.grid, 0)
    return Grid(new_grid)

def paint_diagonal(
    grid,
    *,
    direction: str = "anti",
    include_self: bool = True,
    background_color: int = 0,
    colors: Optional[Sequence[int]] = None,
    stop_on_collision: bool = True,
):
    """Propagate colored pixels along a diagonal direction.

    Parameters
    ----------
    grid: Grid | Sequence[Sequence[int]]
        The source grid to transform.
    direction: {"anti", "main", "both"}
        Which diagonal to extend along. "anti" expands top-right → bottom-left,
        "main" expands top-left → bottom-right, and "both" applies both directions.
    include_self: bool
        Whether to keep the seed pixels in the result. When ``False`` the original
        pixels are reset to ``background_color`` after propagation.
    background_color: int
        Color treated as empty. Pixels with this color are overwritten by the
        propagation.
    colors: Optional[Sequence[int]]
        Restrict propagation to the provided colors. When ``None`` every
        non-background pixel acts as a seed.
    stop_on_collision: bool
        If ``True`` propagation halts when encountering a non-background pixel
        that is not the seed color; otherwise it overwrites encountered pixels.
    """

    if not isinstance(grid, Grid):
        grid = Grid(grid)

    direction = direction.lower()
    direction_vectors: dict[str, Tuple[Tuple[int, int], ...]] = {
        "anti": ((1, -1), (-1, 1)),
        "main": ((1, 1), (-1, -1)),
        "both": ((1, -1), (-1, 1), (1, 1), (-1, -1)),
    }
    if direction not in direction_vectors:
        raise DSLInvariantError(f"Unsupported diagonal direction '{direction}'")

    seeds = (
        np.argwhere(np.isin(grid.grid, list(colors)))
        if colors is not None
        else np.argwhere(grid.grid != background_color)
    )

    new_grid = grid.grid.copy()
    height, width = new_grid.shape

    for r, c in seeds:
        color = grid.grid[r, c]
        if color == background_color:
            continue

        for dr, dc in direction_vectors[direction]:
            nr, nc = r + dr, c + dc
            while 0 <= nr < height and 0 <= nc < width:
                if stop_on_collision and new_grid[nr, nc] not in (background_color, color):
                    break
                new_grid[nr, nc] = color
                nr += dr
                nc += dc

        if not include_self:
            new_grid[r, c] = background_color

    METRICS["paint_diagonal_calls"] += 1
    logger.debug(
        "paint_diagonal",
        extra={
            "event": "paint_diagonal",
            "direction": direction,
            "include_self": include_self,
            "colors": None if colors is None else list(colors),
        },
    )
    return Grid(new_grid)

def fill_rectangle(target, color):
    """Fills the bounding box of an Object, a BoundingBox, or the entire Grid with a specified color."""
    if isinstance(target, Object):
        bbox = target.bounding_box()
        new_grid_arr = np.full((bbox.height, bbox.width), color, dtype=int)
        return Grid(new_grid_arr)
    elif isinstance(target, BoundingBox):
        new_grid_arr = np.full((target.height, target.width), color, dtype=int)
        return Grid(new_grid_arr)
    elif isinstance(target, Grid):
        return paint_grid(target, color)
    else:
        raise ValueError(f"fill_rectangle does not support type {type(target)}")

def translate(obj, offset):
    """Translate an object by an offset (dy, dx) or (dx, dy)."""
    if hasattr(obj, 'pixels'):
        # It's an Object
        return obj.translate(offset)
    else:
        # It's a grid
        if not isinstance(obj, Grid):
            obj = Grid(obj)
        return Grid(translate_func(obj.grid, offset[0], offset[1], fill=0))

def crop(
    grid=None,
    top: Optional[int] = None,
    left: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    *,
    x0: Optional[int] = None,
    y0: Optional[int] = None,
    x1: Optional[int] = None,
    y1: Optional[int] = None,
    **kwargs,
):
    """Crop a grid using either (top, left, height, width) or (x0, y0, x1, y1)."""
    if grid is None:
        return Grid(np.array([[]]))
    kwargs.pop("x2", None)
    kwargs.pop("min_x", None)
    if not isinstance(grid, Grid):
        grid = Grid(grid)

    top_val = y0 if y0 is not None else top if top is not None else 0
    left_val = x0 if x0 is not None else left if left is not None else 0

    if x1 is not None and y1 is not None:
        height_val = y1 - top_val + 1
        width_val = x1 - left_val + 1
    else:
        if height is None or width is None:
            raise DSLInvariantError("crop requires height and width when x1/y1 are not provided")
        height_val = height
        width_val = width

    return grid.crop(int(top_val), int(left_val), int(height_val), int(width_val))

def row(grid, index):
    """Get a specific row from the grid."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    return Grid(grid.grid[index])

def extend_lines(grid, colors=None, background_color=0, extend_until_collision=True):
    """Extend lines from specified colors horizontally and vertically."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)

    new_grid_arr = grid.grid.copy()
    height, width = new_grid_arr.shape

    if colors is None:
        # If no colors are specified, find all non-background colors
        colors = np.unique(new_grid_arr[new_grid_arr != background_color])
    
    if not isinstance(colors, list):
        colors = [colors]

    for r in range(height):
        for c in range(width):
            if new_grid_arr[r, c] in colors:
                color = new_grid_arr[r, c]
                
                # Extend left
                for nc in range(c - 1, -1, -1):
                    if extend_until_collision and new_grid_arr[r, nc] != background_color:
                        break
                    new_grid_arr[r, nc] = color
                
                # Extend right
                for nc in range(c + 1, width):
                    if extend_until_collision and new_grid_arr[r, nc] != background_color:
                        break
                    new_grid_arr[r, nc] = color
                
                # Extend up
                for nr in range(r - 1, -1, -1):
                    if extend_until_collision and new_grid_arr[nr, c] != background_color:
                        break
                    new_grid_arr[nr, c] = color

                # Extend down
                for nr in range(r + 1, height):
                    if extend_until_collision and new_grid_arr[nr, c] != background_color:
                        break
                    new_grid_arr[nr, c] = color
                        
    return Grid(new_grid_arr)

def fill_largest_object(grid, color):
    """Find the largest object in the grid and fill it with the specified color."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)

    objects = find_objects(grid)
    if not objects:
        return grid

    # Find largest object by number of pixels
    largest_obj = max(objects, key=lambda obj: len(obj.pixels))

    # Create a new grid and paint the largest object
    new_grid = grid.grid.copy()
    for pixel in largest_obj.pixels:
        new_grid[pixel[0], pixel[1]] = color

    return Grid(new_grid)

def paint_grid(grid, color):
    """Fill entire grid with a single color."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    new_grid = np.full_like(grid.grid, color)
    return Grid(new_grid)

def paint_object(grid, obj, color=None):
    """Paints an object onto the grid."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    paint_color = color if color is not None else obj.color
    for y, x in obj.pixels:
        if 0 <= y < grid.grid.shape[0] and 0 <= x < grid.grid.shape[1]:
            grid.grid[y, x] = paint_color

class ObjectCollection(list):
    def __init__(self, objects):
        super().__init__(objects)

    def __call__(self, color=None):
        if color is None:
            return list(self)
        if isinstance(color, (list, tuple, set)):
            colors = set(int(c) for c in color)
            return [obj for obj in self if obj.color in colors]
        return [obj for obj in self if obj.color == color]


class GridRow:
    def __init__(self, row_data, source_grid):
        self._row = row_data
        self._source_grid = source_grid

    @property
    def grid(self):
        return self._source_grid

    def __getitem__(self, key):
        return self._row[key]

    def __setitem__(self, key, value):
        self._row[key] = value

    def __len__(self):
        return len(self._row)


class Grid:
    def __init__(self, grid_data):
        if isinstance(grid_data, Grid):
            self._grid = grid_data._grid.copy()
        else:
            self._grid = to_array(grid_data)
        logger.debug(
            "grid_init",
            extra={
                "event": "grid_init",
                "shape": tuple(self._grid.shape),
                "dtype": str(self._grid.dtype),
            },
        )

    @property
    def grid(self):
        return self

    @property
    def shape(self):
        return self._grid.shape

    @property
    def height(self):
        return self._grid.shape[0]

    @property
    def width(self):
        return self._grid.shape[1]

    @property
    def objects(self):
        return ObjectCollection(find_objects(self))

    def objects_by_color(self, color):
        return ObjectCollection(find_objects(self, color=color))

    def to_list(self):
        return self._grid.tolist()

    def copy(self):
        return Grid(self._grid.copy())

    def __array__(self, dtype=None):
        return np.asarray(self._grid, dtype=dtype)

    def __len__(self):
        return len(self._grid)

    def __iter__(self):
        return iter(self._grid)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "grid":
                return self._grid
            if item == "height":
                return self.height
            if item == "width":
                return self.width
            raise KeyError(item)
        result = self._grid[item]
        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                return Grid(result)
            elif result.ndim == 1:
                return GridRow(result, self)
        return result

    def __call__(self, *args, **kwargs):
        return self

    def __setitem__(self, key, value):
        self._grid[key] = value

    def get(self, r, c):
        return self._grid[r, c]

    def replace(self, old, new):
        return self.replace_color(old, new)

    def replace_color(self, old_color, new_color, mask=None, **kwargs):
        kwargs.pop("only_if_surrounded_by", None)
        mapping = {old_color: new_color}
        new_grid = self._grid.copy()
        if mask is not None:
            for r in range(self.height):
                for c in range(self.width):
                    if mask[r, c] and self._grid[r, c] == old_color:
                        new_grid[r, c] = new_color
            return Grid(new_grid)
        else:
            return Grid(color_map_func(self._grid, mapping))

    def map_colors(self, mapping):
        new_grid_data = color_map_func(self._grid, mapping)
        return Grid(new_grid_data)

    def repeat(self, factor_h, factor_w):
        new_grid_data = np.tile(self._grid, (factor_h, factor_w))
        return Grid(new_grid_data)

    def map_color(self, old, new):
        return self.replace_color(old, new)

    def fill_regions(self):
        labeled_grid, num_labels = label(self._grid)
        if num_labels == 0:
            return self
        for i in range(1, num_labels + 1):
            component_mask = (labeled_grid == i)
            colors, counts = np.unique(self._grid[component_mask], return_counts=True)
            if len(colors) > 0:
                dominant_color = colors[np.argmax(counts)]
                self._grid[component_mask] = dominant_color
        return Grid(self._grid)

    def expand_grid(self, factor):
        new_grid_data = np.kron(self._grid, np.ones((factor, factor)))
        return Grid(new_grid_data)

    def crop(self, top, left, height, width):
        new_grid_data = crop_func(self._grid, top, left, height, width)
        return Grid(new_grid_data)

    def translate(self, dx, dy, fill_value=0):
        return Grid(translate_func(self._grid, dy, dx, fill=fill_value))

    def compose(self, *funcs):
        res = self
        for f in funcs:
            res = f(res)
        return res

    def eq(self, other):
        if isinstance(other, Grid):
            return np.array_equal(self._grid, other._grid)
        return np.array_equal(self._grid, other)

    def map(self, func):
        new_grid = np.vectorize(func)(self._grid)
        return Grid(new_grid)

    def max(self):
        """Returns the maximum value in the grid."""
        return self._grid.max()

    def subgrid(self, r, c, h, w):
        return Grid(self._grid[r:r+h, c:c+w])

    def bounding_box(self, ignore_color: int = 0) -> Optional[Tuple[int, int, int, int]]:
        """Return (left, top, right, bottom) around non-background pixels."""
        coords = np.argwhere(self._grid != ignore_color)
        if coords.size == 0:
            return None

        rows = coords[:, 0]
        cols = coords[:, 1]
        return (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))

    def filter_color(self, color):
        return filter_color(self, color)

    def pattern_fill(
        self,
        region: Union[Tuple[int, int, int, int], "Object", "Grid", None],
        color: int,
        *,
        background_color: int = 0,
    ):
        """Fill the provided region with ``color``.

        ``region`` may be a bounding-box tuple, another grid, or an ``Object``.
        If the region is ``None`` the grid is returned unchanged.
        """

        bbox: Optional[Tuple[int, int, int, int]] = None
        if region is None:
            return Grid(self._grid.copy())
        if isinstance(region, Grid):
            bbox = region.bounding_box(ignore_color=background_color)
        elif hasattr(region, "bounding_box"):
            bbox = region.bounding_box()
        elif isinstance(region, (list, tuple)):
            if len(region) != 4:
                raise DSLInvariantError("Bounding box tuples must have four elements")
            bbox = tuple(int(v) for v in region)  # type: ignore[assignment]
        else:
            raise DSLInvariantError(f"Unsupported region type for pattern_fill: {type(region)}")

        if bbox is None:
            return Grid(self._grid.copy())

        left, top, right, bottom = bbox
        if not (0 <= left <= right < self.width and 0 <= top <= bottom < self.height):
            raise DSLInvariantError("Bounding box extends outside grid bounds")

        new_grid = self._grid.copy()
        new_grid[top : bottom + 1, left : right + 1] = color

        METRICS["pattern_fill_calls"] += 1
        logger.debug(
            "pattern_fill",
            extra={
                "event": "pattern_fill",
                "color": color,
                "bbox": [left, top, right, bottom],
            },
        )
        return Grid(new_grid)

    def expand_to_grid(self, width, height, **kwargs):
        """Tiles the current grid to fill a new grid of specified dimensions."""
        new_grid_arr = np.zeros((height, width), dtype=self._grid.dtype)
        h, w = self.shape

        for r in range(0, height, h):
            for c in range(0, width, w):
                # Determine the piece of the source grid to copy
                block_h = min(h, height - r)
                block_w = min(w, width - c)
                source_block = self._grid[:block_h, :block_w]
                # Place it in the new grid
                new_grid_arr[r:r+block_h, c:c+block_w] = source_block
        
        return Grid(new_grid_arr)

    def expand_square(self, factor):
        """Expands each pixel into a square block of size factor x factor."""
        new_grid_data = np.kron(self._grid, np.ones((factor, factor), dtype=self._grid.dtype))
        return Grid(new_grid_data)

    def fill_gaps(self, background_color=0):
        """Fills holes within objects with the color of the surrounding object."""
        from collections import deque

        new_grid = self._grid.copy()
        height, width = self._grid.shape
        visited = np.zeros_like(self._grid, dtype=bool)

        # 1. Find all exterior background pixels by flood-filling from the border
        q = deque()
        for r in range(height):
            if self._grid[r, 0] == background_color:
                q.append((r, 0))
                visited[r, 0] = True
            if self._grid[r, width - 1] == background_color:
                q.append((r, width - 1))
                visited[r, width - 1] = True
        for c in range(width):
            if self._grid[0, c] == background_color:
                q.append((0, c))
                visited[0, c] = True
            if self._grid[height - 1, c] == background_color:
                q.append((height - 1, c))
                visited[height - 1, c] = True

        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and self._grid[nr, nc] == background_color:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        # 2. Iterate and fill any unvisited background pixels (which are holes)
        for r in range(height):
            for c in range(width):
                if self._grid[r, c] == background_color and not visited[r, c]:
                    # This is a hole. Find the surrounding color.
                    fill_color = -1
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and self._grid[nr, nc] != background_color:
                            fill_color = self._grid[nr, nc]
                            break
                    
                    if fill_color != -1:
                        # Flood fill the hole with the found color
                        hole_q = deque([(r, c)])
                        new_grid[r, c] = fill_color
                        visited[r, c] = True
                        while hole_q:
                            hr, hc = hole_q.popleft()
                            for hdr, hdc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nhr, nhc = hr + hdr, hc + hdc
                                if 0 <= nhr < height and 0 <= nhc < width and not visited[nhr, nhc] and self._grid[nhr, nhc] == background_color:
                                    visited[nhr, nhc] = True
                                    new_grid[nhr, nhc] = fill_color
                                    hole_q.append((nhr, nhc))
        return Grid(new_grid)

    def map_rows(self, func): 
        new_grid = self._grid.copy()
        for r in range(self.height):
            new_grid[r] = func(self._grid[r])
        return Grid(new_grid)
    def fill_until_collision(self, background_color=0):
        """Performs a multi-source flood fill from all non-background pixels."""
        new_grid = self._grid.copy()
        height, width = self._grid.shape

        queue = []
        # Initialize queue with all non-background pixels
        for r in range(height):
            for c in range(width):
                if self._grid[r, c] != background_color:
                    queue.append((r, c))

        head = 0
        while head < len(queue):
            r, c = queue[head]
            head += 1
            color = new_grid[r, c]

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc

                if 0 <= nr < height and 0 <= nc < width and new_grid[nr, nc] == background_color:
                    new_grid[nr, nc] = color
                    queue.append((nr, nc))
        
        return Grid(new_grid)

    def filter_rows(self, func):
        new_grid_rows = [self._grid[r] for r in range(self.height) if func(r)]
        if not new_grid_rows:
            return Grid(np.array([[]]))
        return Grid(np.vstack(new_grid_rows))

    def paint_rows(self, filter_func, color):
        """Paints rows selected by a filter function with a specific color."""
        new_grid = self._grid.copy()
        for r in range(self.height):
            if filter_func(r):
                new_grid[r, :] = color
        return Grid(new_grid)

    def pattern_map(self, pattern=None, replacement=None, key=None, values=None, default=None, **kwargs):
        """Maps patterns in the grid to new values."""
        if 'pattern_width' in kwargs or pattern is None:
            # Handle the simple/ambiguous case by doing nothing
            return self

        new_grid = self._grid.copy()
        pattern_height = len(pattern)
        pattern_width = len(pattern[0])

        for r in range(self.height - pattern_height + 1):
            for c in range(self.width - pattern_width + 1):
                view = self._grid[r:r+pattern_height, c:c+pattern_width]
                
                # Use the key function to get a lookup key for the current view
                key_tuple = key(view)
                
                replacement_value = values.get(key_tuple)
                
                if replacement_value is None and default is not None:
                    replacement_value = default(view)
                
                if replacement_value is not None:
                    # Apply the replacement
                    for pr in range(pattern_height):
                        for pc in range(pattern_width):
                            if replacement[pr][pc]:
                                new_grid[r+pr, c+pc] = replacement_value
                                
        return Grid(new_grid)

    def map_objects(self, *args, **kwargs):
        return map_objects(self, *args, **kwargs)

    def filter(self, filter_func, background_color=0):
        """Applies a filter function to each pixel, setting non-matching pixels to background_color."""
        new_grid = self._grid.copy()
        height, width = self.grid.shape

        for r in range(height):
            for c in range(width):
                current_color = self._grid[r, c]
                if not filter_func(current_color):
                    new_grid[r, c] = background_color
        return Grid(new_grid)

    def map_pixels(self, func):
        """Applies a function to each pixel in the grid, passing x, y, and color."""
        new_grid = self._grid.copy()
        height, width = self.grid.shape

        for r in range(height):
            for c in range(width):
                current_color = self._grid[r, c]
                new_color = func(c, r, current_color)  # Pass x, y, color
                new_grid[r, c] = new_color
        return Grid(new_grid)
    def align(self, anchor_color, moving_color, direction):
        """Aligns the moving_color object relative to the anchor_color object."""
        objects = find_objects(self)
        anchor_object = next((obj for obj in objects if obj.color == anchor_color), None)
        moving_object = next((obj for obj in objects if obj.color == moving_color), None)

        if not anchor_object or not moving_object:
            return self # Return original grid if objects aren't found

        # Create a new grid, painting all objects except the one that will be moved
        new_grid = Grid(np.full(self._grid.shape, 0, dtype=int))
        for obj in objects:
            if obj.color != moving_color:
                paint_object(new_grid, obj)

        # Calculate new position
        if direction == "right":
            new_x = anchor_object.x + anchor_object.width
            new_y = anchor_object.y 
        elif direction == "left":
            new_x = anchor_object.x - moving_object.width
            new_y = anchor_object.y
        elif direction == "above":
            new_x = anchor_object.x
            new_y = anchor_object.y - moving_object.height
        elif direction == "below":
            new_x = anchor_object.x
            new_y = anchor_object.y + anchor_object.height
        else:
            return self # Unknown direction

        # Translate and paint the moving object
        dx = new_x - moving_object.x
        dy = new_y - moving_object.y
        translated_object = moving_object.translate((dy, dx))
        paint_object(new_grid, translated_object)

        return new_grid

    def paste(self, source_grid, top: Optional[int] = None, left: Optional[int] = None, *, background_color: int = 0):
        """Pastes another grid onto this one at a specified location."""
        if top is None or left is None:
            return self  # Return original grid if placement is not specified

        if not isinstance(source_grid, Grid):
            source_grid = Grid(source_grid)

        new_grid_arr = self._grid.copy()
        h, w = source_grid.shape
        dest_h, dest_w = new_grid_arr.shape

        for r in range(h):
            for c in range(w):
                dest_r = top + r
                dest_c = left + c
                if 0 <= dest_r < dest_h and 0 <= dest_c < dest_w:
                    pixel = source_grid.get(r, c)
                    # Assuming we don't paste the background color, to allow for "transparent" pixels
                    if pixel != background_color:
                        new_grid_arr[dest_r, dest_c] = pixel
        return Grid(new_grid_arr)

    def paint_object(grid, obj):
        """Paints an object onto the grid."""
        if not isinstance(grid, Grid):
            grid = Grid(grid)
        for y, x in obj.pixels:
            if 0 <= y < grid._grid.shape[0] and 0 <= x < grid._grid.shape[1]:
                grid._grid[y, x] = obj.color
class Case:
    def __init__(self, case_data):
        self._data = case_data
        self.input = Grid(case_data["input"])
        if "output" in case_data and case_data["output"] is not None:
            self.output = Grid(case_data["output"])
        else:
            self.output = None
    
    @property
    def input_grid(self):
        return self.input
    
    def __getitem__(self, key):
        if key == 'input':
            return self.input
        elif key == 'output':
            return self.output
        raise KeyError(f"Case has no key '{key}'")

    @property
    def outputs(self): return self.output
    @property
    def output_grid(self): return self.output
    @property
    def in_grid(self): return self.input

class DualAccessList(list):
    """List-like container that supports both list and dict-style access."""

    def __init__(self, case_objects, raw_dicts):
        super().__init__(case_objects)
        self._raw = list(raw_dicts)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return DualAccessList(super().__getitem__(key), self._raw[key])
        if isinstance(key, int):
            return DualAccessProxy(super().__getitem__(key), self._raw[key])
        if isinstance(key, str):
            return [DualAccessProxy(case, raw)[key] for case, raw in zip(list.__iter__(self), self._raw)]
        return super().__getitem__(key)

    def __iter__(self):
        for case_obj, raw in zip(list.__iter__(self), self._raw):
            yield DualAccessProxy(case_obj, raw)

class DualAccessProxy:
    def __init__(self, case_obj, raw_dict):
        self._case = case_obj
        self._dict = raw_dict
    
    def __getattr__(self, name):
        return getattr(self._case, name)
    
    def __getitem__(self, key):
        if key == 'input':
            return self._case.input
        if key == 'output':
            return self._case.output
        return self._dict[key]

class Task(dict):
    def __init__(self, task_data):
        super().__init__()
        self.task_id = task_data.get("task_id")
        self._raw_data = task_data
        self._train_cases: List[Case] = [Case(p) for p in task_data.get("train", [])]
        self._test_cases: List[Case] = [Case(p) for p in task_data.get("test", [])]
        self._train_view = DualAccessList(self._train_cases, task_data.get("train", []))
        self._test_view = DualAccessList(self._test_cases, task_data.get("test", []))
        self._object_reasoner = ObjectReasoner()

    def __getitem__(self, key):
        if key == "train":
            return self.train
        if key == "test":
            return self.test
        if key == "input":
            return self.input
        if key == "inputs":
            return self.inputs
        if key == "output":
            return self._train_cases[0].output if self._train_cases and self._train_cases[0].output is not None else None
        if key == "outputs":
            return [case.output for case in self._train_cases if case.output is not None]
        if key in self._raw_data:
            return self._raw_data[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def __contains__(self, key):
        return key in self._raw_data

    @property
    def train(self) -> DualAccessList:
        return self._train_view

    @property
    def test(self) -> DualAccessList:
        return self._test_view

    @property
    def test_cases(self):
        return self.test

    @property
    def test_tasks(self):
        return self.test

    @property
    def input(self):
        first = self.train[0]
        return first["input"] if isinstance(first, DualAccessProxy) else first.input

    @property
    def inputs(self):
        return [case.input for case in self._train_cases]

    @property
    def input_grid(self):
        return self._train_cases[0].input

    @property
    def input_height(self):
        return self._train_cases[0].input.height

    @property
    def input_width(self):
        return self._train_cases[0].input.width

    @property
    def grids(self):
        return [case.input for case in self._train_cases]

    def map(self, func):
        if not self._test_cases:
            return []
        result_grid = func(self._test_cases[0].input)
        if isinstance(result_grid, Grid):
            return result_grid.to_list()
        else:
            return to_list(np.array(result_grid))

    def map_input(self, func):
        return self.map(func)

    def map_inputs(self, func):
        return self.map(func)

    def map_grid(self, func):
        return self.map(func)

    def scale(self, factor):
        return self.map(lambda i: i.repeat(factor, factor))

    @property
    def train_tasks(self):
        return self.train

    @property
    def object_reasoning(self):
        return self._object_reasoner

    def i(self, index):
        return self._train_cases[index].input
    def build_grid(self, height, width, func):
        """Constructs a new grid of specified dimensions, filling each cell with a color determined by a function."""
        new_grid_arr = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                new_grid_arr[i, j] = func(i, j)
        return Grid(new_grid_arr)
    @property
    def train_input(self):
        return self._train_cases[0].input
