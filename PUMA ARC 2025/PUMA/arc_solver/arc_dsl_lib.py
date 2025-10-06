import numpy as np
from scipy.ndimage import label, find_objects as find_objects_scipy
from .grid import to_array, to_list, color_map as color_map_func, crop as crop_func, translate as translate_func, flip as flip_func

class Object:
    def __init__(self, pixels, color):
        self.pixels = pixels
        self.color = color
        self.height = np.max(pixels[:, 0]) - np.min(pixels[:, 0]) + 1
        self.width = np.max(pixels[:, 1]) - np.min(pixels[:, 1]) + 1
        self.y = np.min(pixels[:, 0])
        self.x = np.min(pixels[:, 1])

    def __repr__(self):
        return f"Object(color={self.color}, shape=({self.height}, {self.width}), top_left=({self.y}, {self.x}))"

    def bounding_box(self):
        return (self.y, self.x, self.height, self.width)

    def translate(self, offset):
        """Translate an object by an offset (dy, dx)."""
        new_pixels = self.pixels.copy()
        new_pixels[:, 0] += offset[0]  # dy
        new_pixels[:, 1] += offset[1]  # dx
        return Object(new_pixels, self.color)

    def __getitem__(self, key):
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

def find_objects(grid, color=None, ignore_color=0, min_size=1, **kwargs):
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    
    binary_grid = grid.grid != ignore_color
    if color is not None:
        binary_grid = grid.grid == color

    labeled_grid, num_labels = label(binary_grid)
    if num_labels == 0:
        return []
    
    objects = []
    slices = find_objects_scipy(labeled_grid)
    for i in range(num_labels):
        pixels = np.argwhere(labeled_grid == (i + 1))
        if len(pixels) < min_size:
            continue
        obj_color = grid.grid[pixels[0][0], pixels[0][1]]
        objects.append(Object(pixels, obj_color))
        
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

def map_objects(grid, func, filter_func=None, **kwargs):
    """Apply a function to each object in the grid and compose the results."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)

    all_objects = find_objects(grid, **kwargs)
    objects_to_map = [obj for obj in all_objects if filter_func(obj)] if filter_func else all_objects

    result_grid = grid.copy()

    for obj in objects_to_map:
        transformed = func(obj)
        
        def compose_grid(base, overlay):
            if not isinstance(base, Grid):
                base = Grid(base)
            if not isinstance(overlay, Grid):
                overlay = Grid(overlay)
            # Create a copy of the base grid to modify
            new_grid_array = base.grid.copy()
            # Find non-background pixels in the overlay
            mask = overlay.grid != 0
            # Place them onto the new grid
            new_grid_array[mask] = overlay.grid[mask]
            return Grid(new_grid_array)

        if isinstance(transformed, Grid):
            result_grid = compose_grid(result_grid, transformed)
        elif isinstance(transformed, (list, tuple)):
            for item in transformed:
                if isinstance(item, Grid):
                    result_grid = compose_grid(result_grid, item)

    return result_grid

def paint(grid, color, mask):
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    new_grid = grid.grid.copy()
    new_grid[mask] = color
    return Grid(new_grid)

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

def copy(grid, *args, **kwargs):
    if isinstance(grid, Grid):
        return grid.copy()
    return Grid(grid)

def copy_grid(grid):
    print(f"DEBUG: copy_grid received grid type: {type(grid)}")
    result = grid.copy()
    print(f"DEBUG: copy_grid returning type: {type(result)}")
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

def paint_diagonal(grid, *args, **kwargs):
    raise NotImplementedError("paint_diagonal is not implemented")

def fill_rectangle(target, color):
    """Fills the bounding box of an Object or the entire Grid with a specified color."""
    if isinstance(target, Object):
        y, x, h, w = target.bounding_box()
        new_grid_arr = np.full((h, w), color, dtype=int)
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

def crop(grid, top, left, height, width):
    """Crop a grid to the specified rectangle."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    return grid.crop(top, left, height, width)

def row(grid, index):
    """Get a specific row from the grid."""
    if not isinstance(grid, Grid):
        grid = Grid(grid)
    return grid.grid[index]

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

class Grid:
    def __init__(self, grid_data):
        print(f"DEBUG: Grid constructor called with grid_data type: {type(grid_data)}")
        if isinstance(grid_data, Grid):
            self.grid = grid_data.grid.copy()
        else:
            self.grid = to_array(grid_data)

    @property
    def shape(self):
        return self.grid.shape

    @property
    def height(self):
        return self.grid.shape[0]

    @property
    def width(self):
        return self.grid.shape[1]

    @property
    def objects(self):
        return find_objects(self)

    def to_list(self):
        return self.grid.tolist()

    def __array__(self):
        return self.grid

    def __len__(self):
        return len(self.grid)

    def __iter__(self):
        return iter(self.grid)

    def __getitem__(self, item):
        return self.grid[item]

    def __setitem__(self, key, value):
        self.grid[key] = value

    def get(self, r, c):
        return self.grid[r, c]

    def replace(self, old, new):
        return self.replace_color(old, new)

    def replace_color(self, old_color, new_color, mask=None):
        mapping = {old_color: new_color}
        new_grid = self.grid.copy()
        if mask is not None:
            for r in range(self.height):
                for c in range(self.width):
                    if mask[r, c] and self.grid[r, c] == old_color:
                        new_grid[r, c] = new_color
            return Grid(new_grid)
        else:
            return Grid(color_map_func(self.grid, mapping))

    def map_colors(self, mapping):
        new_grid_data = color_map_func(self.grid, mapping)
        return Grid(new_grid_data)

    def repeat(self, factor_h, factor_w):
        new_grid_data = np.tile(self.grid, (factor_h, factor_w))
        return Grid(new_grid_data)

    def map_color(self, old, new):
        return self.replace_color(old, new)

    def fill_regions(self):
        labeled_grid, num_labels = label(self.grid)
        if num_labels == 0:
            return self
        for i in range(1, num_labels + 1):
            component_mask = (labeled_grid == i)
            colors, counts = np.unique(self.grid[component_mask], return_counts=True)
            if len(colors) > 0:
                dominant_color = colors[np.argmax(counts)]
                self.grid[component_mask] = dominant_color
        return Grid(self.grid)

    def expand_grid(self, factor):
        new_grid_data = np.kron(self.grid, np.ones((factor, factor)))
        return Grid(new_grid_data)

    def crop(self, top, left, height, width):
        new_grid_data = crop_func(self.grid, top, left, height, width)
        return Grid(new_grid_data)

    def translate(self, dx, dy, fill_value=0):
        return Grid(translate_func(self.grid, dy, dx, fill=fill_value))

    def compose(self, *funcs):
        res = self
        for f in funcs:
            res = f(res)
        return res

    def eq(self, other):
        if isinstance(other, Grid):
            return np.array_equal(self.grid, other.grid)
        return np.array_equal(self.grid, other)

    def map(self, func):
        new_grid = np.vectorize(func)(self.grid)
        return Grid(new_grid)

    def subgrid(self, r, c, h, w):
        return Grid(self.grid[r:r+h, c:c+w])

    def expand_to_grid(self, width, height, **kwargs):
        """Tiles the current grid to fill a new grid of specified dimensions."""
        new_grid_arr = np.zeros((height, width), dtype=self.grid.dtype)
        h, w = self.shape

        for r in range(0, height, h):
            for c in range(0, width, w):
                # Determine the piece of the source grid to copy
                block_h = min(h, height - r)
                block_w = min(w, width - c)
                source_block = self.grid[:block_h, :block_w]
                # Place it in the new grid
                new_grid_arr[r:r+block_h, c:c+block_w] = source_block
        
        return Grid(new_grid_arr)

    def expand_square(self, factor):
        """Expands each pixel into a square block of size factor x factor."""
        new_grid_data = np.kron(self.grid, np.ones((factor, factor), dtype=self.grid.dtype))
        return Grid(new_grid_data)

    def fill_gaps(self, background_color=0):
        """Fills holes within objects with the color of the surrounding object."""
        from collections import deque

        new_grid = self.grid.copy()
        height, width = self.grid.shape
        visited = np.zeros_like(self.grid, dtype=bool)

        # 1. Find all exterior background pixels by flood-filling from the border
        q = deque()
        for r in range(height):
            if self.grid[r, 0] == background_color:
                q.append((r, 0))
                visited[r, 0] = True
            if self.grid[r, width - 1] == background_color:
                q.append((r, width - 1))
                visited[r, width - 1] = True
        for c in range(width):
            if self.grid[0, c] == background_color:
                q.append((0, c))
                visited[0, c] = True
            if self.grid[height - 1, c] == background_color:
                q.append((height - 1, c))
                visited[height - 1, c] = True

        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and self.grid[nr, nc] == background_color:
                    visited[nr, nc] = True
                    q.append((nr, nc))

        # 2. Iterate and fill any unvisited background pixels (which are holes)
        for r in range(height):
            for c in range(width):
                if self.grid[r, c] == background_color and not visited[r, c]:
                    # This is a hole. Find the surrounding color.
                    fill_color = -1
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width and self.grid[nr, nc] != background_color:
                            fill_color = self.grid[nr, nc]
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
                                if 0 <= nhr < height and 0 <= nhc < width and not visited[nhr, nhc] and self.grid[nhr, nhc] == background_color:
                                    visited[nhr, nhc] = True
                                    new_grid[nhr, nhc] = fill_color
                                    hole_q.append((nhr, nhc))
        return Grid(new_grid)

    def map_rows(self, func): 
        new_grid = self.grid.copy()
        for r in range(self.height):
            new_grid[r] = func(self.grid[r])
        return Grid(new_grid)


    def copy(self, *args, **kwargs): return self

    def fill_until_collision(self, background_color=0):
        """Performs a multi-source flood fill from all non-background pixels."""
        new_grid = self.grid.copy()
        height, width = self.grid.shape

        queue = []
        # Initialize queue with all non-background pixels
        for r in range(height):
            for c in range(width):
                if self.grid[r, c] != background_color:
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
        new_grid_rows = [self.grid[r] for r in range(self.height) if func(r)]
        if not new_grid_rows:
            return Grid(np.array([[]]))
        return Grid(np.vstack(new_grid_rows))

    def paint_rows(self, filter_func, color):
        """Paints rows selected by a filter function with a specific color."""
        new_grid = self.grid.copy()
        for r in range(self.height):
            if filter_func(r):
                new_grid[r, :] = color
        return Grid(new_grid)

    def pattern_map(self, pattern=None, replacement=None, key=None, values=None, default=None, **kwargs):
        """Maps patterns in the grid to new values."""
        if 'pattern_width' in kwargs or pattern is None:
            # Handle the simple/ambiguous case by doing nothing
            return self

        new_grid = self.grid.copy()
        pattern_height = len(pattern)
        pattern_width = len(pattern[0])

        for r in range(self.height - pattern_height + 1):
            for c in range(self.width - pattern_width + 1):
                view = self.grid[r:r+pattern_height, c:c+pattern_width]
                
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

    def map_objects(self, func, filter_func=None, **kwargs):
        return map_objects(self, func, filter_func=filter_func, **kwargs)

    def filter(self, filter_func, background_color=0):
        """Applies a filter function to each pixel, setting non-matching pixels to background_color."""
        new_grid = self.grid.copy()
        height, width = self.grid.shape

        for r in range(height):
            for c in range(width):
                current_color = self.grid[r, c]
                if not filter_func(current_color):
                    new_grid[r, c] = background_color
        return Grid(new_grid)

    def map_pixels(self, func):
        """Applies a function to each pixel in the grid, passing x, y, and color."""
        new_grid = self.grid.copy()
        height, width = self.grid.shape

        for r in range(height):
            for c in range(width):
                current_color = self.grid[r, c]
                new_color = func(c, r, current_color) # Pass x, y, color
                new_grid[r, c] = new_color
        return Grid(new_grid)
    def pattern_fill(self, *args, **kwargs): raise NotImplementedError(".pattern_fill() is not implemented")
    def align(self, anchor_color, moving_color, direction):
        """Aligns the moving_color object relative to the anchor_color object."""
        objects = find_objects(self)
        anchor_object = next((obj for obj in objects if obj.color == anchor_color), None)
        moving_object = next((obj for obj in objects if obj.color == moving_color), None)

        if not anchor_object or not moving_object:
            return self # Return original grid if objects aren't found

        # Create a new grid, painting all objects except the one that will be moved
        new_grid = Grid(np.full(self.grid.shape, 0, dtype=int))
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

    def paint_object(grid, obj):
        """Paints an object onto the grid."""
        if not isinstance(grid, Grid):
            grid = Grid(grid)
        for y, x in obj.pixels:
            if 0 <= y < grid.grid.shape[0] and 0 <= x < grid.grid.shape[1]:
                grid.grid[y, x] = obj.color

class Task:
    """A wrapper for a single ARC task, providing convenient access to train/test cases."""


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
            return self._data['input']
        elif key == 'output':
            return self._data.get('output')
        raise KeyError(f"Case has no key '{key}'")

    @property
    def outputs(self): return self.output
    @property
    def output_grid(self): return self.output
    @property
    def in_grid(self): return self.input

class DualAccessList(list):
    def __init__(self, case_objects, raw_dicts):
        super().__init__(case_objects)
        self._raw = raw_dicts
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if isinstance(index, int):
            return DualAccessProxy(item, self._raw[index])
        return item

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
        dict.__init__(self)
        self.task_id = task_data.get("task_id")
        self.train = [Case(p) for p in task_data.get("train", [])]
        self.test = [Case(p) for p in task_data.get("test", [])]
        self._raw_data = task_data

    def __getitem__(self, key):
        if key == 'train':
            return DualAccessList(self.train, self._raw_data.get('train', []))
        elif key == 'test':
            return DualAccessList(self.test, self._raw_data.get('test', []))
        elif key in self._raw_data:
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
    def test_cases(self):
        return self.test

    @property
    def input(self):
        return self.train[0].input

    @property
    def inputs(self):
        return [p.input for p in self.train]

    @property
    def input_grid(self):
        return self.train[0].input

    @property
    def input_height(self):
        return self.train[0].input.height

    @property
    def input_width(self):
        return self.train[0].input.width

    @property
    def grids(self):
        return [case.input for case in self.train]

    def map(self, func):
        if not self.test:
            return []
        result_grid = func(self.test[0].input)
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
    def train_tasks(self): return self.train
    @property
    def object_reasoning(self): return None
    def i(self, index): return self.train[index].input
    def build_grid(self, height, width, func):
        """Constructs a new grid of specified dimensions, filling each cell with a color determined by a function."""
        new_grid_arr = np.zeros((height, width), dtype=int)
        for i in range(height):
            for j in range(width):
                new_grid_arr[i, j] = func(i, j)
        return Grid(new_grid_arr)
    @property
    def train_input(self): return self.train[0].input