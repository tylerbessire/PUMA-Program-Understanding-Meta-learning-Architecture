"""
Object-based RFT reasoning for ARC tasks.

Converts grids to objects, builds spatial relations, and reasons at object level.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .grid import Array


@dataclass
class ARCObject:
    """A discrete object extracted from the grid."""
    id: int
    color: int
    positions: Set[Tuple[int, int]]  # Set of (row, col) positions
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    shape_type: str  # 'rectangle', 'line', 'L_shape', 'cross', 'single', 'irregular'
    size: int
    descriptors: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Center point of the object."""
        positions = list(self.positions)
        avg_row = sum(pos[0] for pos in positions) / len(positions)
        avg_col = sum(pos[1] for pos in positions) / len(positions)
        return (avg_row, avg_col)
    
    @property
    def width(self) -> int:
        return self.bounding_box[3] - self.bounding_box[1] + 1
    
    @property
    def height(self) -> int:
        return self.bounding_box[2] - self.bounding_box[0] + 1


@dataclass
class SpatialRelation:
    """Spatial relationship between two objects."""
    obj1_id: int
    obj2_id: int
    relation: str  # 'left_of', 'right_of', 'above', 'below', 'inside', 'touching', 'aligned_horizontal', 'aligned_vertical'
    distance: float
    confidence: float


@dataclass
class ObjectTransformation:
    """A transformation applied to objects."""
    operation: str  # 'move', 'copy', 'delete', 'recolor', 'resize', 'rotate'
    obj_selector: Dict[str, Any]  # How to select objects {'color': 3, 'shape': 'rectangle'}
    parameters: Dict[str, Any]  # Operation parameters {'direction': 'right', 'distance': 3}
    target_region: Optional[Tuple[int, int, int, int]] = None  # Where to apply the transformation


class ObjectExtractor:
    """Extracts discrete objects from ARC grids."""
    
    def extract_objects(self, grid: Array, ignore_color: int = 0) -> List[ARCObject]:
        """Extract all objects from a grid using connected components."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        obj_id = 0
        
        h, w = grid.shape
        
        for r in range(h):
            for c in range(w):
                if not visited[r, c] and grid[r, c] != ignore_color:
                    # Found new object - flood fill to get all positions
                    color = grid[r, c]
                    positions = self._flood_fill(grid, visited, r, c, color)
                    
                    if len(positions) > 0:
                        obj = self._create_object(grid, obj_id, color, positions)
                        objects.append(obj)
                        obj_id += 1
        
        return objects
    
    def _flood_fill(self, grid: Array, visited: np.ndarray, start_r: int, start_c: int, color: int) -> Set[Tuple[int, int]]:
        """Flood fill to find connected component."""
        h, w = grid.shape
        positions = set()
        stack = [(start_r, start_c)]
        
        while stack:
            r, c = stack.pop()
            if (r < 0 or r >= h or c < 0 or c >= w or 
                visited[r, c] or grid[r, c] != color):
                continue
                
            visited[r, c] = True
            positions.add((r, c))
            
            # Add 4-connected neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((r + dr, c + dc))
        
        return positions
    
    def _create_object(
        self,
        grid: Array,
        obj_id: int,
        color: int,
        positions: Set[Tuple[int, int]],
    ) -> ARCObject:
        """Create object from positions."""
        pos_list = list(positions)
        min_r = min(pos[0] for pos in pos_list)
        max_r = max(pos[0] for pos in pos_list)
        min_c = min(pos[1] for pos in pos_list)
        max_c = max(pos[1] for pos in pos_list)
        
        bounding_box = (min_r, min_c, max_r, max_c)
        shape_type = self._classify_shape(positions, bounding_box)
        
        descriptors = self._compute_descriptors(grid, positions, bounding_box)

        return ARCObject(
            id=obj_id,
            color=color,
            positions=positions,
            bounding_box=bounding_box,
            shape_type=shape_type,
            size=len(positions),
            descriptors=descriptors,
        )
    
    def _classify_shape(self, positions: Set[Tuple[int, int]], bbox: Tuple[int, int, int, int]) -> str:
        """Classify the shape of an object."""
        size = len(positions)
        min_r, min_c, max_r, max_c = bbox
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        expected_size = width * height
        
        if size == 1:
            return 'single'
        elif size == expected_size:
            if width == 1 or height == 1:
                return 'line'
            else:
                return 'rectangle'
        elif size == width + height - 1:  # L-shape pattern
            return 'L_shape'
        elif size == width + height - 1 and width > 2 and height > 2:  # Cross pattern
            return 'cross'
        else:
            return 'irregular'

    def _compute_descriptors(
        self,
        grid: Array,
        positions: Set[Tuple[int, int]],
        bbox: Tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """Derive contextual descriptors used by higher-level reasoning."""

        min_r, min_c, max_r, max_c = bbox
        h, w = grid.shape
        patch = grid[min_r:max_r + 1, min_c:max_c + 1]

        # Border vs interior palettes
        border_mask = np.zeros_like(patch, dtype=bool)
        border_mask[0, :] = border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        border_colors = np.unique(patch[border_mask]).tolist()
        interior_colors = np.unique(patch[~border_mask]).tolist() if patch.size > border_mask.sum() else []

        # Symmetry flags
        horizontal_sym = bool(patch.shape[1] > 1 and np.array_equal(patch, np.fliplr(patch)))
        vertical_sym = bool(patch.shape[0] > 1 and np.array_equal(patch, np.flipud(patch)))

        # Stripe summaries
        row_stripes = [tuple(np.unique(row)) for row in patch]
        col_stripes = [tuple(np.unique(col)) for col in patch.T]

        # Distances to borders
        dist_top = min_r
        dist_left = min_c
        dist_bottom = (h - 1) - max_r
        dist_right = (w - 1) - max_c

        touches_border = dist_top == 0 or dist_left == 0 or dist_bottom == 0 or dist_right == 0

        descriptors: Dict[str, Any] = {
            "border_colors": border_colors,
            "interior_colors": interior_colors,
            "row_stripes": row_stripes,
            "column_stripes": col_stripes,
            "symmetry_horizontal": horizontal_sym,
            "symmetry_vertical": vertical_sym,
            "distance_top": dist_top,
            "distance_left": dist_left,
            "distance_bottom": dist_bottom,
            "distance_right": dist_right,
            "touches_border": touches_border,
        }

        return descriptors


class SpatialAnalyzer:
    """Analyzes spatial relationships between objects."""
    
    def analyze_relations(self, objects: List[ARCObject]) -> List[SpatialRelation]:
        """Find all spatial relationships between objects."""
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicates and self-relations
                    continue
                
                obj_relations = self._find_relations_between(obj1, obj2)
                relations.extend(obj_relations)
        
        return relations
    
    def _find_relations_between(self, obj1: ARCObject, obj2: ARCObject) -> List[SpatialRelation]:
        """Find spatial relations between two objects."""
        relations = []
        
        # Get centers
        c1_r, c1_c = obj1.center
        c2_r, c2_c = obj2.center
        
        # Calculate distance
        distance = np.sqrt((c1_r - c2_r)**2 + (c1_c - c2_c)**2)
        
        # Directional relations
        horizontal_threshold = max(obj1.height, obj2.height) / 2
        vertical_threshold = max(obj1.width, obj2.width) / 2
        
        if abs(c1_r - c2_r) < horizontal_threshold:  # Roughly same row
            if c1_c < c2_c:
                relations.append(SpatialRelation(obj1.id, obj2.id, 'left_of', distance, 0.8))
            else:
                relations.append(SpatialRelation(obj1.id, obj2.id, 'right_of', distance, 0.8))
        
        if abs(c1_c - c2_c) < vertical_threshold:  # Roughly same column
            if c1_r < c2_r:
                relations.append(SpatialRelation(obj1.id, obj2.id, 'above', distance, 0.8))
            else:
                relations.append(SpatialRelation(obj1.id, obj2.id, 'below', distance, 0.8))
        
        # Containment
        if self._is_inside(obj1, obj2):
            relations.append(SpatialRelation(obj1.id, obj2.id, 'inside', 0, 0.9))
        elif self._is_inside(obj2, obj1):
            relations.append(SpatialRelation(obj2.id, obj1.id, 'inside', 0, 0.9))
        
        # Touching
        if self._are_touching(obj1, obj2):
            relations.append(SpatialRelation(obj1.id, obj2.id, 'touching', 0, 0.9))
        
        return relations
    
    def _is_inside(self, inner: ARCObject, outer: ARCObject) -> bool:
        """Check if inner object is inside outer object."""
        inner_bbox = inner.bounding_box
        outer_bbox = outer.bounding_box
        
        return (inner_bbox[0] >= outer_bbox[0] and inner_bbox[1] >= outer_bbox[1] and
                inner_bbox[2] <= outer_bbox[2] and inner_bbox[3] <= outer_bbox[3])
    
    def _are_touching(self, obj1: ARCObject, obj2: ARCObject) -> bool:
        """Check if objects are touching (adjacent)."""
        for pos1 in obj1.positions:
            for pos2 in obj2.positions:
                # Check if positions are adjacent (1 unit apart)
                if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                    return True
        return False


class ObjectReasoner:
    """RFT-based reasoning over objects and their relationships."""
    
    def __init__(self):
        self.extractor = ObjectExtractor()
        self.analyzer = SpatialAnalyzer()
    
    def analyze_transformation(self, input_grid: Array, output_grid: Array) -> List[ObjectTransformation]:
        """Analyze what object-level transformations occurred."""
        # Extract objects from both grids
        input_objects = self.extractor.extract_objects(input_grid)
        output_objects = self.extractor.extract_objects(output_grid)
        
        # Find the transformations
        transformations = []
        
        # Simple heuristics for common transformations
        transformations.extend(self._detect_movement(input_objects, output_objects))
        transformations.extend(self._detect_copying(input_objects, output_objects))
        transformations.extend(self._detect_recoloring(input_objects, output_objects))
        transformations.extend(self._detect_region_filling(input_grid, output_grid, input_objects, output_objects))
        
        return transformations
    
    def _detect_movement(self, input_objs: List[ARCObject], output_objs: List[ARCObject]) -> List[ObjectTransformation]:
        """Detect object movement patterns."""
        transformations = []
        
        # Match objects by color and shape
        for inp_obj in input_objs:
            best_match = None
            best_score = 0
            
            for out_obj in output_objs:
                if (inp_obj.color == out_obj.color and 
                    inp_obj.shape_type == out_obj.shape_type and
                    inp_obj.size == out_obj.size):
                    
                    # Calculate similarity score
                    score = 0.8  # Base score for matching properties
                    best_match = out_obj
                    best_score = score
            
            if best_match and best_score > 0.7:
                # Calculate movement
                inp_center = inp_obj.center
                out_center = best_match.center
                
                dr = out_center[0] - inp_center[0]
                dc = out_center[1] - inp_center[1]
                
                if abs(dr) > 0.5 or abs(dc) > 0.5:  # Significant movement
                    direction = self._get_direction(dr, dc)
                    distance = np.sqrt(dr**2 + dc**2)
                    
                    transformations.append(ObjectTransformation(
                        operation='move',
                        obj_selector={'color': inp_obj.color, 'shape': inp_obj.shape_type},
                        parameters={'direction': direction, 'distance': distance, 'dr': dr, 'dc': dc}
                    ))
        
        return transformations
    
    def _detect_copying(self, input_objs: List[ARCObject], output_objs: List[ARCObject]) -> List[ObjectTransformation]:
        """Detect object copying/duplication patterns."""
        transformations = []
        
        # Count objects by type
        input_counts = defaultdict(int)
        output_counts = defaultdict(int)
        
        for obj in input_objs:
            key = (obj.color, obj.shape_type, obj.size)
            input_counts[key] += 1
        
        for obj in output_objs:
            key = (obj.color, obj.shape_type, obj.size)
            output_counts[key] += 1
        
        # Find increased counts
        for key, out_count in output_counts.items():
            in_count = input_counts.get(key, 0)
            if out_count > in_count:
                color, shape, size = key
                transformations.append(ObjectTransformation(
                    operation='copy',
                    obj_selector={'color': color, 'shape': shape, 'size': size},
                    parameters={'copies': out_count - in_count}
                ))
        
        return transformations
    
    def _detect_recoloring(self, input_objs: List[ARCObject], output_objs: List[ARCObject]) -> List[ObjectTransformation]:
        """Detect recoloring patterns."""
        transformations = []
        
        # Simple recoloring detection
        for inp_obj in input_objs:
            for out_obj in output_objs:
                if (inp_obj.positions == out_obj.positions and 
                    inp_obj.color != out_obj.color):
                    
                    transformations.append(ObjectTransformation(
                        operation='recolor',
                        obj_selector={'color': inp_obj.color, 'shape': inp_obj.shape_type},
                        parameters={'new_color': out_obj.color}
                    ))
        
        return transformations
    
    def _detect_region_filling(self, input_grid: Array, output_grid: Array, 
                             input_objs: List[ARCObject], output_objs: List[ARCObject]) -> List[ObjectTransformation]:
        """Detect filling of rectangular regions (like our 8-filled regions)."""
        transformations = []
        
        # Look for rectangular regions in input that got filled differently in output
        for inp_obj in input_objs:
            if inp_obj.shape_type == 'rectangle' and inp_obj.size > 4:  # Rectangular placeholders
                # Check if this region changed significantly in output
                bbox = inp_obj.bounding_box
                
                # Bounds check
                if (bbox[2] >= output_grid.shape[0] or bbox[3] >= output_grid.shape[1]):
                    continue
                    
                input_region = input_grid[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
                output_region = output_grid[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
                
                if not np.array_equal(input_region, output_region):
                    # Find where the content comes from (symmetric extraction)
                    source_content = self._find_replacement_content(input_grid, bbox, output_region.shape)
                    
                    transformations.append(ObjectTransformation(
                        operation='replace_region',
                        obj_selector={'color': inp_obj.color, 'shape': 'rectangle', 'min_size': 4},
                        parameters={
                            'target_region': bbox,
                            'replacement_strategy': 'symmetric_extraction',
                            'source_content': source_content
                        },
                        target_region=bbox
                    ))
        
        # Also detect size-change extractions (30x30 -> 9x4)
        if input_grid.shape != output_grid.shape:
            if np.prod(output_grid.shape) < np.prod(input_grid.shape):
                # Extraction transformation
                extraction_source = self._find_extraction_source(input_grid, output_grid)
                if extraction_source:
                    transformations.append(ObjectTransformation(
                        operation='extract_region',
                        obj_selector={'pattern': 'size_reduction'},
                        parameters={
                            'source_region': extraction_source,
                            'target_shape': output_grid.shape,
                            'extraction_method': 'pattern_match'
                        }
                    ))
        
        return transformations
    
    def _find_replacement_content(self, grid: Array, target_bbox: Tuple[int, int, int, int], 
                                target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find content to replace a region with using spatial reasoning."""
        h, w = grid.shape
        target_h, target_w = target_shape
        min_r, min_c, max_r, max_c = target_bbox
        
        # Try symmetric/mirrored positions
        center_col = w // 2
        center_row = h // 2
        
        # Horizontal mirror
        mirror_min_c = center_col - (max_c - center_col)
        mirror_max_c = center_col - (min_c - center_col)
        
        if mirror_min_c >= 0 and mirror_max_c < w and mirror_max_c - mirror_min_c + 1 == target_w:
            mirror_content = grid[min_r:max_r+1, mirror_min_c:mirror_max_c+1]
            if mirror_content.shape == target_shape and not np.all(mirror_content == grid[min_r, min_c]):
                return mirror_content
        
        # Try adjacent regions with same dimensions
        candidates = []
        
        # Left adjacent
        if min_c - target_w >= 0:
            left_content = grid[min_r:min_r+target_h, min_c-target_w:min_c]
            if left_content.shape == target_shape:
                candidates.append(left_content)
        
        # Right adjacent
        if max_c + target_w < w:
            right_content = grid[min_r:min_r+target_h, max_c+1:max_c+1+target_w]
            if right_content.shape == target_shape:
                candidates.append(right_content)
        
        # Return most diverse candidate
        best_content = None
        best_diversity = 0
        for candidate in candidates:
            diversity = len(np.unique(candidate))
            if diversity > best_diversity and diversity > 1:
                best_diversity = diversity
                best_content = candidate
        
        return best_content
    
    def _find_extraction_source(self, input_grid: Array, output_grid: Array) -> Optional[Tuple[int, int, int, int]]:
        """Find where the output content was extracted from in the input."""
        out_h, out_w = output_grid.shape
        in_h, in_w = input_grid.shape
        
        best_match_pos = None
        best_match_score = 0
        
        # ENHANCED: Multiple search strategies
        
        # Strategy 1: Direct pattern matching
        for r in range(in_h - out_h + 1):
            for c in range(in_w - out_w + 1):
                input_region = input_grid[r:r+out_h, c:c+out_w]
                
                # Calculate similarity
                matches = np.sum(input_region == output_grid)
                total = output_grid.size
                score = matches / total
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_pos = (r, c, r+out_h-1, c+out_w-1)
        
        # Strategy 2: Look for regions with same shape and complexity
        if best_match_score < 0.8:
            best_alternative = self._find_complex_extraction_source(input_grid, output_grid)
            if best_alternative and best_alternative[1] > best_match_score:
                best_match_pos = best_alternative[0]
                best_match_score = best_alternative[1]
        
        # Return if we found a good match (>30% similarity, lowered threshold)
        if best_match_score > 0.3:
            return best_match_pos
        
        return None
    
    def _find_complex_extraction_source(self, input_grid: Array, output_grid: Array) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        """Find extraction source using pattern complexity matching."""
        out_h, out_w = output_grid.shape
        in_h, in_w = input_grid.shape
        
        # Calculate output pattern characteristics
        output_colors = len(np.unique(output_grid))
        output_entropy = self._calculate_pattern_entropy(output_grid)
        
        best_match = None
        best_score = 0
        
        for r in range(in_h - out_h + 1):
            for c in range(in_w - out_w + 1):
                input_region = input_grid[r:r+out_h, c:c+out_w]
                
                # Skip regions dominated by single color (like 8-filled regions)
                if np.max(np.bincount(input_region.flatten())) > input_region.size * 0.7:
                    continue
                
                # Calculate pattern characteristics
                region_colors = len(np.unique(input_region))
                region_entropy = self._calculate_pattern_entropy(input_region)
                
                # Score based on pattern similarity
                color_similarity = 1.0 - abs(output_colors - region_colors) / max(output_colors, region_colors)
                entropy_similarity = 1.0 - abs(output_entropy - region_entropy) / max(output_entropy, region_entropy)
                
                combined_score = (color_similarity + entropy_similarity) / 2
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (r, c, r+out_h-1, c+out_w-1)
        
        return (best_match, best_score) if best_match else None
    
    def _calculate_pattern_entropy(self, grid: Array) -> float:
        """Calculate pattern entropy/complexity of a grid region."""
        if grid.size == 0:
            return 0.0
        
        # Calculate color distribution entropy
        unique, counts = np.unique(grid, return_counts=True)
        probabilities = counts / grid.size
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def _get_direction(self, dr: float, dc: float) -> str:
        """Convert movement vector to direction."""
        if abs(dr) > abs(dc):
            return 'down' if dr > 0 else 'up'
        else:
            return 'right' if dc > 0 else 'left'
    
    def apply_transformation(self, grid: Array, transformation: ObjectTransformation) -> Array:
        """Apply an object transformation to a grid."""
        result = grid.copy()
        
        # ENHANCED: Handle extraction operations first (they change grid size)
        if transformation.operation == 'extract_region':
            return self._apply_extract_region(result, transformation.parameters)
        
        # For other operations, extract objects
        objects = self.extractor.extract_objects(grid)
        
        # Find objects matching the selector
        matching_objects = []
        for obj in objects:
            if self._matches_selector(obj, transformation.obj_selector):
                matching_objects.append(obj)
        
        # Apply transformation to matching objects
        for obj in matching_objects:
            if transformation.operation == 'move':
                result = self._apply_move(result, obj, transformation.parameters)
            elif transformation.operation == 'copy':
                result = self._apply_copy(result, obj, transformation.parameters)
            elif transformation.operation == 'recolor':
                result = self._apply_recolor(result, obj, transformation.parameters)
            elif transformation.operation == 'replace_region':
                result = self._apply_replace_region(result, obj, transformation.parameters)
        
        # Handle other non-object-specific transformations
        if transformation.operation == 'fill_region':
            result = self._apply_fill_region(result, transformation)
        
        return result
    
    def _matches_selector(self, obj: ARCObject, selector: Dict[str, Any]) -> bool:
        """Check if object matches selector criteria."""
        for key, value in selector.items():
            if key == 'color' and obj.color != value:
                return False
            elif key == 'shape' and obj.shape_type != value:
                return False
            elif key == 'size' and obj.size != value:
                return False
            elif key == 'min_size' and obj.size < value:
                return False
            elif key == 'max_size' and obj.size > value:
                return False
            # Skip special keys that don't match object properties
            elif key in ['pattern', 'min_size', 'max_size']:
                continue
        return True
    
    def _apply_move(self, grid: Array, obj: ARCObject, params: Dict[str, Any]) -> Array:
        """Apply movement transformation."""
        result = grid.copy()
        
        # Clear original positions
        for pos in obj.positions:
            result[pos] = 0
        
        # Move to new positions
        dr = params.get('dr', 0)
        dc = params.get('dc', 0)
        
        for pos in obj.positions:
            new_r = int(pos[0] + dr)
            new_c = int(pos[1] + dc)
            
            # Check bounds
            if 0 <= new_r < result.shape[0] and 0 <= new_c < result.shape[1]:
                result[new_r, new_c] = obj.color
        
        return result
    
    def _apply_copy(self, grid: Array, obj: ARCObject, params: Dict[str, Any]) -> Array:
        """Apply copying transformation."""
        # For now, just return original grid
        # Real implementation would place copies in logical positions
        return grid
    
    def _apply_recolor(self, grid: Array, obj: ARCObject, params: Dict[str, Any]) -> Array:
        """Apply recoloring transformation."""
        result = grid.copy()
        new_color = params.get('new_color', obj.color)
        
        for pos in obj.positions:
            result[pos] = new_color
        
        return result
    
    def _apply_replace_region(self, grid: Array, obj: ARCObject, params: Dict[str, Any]) -> Array:
        """Apply region replacement transformation."""
        result = grid.copy()
        
        target_region = params.get('target_region')
        source_content = params.get('source_content')
        
        if target_region and source_content is not None:
            min_r, min_c, max_r, max_c = target_region
            
            # Bounds check
            if (max_r < result.shape[0] and max_c < result.shape[1] and 
                min_r >= 0 and min_c >= 0):
                
                # Ensure source content fits
                target_h, target_w = max_r - min_r + 1, max_c - min_c + 1
                
                if hasattr(source_content, 'shape'):
                    src_h, src_w = source_content.shape
                    if src_h == target_h and src_w == target_w:
                        result[min_r:max_r+1, min_c:max_c+1] = source_content
                    else:
                        # Resize or tile to fit
                        if src_h <= target_h and src_w <= target_w:
                            # Center the content
                            offset_r = (target_h - src_h) // 2
                            offset_c = (target_w - src_w) // 2
                            result[min_r + offset_r:min_r + offset_r + src_h,
                                   min_c + offset_c:min_c + offset_c + src_w] = source_content
        
        return result
    
    def _apply_extract_region(self, grid: Array, params: Dict[str, Any]) -> Array:
        """Apply region extraction transformation."""
        source_region = params.get('source_region')
        target_shape = params.get('target_shape')
        
        if source_region and target_shape:
            min_r, min_c, max_r, max_c = source_region
            
            # Bounds check
            if (max_r < grid.shape[0] and max_c < grid.shape[1] and 
                min_r >= 0 and min_c >= 0):
                
                extracted = grid[min_r:max_r+1, min_c:max_c+1].copy()
                
                # Resize to target shape if needed
                if extracted.shape != target_shape:
                    target_h, target_w = target_shape
                    src_h, src_w = extracted.shape
                    
                    if src_h >= target_h and src_w >= target_w:
                        # Crop to target size
                        start_r = (src_h - target_h) // 2
                        start_c = (src_w - target_w) // 2
                        extracted = extracted[start_r:start_r+target_h, start_c:start_c+target_w]
                    else:
                        # Pad to target size
                        result = np.zeros(target_shape, dtype=extracted.dtype)
                        start_r = (target_h - src_h) // 2
                        start_c = (target_w - src_w) // 2
                        result[start_r:start_r+src_h, start_c:start_c+src_w] = extracted
                        extracted = result
                
                return extracted
        
        # Fallback: return original grid
        return grid
    
    def _apply_fill_region(self, grid: Array, transformation: ObjectTransformation) -> Array:
        """Fill a region with extracted content."""
        result = grid.copy()
        
        if transformation.target_region:
            # This is where we'd apply sophisticated filling logic
            # For now, use simple pattern based on the grid structure
            bbox = transformation.target_region
            h, w = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            
            # Try to find a pattern to fill with (simplified heuristic)
            # In a real implementation, this would use the comprehensive memory
            # and pattern matching to find the right content
            
            # For demonstration: fill with a simple pattern
            for r in range(bbox[0], bbox[2] + 1):
                for c in range(bbox[1], bbox[3] + 1):
                    # Simple alternating pattern (placeholder)
                    if (r + c) % 2 == 0:
                        result[r, c] = 3
                    else:
                        result[r, c] = 1
        
        return result


class ObjectHypothesisGenerator:
    """Generates object-level hypotheses for transformations."""
    
    def __init__(self):
        self.reasoner = ObjectReasoner()
    
    def generate_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[ObjectTransformation]:
        """Generate object-level transformation hypotheses from training examples."""
        all_transformations = []
        
        for input_grid, output_grid in train_pairs:
            transformations = self.reasoner.analyze_transformation(input_grid, output_grid)
            all_transformations.extend(transformations)
        
        # Find consistent transformations across examples
        transformation_counts = defaultdict(int)
        transformation_examples = defaultdict(list)
        
        for trans in all_transformations:
            # Create a key for grouping similar transformations
            key = (trans.operation, str(trans.obj_selector), str(trans.parameters))
            transformation_counts[key] += 1
            transformation_examples[key].append(trans)
        
        # Select transformations that appear in multiple examples
        consistent_transformations = []
        for key, count in transformation_counts.items():
            if count >= len(train_pairs) * 0.5:  # Appears in at least half the examples
                # Use the first example of this transformation type
                consistent_transformations.append(transformation_examples[key][0])
        
        return consistent_transformations
    
    def test_hypothesis(self, transformation: ObjectTransformation, train_pairs: List[Tuple[Array, Array]]) -> float:
        """Test how well a transformation hypothesis works on training data."""
        correct = 0
        total = len(train_pairs)
        
        for input_grid, expected_output in train_pairs:
            try:
                predicted_output = self.reasoner.apply_transformation(input_grid, transformation)
                if np.array_equal(predicted_output, expected_output):
                    correct += 1
            except Exception:
                continue
        
        return correct / total if total > 0 else 0.0
