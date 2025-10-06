#!/usr/bin/env python3
"""Object Identity Tracking and Evolution Analysis

Core insight: Every object in input has a corresponding object in output (or doesn't exist).
By tracking what happened to each object, we can find the transformation rules.

Process:
1. Extract ALL objects from each input example -> give them identity hashes
2. Extract ALL objects from each output example -> give them identity hashes
3. Match objects across input->output (same object that changed properties)
4. Track evolution: what changed? (color, position, size, shape)
5. Find patterns across examples: what's common? what varies?
6. Generate pliance rules anchored to these object transformations
7. Test rules, track failures, refine
"""

from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
from numpy.typing import NDArray as Array
import hashlib
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ObjectIdentity:
    """A unique object with properties and identity hash."""
    id: str  # Hash-based unique ID
    color: int
    pixels: List[Tuple[int, int]]  # List of (row, col) coordinates
    bbox: Tuple[int, int, int, int]  # (r, c, h, w)
    size: int
    shape_type: str  # 'single', 'rectangle', 'line', 'complex', etc.
    centroid: Tuple[float, float]

    # Relational properties (filled during analysis)
    relationships: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def compute_properties_hash(self) -> str:
        """Hash based on spatial structure (position-invariant)."""
        # Normalize to top-left origin
        if not self.pixels:
            return "empty"

        min_r = min(p[0] for p in self.pixels)
        min_c = min(p[1] for p in self.pixels)
        normalized = tuple(sorted((r - min_r, c - min_c) for r, c in self.pixels))

        # Hash: normalized_shape + color
        content = f"{normalized}_{self.color}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class ObjectEvolution:
    """Tracks how an object changed from input to output."""
    input_obj: ObjectIdentity
    output_obj: Optional[ObjectIdentity]

    # What changed?
    changes: Dict[str, Any] = field(default_factory=dict)

    def analyze_changes(self):
        """Determine what properties changed."""
        if self.output_obj is None:
            self.changes['deleted'] = True
            return

        inp, out = self.input_obj, self.output_obj

        # Color change?
        if inp.color != out.color:
            self.changes['color'] = {'from': inp.color, 'to': out.color}

        # Position change?
        if inp.centroid != out.centroid:
            dr = out.centroid[0] - inp.centroid[0]
            dc = out.centroid[1] - inp.centroid[1]
            self.changes['position'] = {'delta_r': dr, 'delta_c': dc}

        # Size change?
        if inp.size != out.size:
            self.changes['size'] = {'from': inp.size, 'to': out.size, 'ratio': out.size / inp.size}

        # Shape change?
        if inp.bbox[2:] != out.bbox[2:]:
            self.changes['dimensions'] = {
                'from': inp.bbox[2:],
                'to': out.bbox[2:],
                'height_change': out.bbox[2] - inp.bbox[2],
                'width_change': out.bbox[3] - inp.bbox[3]
            }

        # Pixel-level transformation?
        if inp.pixels != out.pixels:
            self.changes['pixel_structure'] = 'modified'


class ObjectIdentityTracker:
    """Tracks objects across input/output pairs and finds evolution patterns."""

    def __init__(self):
        self.input_objects: List[List[ObjectIdentity]] = []  # Per example
        self.output_objects: List[List[ObjectIdentity]] = []  # Per example
        self.evolutions: List[List[ObjectEvolution]] = []  # Per example
        self.patterns: Dict[str, Any] = {}

    def extract_objects(self, grid: Array, background: int = 0) -> List[ObjectIdentity]:
        """Extract all objects from a grid."""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if visited[r, c] or grid[r, c] == background:
                    continue

                # Flood fill to get connected component
                color = int(grid[r, c])
                pixels = self._flood_fill(grid, r, c, color, visited)

                if pixels:
                    obj = self._create_object_identity(pixels, color)
                    objects.append(obj)

        return objects

    def _flood_fill(self, grid: Array, r: int, c: int, color: int, visited: Array) -> List[Tuple[int, int]]:
        """Get all connected pixels of the same color."""
        pixels = []
        stack = [(r, c)]

        while stack:
            cr, cc = stack.pop()
            if cr < 0 or cr >= grid.shape[0] or cc < 0 or cc >= grid.shape[1]:
                continue
            if visited[cr, cc] or grid[cr, cc] != color:
                continue

            visited[cr, cc] = True
            pixels.append((cr, cc))

            # 4-connectivity
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((cr + dr, cc + dc))

        return pixels

    def _create_object_identity(self, pixels: List[Tuple[int, int]], color: int) -> ObjectIdentity:
        """Create an object with computed properties."""
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]

        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)

        bbox = (r_min, c_min, r_max - r_min + 1, c_max - c_min + 1)
        centroid = (sum(rows) / len(rows), sum(cols) / len(cols))

        # Determine shape type
        if len(pixels) == 1:
            shape_type = 'single'
        elif bbox[2] * bbox[3] == len(pixels):
            shape_type = 'rectangle'
        elif bbox[2] == 1 or bbox[3] == 1:
            shape_type = 'line'
        else:
            shape_type = 'complex'

        obj = ObjectIdentity(
            id="",  # Will be set
            color=color,
            pixels=sorted(pixels),
            bbox=bbox,
            size=len(pixels),
            shape_type=shape_type,
            centroid=centroid
        )

        obj.id = obj.compute_properties_hash()
        return obj

    def match_objects(self, input_objs: List[ObjectIdentity], output_objs: List[ObjectIdentity]) -> List[ObjectEvolution]:
        """Match input objects to output objects and track evolution."""
        evolutions = []
        matched_outputs = set()

        for inp_obj in input_objs:
            # Try to find best match in output
            best_match = None
            best_score = -1

            for out_obj in output_objs:
                if out_obj in matched_outputs:
                    continue

                score = self._similarity_score(inp_obj, out_obj)
                if score > best_score:
                    best_score = score
                    best_match = out_obj

            # Match if score is high enough
            if best_score > 0.3:  # Threshold for matching
                matched_outputs.add(best_match)
                evolution = ObjectEvolution(inp_obj, best_match)
            else:
                evolution = ObjectEvolution(inp_obj, None)  # Deleted

            evolution.analyze_changes()
            evolutions.append(evolution)

        # Check for new objects in output
        for out_obj in output_objs:
            if out_obj not in matched_outputs:
                # New object created
                evolution = ObjectEvolution(
                    input_obj=ObjectIdentity(id="new", color=0, pixels=[], bbox=(0,0,0,0), size=0, shape_type="none", centroid=(0,0)),
                    output_obj=out_obj
                )
                evolution.changes['created'] = True
                evolutions.append(evolution)

        return evolutions

    def _similarity_score(self, obj1: ObjectIdentity, obj2: ObjectIdentity) -> float:
        """Compute similarity between two objects (0-1)."""
        score = 0.0

        # Same color? (strong signal)
        if obj1.color == obj2.color:
            score += 0.4

        # Similar size?
        if obj1.size > 0 and obj2.size > 0:
            size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
            score += 0.2 * size_ratio

        # Similar position?
        dist = np.sqrt((obj1.centroid[0] - obj2.centroid[0])**2 +
                      (obj1.centroid[1] - obj2.centroid[1])**2)
        if dist < 5:  # Close positions
            score += 0.2 * (1 - dist / 5)

        # Same shape type?
        if obj1.shape_type == obj2.shape_type:
            score += 0.2

        return score

    def analyze_training_pairs(self, train_pairs: List[Tuple[Array, Array]]):
        """Analyze all training examples to find patterns."""
        print(f"\n=== OBJECT IDENTITY TRACKING ===")
        print(f"Analyzing {len(train_pairs)} training examples...\n")

        for i, (inp, out) in enumerate(train_pairs):
            print(f"Example {i}:")

            # Extract objects
            inp_objs = self.extract_objects(inp)
            out_objs = self.extract_objects(out)

            print(f"  Input: {len(inp_objs)} objects")
            print(f"  Output: {len(out_objs)} objects")

            # Track evolution
            evolutions = self.match_objects(inp_objs, out_objs)

            self.input_objects.append(inp_objs)
            self.output_objects.append(out_objs)
            self.evolutions.append(evolutions)

            # Show evolutions
            for evo in evolutions:
                if evo.changes:
                    changes_str = ", ".join(f"{k}:{v}" for k, v in list(evo.changes.items())[:2])
                    print(f"    Object {evo.input_obj.id[:6]}... -> {changes_str}")
            print()

        # Find common patterns
        self._find_common_patterns()

    def _find_common_patterns(self):
        """Find what's common across all examples."""
        print("=== CROSS-EXAMPLE PATTERN ANALYSIS ===\n")

        # Collect all change types
        change_types = defaultdict(int)
        change_values = defaultdict(list)

        for example_evos in self.evolutions:
            for evo in example_evos:
                for change_type, change_val in evo.changes.items():
                    change_types[change_type] += 1
                    change_values[change_type].append(change_val)

        total_examples = len(self.evolutions)

        print("Common transformations:")
        for change_type, count in sorted(change_types.items(), key=lambda x: -x[1]):
            frequency = count / total_examples
            print(f"  {change_type}: {count}/{total_examples} examples ({frequency:.1%})")

            # Show pattern details
            values = change_values[change_type]
            if change_type == 'color' and len(values) > 1:
                # Check if color changes are consistent
                from_colors = [v.get('from') for v in values if isinstance(v, dict)]
                to_colors = [v.get('to') for v in values if isinstance(v, dict)]
                if from_colors and to_colors:
                    print(f"    Color mappings: {set(from_colors)} -> {set(to_colors)}")

            elif change_type == 'position' and len(values) > 1:
                # Check if movements are consistent
                deltas = [(v.get('delta_r', 0), v.get('delta_c', 0)) for v in values if isinstance(v, dict)]
                unique_deltas = set(deltas)
                if len(unique_deltas) == 1:
                    print(f"    Consistent movement: {unique_deltas.pop()}")
                else:
                    print(f"    Variable movements: {len(unique_deltas)} different patterns")

        self.patterns = {
            'change_types': dict(change_types),
            'change_values': dict(change_values),
            'frequency': {k: v/total_examples for k, v in change_types.items()}
        }

        print()
        return self.patterns

    def generate_pliance_rules(self) -> List[Dict[str, Any]]:
        """Generate pliance rules from observed patterns."""
        rules = []

        for change_type, frequency in self.patterns.get('frequency', {}).items():
            if frequency >= 0.5:  # Present in at least half the examples
                rule = {
                    'type': 'object_transformation',
                    'change': change_type,
                    'frequency': frequency,
                    'confidence': frequency,
                    'pattern_data': self.patterns['change_values'].get(change_type, [])
                }
                rules.append(rule)

        return rules


def demonstrate_object_tracking():
    """Demo showing object tracking on a simple example."""
    # Create simple example: red square becomes blue square
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    output_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
    ])

    tracker = ObjectIdentityTracker()
    tracker.analyze_training_pairs([(input_grid, output_grid)])

    rules = tracker.generate_pliance_rules()
    print(f"Generated {len(rules)} pliance rules")
    for rule in rules:
        print(f"  Rule: {rule['type']} - {rule['change']} (confidence: {rule['confidence']:.1%})")


if __name__ == "__main__":
    demonstrate_object_tracking()
