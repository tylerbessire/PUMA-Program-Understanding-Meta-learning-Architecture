#!/usr/bin/env python3
"""Measurement Primitives for Adaptive Pattern Learning

These primitives extract properties and measurements from grids to enable
relational reasoning, following RFT principles of deriving output properties
from input properties.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray as Array


class MeasurementPrimitives:
    """Collection of measurement primitives for adaptive patterns."""

    @staticmethod
    def find_background_color(grid: Array) -> int:
        """Find the most common color (assumed to be background)."""
        colors, counts = np.unique(grid, return_counts=True)
        return int(colors[counts.argmax()])

    @staticmethod
    def find_objects_by_color(grid: Array, background: Optional[int] = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        Find all objects grouped by color.

        Returns:
            Dict mapping color -> list of (row, col) coordinates
        """
        if background is None:
            background = MeasurementPrimitives.find_background_color(grid)

        objects_by_color = {}
        for color in np.unique(grid):
            if color != background:
                coords = list(zip(*np.where(grid == color)))
                objects_by_color[int(color)] = coords

        return objects_by_color

    @staticmethod
    def find_bounding_box(coords: List[Tuple[int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Find bounding box of a set of coordinates.

        Returns:
            (min_row, min_col, height, width) or None if empty
        """
        if not coords:
            return None

        rows = [r for r, c in coords]
        cols = [c for r, c in coords]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        height = max_row - min_row + 1
        width = max_col - min_col + 1

        return (min_row, min_col, height, width)

    @staticmethod
    def find_content_bounding_box(grid: Array) -> Tuple[int, int, int, int]:
        """Find bounding box of all non-background content."""
        background = MeasurementPrimitives.find_background_color(grid)
        non_bg_coords = list(zip(*np.where(grid != background)))

        if not non_bg_coords:
            return (0, 0, grid.shape[0], grid.shape[1])

        return MeasurementPrimitives.find_bounding_box(non_bg_coords)

    @staticmethod
    def measure_largest_object(grid: Array) -> Dict[str, Any]:
        """
        Find and measure the largest object.

        Returns:
            Dict with 'color', 'bbox', 'height', 'width', 'size'
        """
        objects = MeasurementPrimitives.find_objects_by_color(grid)

        if not objects:
            return {'color': 0, 'bbox': None, 'height': 0, 'width': 0, 'size': 0}

        # Find largest by number of pixels
        largest_color = max(objects.keys(), key=lambda c: len(objects[c]))
        coords = objects[largest_color]
        bbox = MeasurementPrimitives.find_bounding_box(coords)

        return {
            'color': largest_color,
            'coords': coords,
            'bbox': bbox,
            'height': bbox[2] if bbox else 0,
            'width': bbox[3] if bbox else 0,
            'size': len(coords)
        }

    @staticmethod
    def find_region_with_marker(grid: Array, marker_color: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Find a rectangular region marked by a specific color.

        This detects rectangular boundaries made of marker_color pixels.
        """
        marker_coords = list(zip(*np.where(grid == marker_color)))

        if not marker_coords:
            return None

        # Check if markers form a rectangular frame
        rows = sorted(set(r for r, c in marker_coords))
        cols = sorted(set(c for r, c in marker_coords))

        if len(rows) < 2 or len(cols) < 2:
            return None

        # Check for rectangular pattern (markers on edges)
        min_row, max_row = rows[0], rows[-1]
        min_col, max_col = cols[0], cols[-1]

        # The region inside the markers
        inner_min_row = min_row + 1
        inner_min_col = min_col + 1
        inner_height = max_row - min_row - 1
        inner_width = max_col - min_col - 1

        if inner_height > 0 and inner_width > 0:
            return (inner_min_row, inner_min_col, inner_height, inner_width)

        return None

    @staticmethod
    def derive_extraction_rule(
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[Dict[str, Any]]:
        """
        Derive an adaptive extraction rule from training examples.

        This implements RFT-based relational reasoning to find how output
        dimensions relate to input properties.

        Returns:
            Dict describing the relational rule, or None
        """
        if not train_pairs:
            return None

        # Hypothesis 1: Output size matches bounding box of largest colored region
        hypothesis_1_valid = True
        for inp, out in train_pairs:
            largest = MeasurementPrimitives.measure_largest_object(inp)
            if (largest['height'], largest['width']) != out.shape:
                hypothesis_1_valid = False
                break

        if hypothesis_1_valid:
            return {
                'type': 'extract_largest_object_bbox',
                'description': 'Output dimensions match the bounding box of the largest object'
            }

        # Hypothesis 2: Output size matches content bounding box
        hypothesis_2_valid = True
        for inp, out in train_pairs:
            bbox = MeasurementPrimitives.find_content_bounding_box(inp)
            if (bbox[2], bbox[3]) != out.shape:
                hypothesis_2_valid = False
                break

        if hypothesis_2_valid:
            return {
                'type': 'extract_content_bbox',
                'description': 'Output dimensions match the bounding box of all content'
            }

        # Hypothesis 3: Output is region marked by a specific color
        # Try each color as potential marker
        background = MeasurementPrimitives.find_background_color(train_pairs[0][0])
        colors = set()
        for inp, _ in train_pairs:
            colors.update(np.unique(inp))
        colors.discard(background)

        for marker_color in colors:
            hypothesis_3_valid = True
            for inp, out in train_pairs:
                region = MeasurementPrimitives.find_region_with_marker(inp, int(marker_color))
                if region is None or (region[2], region[3]) != out.shape:
                    hypothesis_3_valid = False
                    break

            if hypothesis_3_valid:
                return {
                    'type': 'extract_marked_region',
                    'marker_color': int(marker_color),
                    'description': f'Output dimensions match region marked by color {marker_color}'
                }

        # Hypothesis 4: Output size is derived from specific object colors
        # Group by color and check if any color's bbox consistently matches output
        for inp, out in train_pairs[:1]:  # Check first example
            # Don't auto-detect background - check ALL colors including potential markers
            objects = MeasurementPrimitives.find_objects_by_color(inp, background=-1)

            for color, coords in objects.items():
                bbox = MeasurementPrimitives.find_bounding_box(coords)
                if bbox and (bbox[2], bbox[3]) == out.shape:
                    # Test if this holds for all examples
                    hypothesis_4_valid = True
                    for inp2, out2 in train_pairs:
                        objects2 = MeasurementPrimitives.find_objects_by_color(inp2, background=-1)
                        if color not in objects2:
                            hypothesis_4_valid = False
                            break
                        bbox2 = MeasurementPrimitives.find_bounding_box(objects2[color])
                        if not bbox2 or (bbox2[2], bbox2[3]) != out2.shape:
                            hypothesis_4_valid = False
                            break

                    if hypothesis_4_valid:
                        return {
                            'type': 'extract_color_bbox',
                            'color': int(color),
                            'description': f'Output dimensions match bounding box of color {color} objects'
                        }

        return None

    @staticmethod
    def apply_extraction_rule(
        grid: Array,
        rule: Dict[str, Any]
    ) -> Optional[Array]:
        """
        Apply a derived extraction rule to extract a region from the grid.

        Args:
            grid: Input grid
            rule: Extraction rule from derive_extraction_rule()

        Returns:
            Extracted region, or None
        """
        rule_type = rule.get('type')

        if rule_type == 'extract_largest_object_bbox':
            largest = MeasurementPrimitives.measure_largest_object(grid)
            bbox = largest['bbox']
            if bbox:
                r, c, h, w = bbox
                return grid[r:r+h, c:c+w].copy()

        elif rule_type == 'extract_content_bbox':
            bbox = MeasurementPrimitives.find_content_bounding_box(grid)
            r, c, h, w = bbox
            return grid[r:r+h, c:c+w].copy()

        elif rule_type == 'extract_marked_region':
            marker_color = rule['marker_color']
            region = MeasurementPrimitives.find_region_with_marker(grid, marker_color)
            if region:
                r, c, h, w = region
                return grid[r:r+h, c:c+w].copy()

        elif rule_type == 'extract_color_bbox':
            color = rule['color']
            # Don't auto-detect background - we specifically need the marker color
            objects = MeasurementPrimitives.find_objects_by_color(grid, background=-1)
            if color in objects:
                bbox = MeasurementPrimitives.find_bounding_box(objects[color])
                if bbox:
                    r, c, h, w = bbox
                    # Extract the region containing the marker color
                    region = grid[r:r+h, c:c+w].copy()

                    # Check if marker forms edges (hollow frame pattern vs filled placeholder)
                    if region.shape[0] > 2 and region.shape[1] > 2:
                        top_is_marker = np.all(region[0, :] == color)
                        bottom_is_marker = np.all(region[-1, :] == color)
                        left_is_marker = np.all(region[:, 0] == color)
                        right_is_marker = np.all(region[:, -1] == color)

                        if top_is_marker and bottom_is_marker and left_is_marker and right_is_marker:
                            # Check if it's a hollow frame (interior has other colors) or filled placeholder
                            interior = region[1:-1, 1:-1]
                            interior_has_marker = np.any(interior == color)

                            if not interior_has_marker:
                                # Hollow frame - extract interior only (non-marker content)
                                return interior.copy()
                            else:
                                # Filled placeholder - remove marker color but preserve dimensions
                                result = region.copy()
                                result[result == color] = 0  # Replace marker with background
                                return result

                    # For other patterns: return the bbox itself
                    return region

        return None
