#!/usr/bin/env python3
"""Pattern-to-Program Compiler

Converts high-level learned patterns into executable program sequences.
Bridges the gap between pattern detection and program synthesis.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from numpy.typing import NDArray as Array
import re
from .measurement_primitives import MeasurementPrimitives


class PatternCompiler:
    """Compiles learned patterns into executable transformation programs."""

    def __init__(self):
        """Initialize the pattern compiler."""
        self.primitives = self._initialize_primitives()

    def _initialize_primitives(self) -> Dict[str, callable]:
        """Initialize available primitive operations."""
        return {
            'extract_region': self._extract_region,
            'resize': self._resize_grid,
            'remove_colors': self._remove_colors,
            'recolor': self._recolor,
            'crop': self._crop_to_content,
            'fill_background': self._fill_background,
        }

    def compile_from_pattern(
        self,
        pattern: Dict[str, Any],
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """
        Compile a learned pattern into an executable program.

        Args:
            pattern: Pattern dict with 'type' and 'description'
            train_pairs: Training examples to extract parameters from

        Returns:
            Executable function that transforms input grid to output grid
        """
        pattern_type = pattern.get('type', '')
        description = pattern.get('description', '')

        # FIRST: Try adaptive relational pattern (RFT-based)
        if pattern_type in ['extraction', 'size_reduction', 'resized', 'transformed']:
            adaptive = self._compile_adaptive_extraction(train_pairs)
            if adaptive:
                print(f"DEBUG PatternCompiler: Compiled adaptive {pattern_type} pattern!")
                return adaptive

        # SECOND: Try to compile composite pattern from description
        if description and '|' in description:
            try:
                composite = self._compile_composite_from_description(description, train_pairs)
                if composite:
                    return composite
            except Exception as e:
                print(f"DEBUG PatternCompiler: Failed to compile composite: {e}")

        # FALLBACK: Parse pattern and compile to program
        if pattern_type == 'size_reduction' or 'reduce' in description.lower():
            return self._compile_size_reduction(description, train_pairs)

        elif pattern_type == 'extraction' or 'extract' in description.lower():
            return self._compile_extraction(description, train_pairs)

        elif pattern_type == 'color_removal' or 'remove color' in description.lower():
            return self._compile_color_removal(description, train_pairs)

        elif pattern_type == 'recoloring':
            return self._compile_recoloring(description, train_pairs)

        return None

    def _compile_adaptive_extraction(
        self,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """
        Compile an adaptive extraction pattern using relational reasoning.

        This implements RFT principles: derive output size from input properties
        rather than using static values.
        """
        # Derive the relational rule from training data
        rule = MeasurementPrimitives.derive_extraction_rule(train_pairs)

        if not rule:
            return None

        print(f"DEBUG PatternCompiler: Found adaptive rule - {rule['description']}")

        # Create adaptive program that applies the rule
        def adaptive_extraction(grid: Array) -> Array:
            result = MeasurementPrimitives.apply_extraction_rule(grid, rule)
            if result is not None:
                return result
            # Fallback: return cropped content
            bbox = MeasurementPrimitives.find_content_bounding_box(grid)
            r, c, h, w = bbox
            return grid[r:r+h, c:c+w].copy()

        return adaptive_extraction

    def _compile_composite_from_description(
        self,
        description: str,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """
        Compile a composite pattern from the correct_approach description.

        Example: "Transform: Reduce from (30, 30) to (9, 3) | Extract region of size (9, 3)"
        """
        # Split by pipe to get individual operations
        operations = [op.strip() for op in description.split('|')]

        compiled_ops = []

        for op_desc in operations:
            # Extract region
            if 'extract region of size' in op_desc.lower():
                match = re.search(r'size \((\d+),\s*(\d+)\)', op_desc)
                if match:
                    target_h, target_w = int(match.group(1)), int(match.group(2))

                    def extract_op(grid: Array, th=target_h, tw=target_w) -> Array:
                        background = self._find_background_color(grid)
                        # Find non-background content
                        for y in range(grid.shape[0] - th + 1):
                            for x in range(grid.shape[1] - tw + 1):
                                region = grid[y:y+th, x:x+tw]
                                if np.sum(region != background) > (th * tw * 0.1):
                                    return region
                        # Fallback: extract from top-left
                        return grid[:th, :tw] if grid.shape[0] >= th and grid.shape[1] >= tw else grid

                    compiled_ops.append(extract_op)

            # Reduce/resize
            elif 'reduce from' in op_desc.lower():
                match = re.search(r'from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)', op_desc)
                if match:
                    from_h, from_w = int(match.group(1)), int(match.group(2))
                    to_h, to_w = int(match.group(3)), int(match.group(4))

                    def resize_op(grid: Array, th=to_h, tw=to_w) -> Array:
                        # Crop to content then resize
                        content = self._crop_to_content(grid)
                        if content.shape == (th, tw):
                            return content
                        return self._resize_grid(content, (th, tw))

                    compiled_ops.append(resize_op)

            # Remove colors
            elif 'remove colors' in op_desc.lower():
                match = re.findall(r'\d+', op_desc)
                if match:
                    colors_to_remove = [int(c) for c in match]

                    def remove_colors_op(grid: Array, colors=colors_to_remove) -> Array:
                        result = grid.copy()
                        background = self._find_background_color(grid)
                        for color in colors:
                            result[result == color] = background
                        return result

                    compiled_ops.append(remove_colors_op)

        if not compiled_ops:
            print(f"DEBUG PatternCompiler: No operations compiled from description: {description[:100]}")
            return None

        print(f"DEBUG PatternCompiler: Successfully compiled {len(compiled_ops)} operations")

        # Create composite program
        def composite_program(grid: Array) -> Array:
            result = grid.copy()
            for op in compiled_ops:
                result = op(result)
            return result

        return composite_program

    def _compile_size_reduction(
        self,
        description: str,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """Compile size reduction pattern into program."""
        # Parse dimensions from description (e.g., "Reduce from (30, 30) to (9, 3)")
        match = re.search(r'from \((\d+),\s*(\d+)\) to \((\d+),\s*(\d+)\)', description)

        if not match:
            # Infer from training data
            if train_pairs:
                input_grid, output_grid = train_pairs[0]
                from_h, from_w = input_grid.shape
                to_h, to_w = output_grid.shape
            else:
                return None
        else:
            from_h, from_w = int(match.group(1)), int(match.group(2))
            to_h, to_w = int(match.group(3)), int(match.group(4))

        # Create program that extracts and resizes
        def size_reduction_program(grid: Array) -> Array:
            # Extract content region (skip background)
            content_grid = self._crop_to_content(grid)

            # If target size matches content, return it
            if content_grid.shape == (to_h, to_w):
                return content_grid

            # Otherwise resize/subsample
            return self._resize_grid(content_grid, (to_h, to_w))

        return size_reduction_program

    def _compile_extraction(
        self,
        description: str,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """Compile extraction pattern into program."""
        # Parse extraction parameters
        match = re.search(r'extract.*?(\d+).*?(\d+)', description.lower())

        if not match and train_pairs:
            # Infer extraction region from training data
            input_grid, output_grid = train_pairs[0]
            target_h, target_w = output_grid.shape

            def extraction_program(grid: Array) -> Array:
                # Find non-background content
                background = self._find_background_color(grid)

                # Extract regions that match output size
                for y in range(grid.shape[0] - target_h + 1):
                    for x in range(grid.shape[1] - target_w + 1):
                        region = grid[y:y+target_h, x:x+target_w]

                        # Check if region has meaningful content
                        if np.sum(region != background) > (target_h * target_w * 0.1):
                            return region

                # Fallback: extract from top-left
                return grid[:target_h, :target_w]

            return extraction_program

        return None

    def _compile_color_removal(
        self,
        description: str,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """Compile color removal pattern into program."""
        # Parse colors to remove
        colors_match = re.findall(r'\d+', description)

        if colors_match:
            colors_to_remove = [int(c) for c in colors_match]
        elif train_pairs:
            # Infer from training data
            input_grid, output_grid = train_pairs[0]
            input_colors = set(input_grid.flatten())
            output_colors = set(output_grid.flatten())
            colors_to_remove = list(input_colors - output_colors)
        else:
            return None

        def color_removal_program(grid: Array) -> Array:
            result = grid.copy()
            background = self._find_background_color(grid)

            for color in colors_to_remove:
                result[result == color] = background

            return result

        return color_removal_program

    def _compile_recoloring(
        self,
        description: str,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """Compile recoloring pattern into program."""
        if not train_pairs:
            return None

        # Build color mapping from training data
        color_map = {}
        for input_grid, output_grid in train_pairs:
            for in_color in np.unique(input_grid):
                out_colors = output_grid[input_grid == in_color]
                if len(out_colors) > 0:
                    most_common = np.bincount(out_colors).argmax()
                    color_map[int(in_color)] = int(most_common)

        def recoloring_program(grid: Array) -> Array:
            result = grid.copy()
            for in_color, out_color in color_map.items():
                result[grid == in_color] = out_color
            return result

        return recoloring_program

    def compile_composite_pattern(
        self,
        patterns: List[Dict[str, Any]],
        train_pairs: List[Tuple[Array, Array]]
    ) -> Optional[callable]:
        """
        Compile multiple patterns into a composite program.
        Chains operations together in sequence.
        """
        programs = []

        for pattern in patterns:
            program = self.compile_from_pattern(pattern, train_pairs)
            if program:
                programs.append(program)

        if not programs:
            return None

        def composite_program(grid: Array) -> Array:
            result = grid.copy()
            for program in programs:
                result = program(result)
            return result

        return composite_program

    # Primitive operations

    def _extract_region(self, grid: Array, y: int, x: int, h: int, w: int) -> Array:
        """Extract rectangular region from grid."""
        return grid[y:y+h, x:x+w].copy()

    def _resize_grid(self, grid: Array, target_shape: Tuple[int, int]) -> Array:
        """Resize grid to target shape via subsampling."""
        h, w = grid.shape
        target_h, target_w = target_shape

        # Calculate sampling intervals
        step_h = h / target_h
        step_w = w / target_w

        result = np.zeros(target_shape, dtype=grid.dtype)
        for i in range(target_h):
            for j in range(target_w):
                y = int(i * step_h)
                x = int(j * step_w)
                result[i, j] = grid[y, x]

        return result

    def _remove_colors(self, grid: Array, colors: List[int], replacement: int = 0) -> Array:
        """Remove specified colors by replacing with background."""
        result = grid.copy()
        for color in colors:
            result[result == color] = replacement
        return result

    def _recolor(self, grid: Array, from_color: int, to_color: int) -> Array:
        """Change all instances of one color to another."""
        result = grid.copy()
        result[result == from_color] = to_color
        return result

    def _crop_to_content(self, grid: Array) -> Array:
        """Crop grid to non-background content."""
        background = self._find_background_color(grid)

        # Find bounding box of non-background pixels
        non_bg = np.argwhere(grid != background)

        if len(non_bg) == 0:
            return grid

        y_min, x_min = non_bg.min(axis=0)
        y_max, x_max = non_bg.max(axis=0)

        return grid[y_min:y_max+1, x_min:x_max+1].copy()

    def _fill_background(self, grid: Array, color: int) -> Array:
        """Fill background with specified color."""
        result = grid.copy()
        background = self._find_background_color(grid)
        result[result == background] = color
        return result

    def _find_background_color(self, grid: Array) -> int:
        """Find most common color (assumed to be background)."""
        colors, counts = np.unique(grid, return_counts=True)
        return int(colors[counts.argmax()])

    def validate_program(
        self,
        program: callable,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Tuple[float, int, int]:
        """
        Validate a compiled program against training examples.

        Returns:
            (accuracy, correct_count, total_count)
        """
        correct = 0
        total = len(train_pairs)

        for input_grid, expected_output in train_pairs:
            try:
                actual_output = program(input_grid)

                if np.array_equal(actual_output, expected_output):
                    correct += 1
            except Exception:
                # Program failed on this example
                pass

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total
