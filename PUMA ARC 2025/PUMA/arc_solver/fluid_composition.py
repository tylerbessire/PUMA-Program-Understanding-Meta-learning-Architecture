#!/usr/bin/env python3
"""Fluid Intelligence: Adaptive Rule Composition with Per-Example Variation

This module enables the system to create rules by chaining primitives together
where parameters can vary per example based on relational/adaptive anchors.

Key concepts:
1. Primitives can be composed into multi-step transformations
2. Parameters can be adaptive (computed per-input) rather than fixed
3. Rules are valid if they work per-example even with varying parameters
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
from numpy.typing import NDArray as Array


class AdaptiveParameter:
    """A parameter that computes its value per-example based on input properties."""

    def __init__(self, extractor: Callable[[Array], Any], description: str):
        """
        Args:
            extractor: Function that takes an input grid and returns the parameter value
            description: Human-readable description of what this parameter represents
        """
        self.extractor = extractor
        self.description = description

    def get_value(self, grid: Array) -> Any:
        """Compute the parameter value for a specific input."""
        return self.extractor(grid)


class FluidPrimitive:
    """A primitive operation with potentially adaptive parameters."""

    def __init__(
        self,
        name: str,
        operation: Callable,
        parameters: Dict[str, Any]
    ):
        """
        Args:
            name: Name of the primitive operation
            operation: The actual function to execute
            parameters: Dict of parameter_name -> value or AdaptiveParameter
        """
        self.name = name
        self.operation = operation
        self.parameters = parameters

    def execute(self, grid: Array) -> Array:
        """Execute this primitive on the given grid."""
        # Resolve adaptive parameters
        resolved_params = {}
        for param_name, param_value in self.parameters.items():
            if isinstance(param_value, AdaptiveParameter):
                resolved_params[param_name] = param_value.get_value(grid)
            else:
                resolved_params[param_name] = param_value

        # Execute with resolved parameters
        return self.operation(grid, **resolved_params)


class FluidProgram:
    """A composition of primitives that can adapt per-example."""

    def __init__(self, primitives: List[FluidPrimitive], description: str = ""):
        """
        Args:
            primitives: Ordered list of primitives to execute in sequence
            description: Human-readable description of what this program does
        """
        self.primitives = primitives
        self.description = description

    def __call__(self, grid: Array) -> Array:
        """Execute the program on an input grid."""
        result = grid.copy()
        for primitive in self.primitives:
            result = primitive.execute(result)
        return result

    def validate(
        self,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Tuple[float, int, int, List[Dict[str, Any]]]:
        """
        Validate this program against training examples.

        Returns:
            (accuracy, correct_count, total_count, per_example_params)

        The per_example_params shows what parameters were used for each example,
        demonstrating the fluid adaptation.
        """
        correct = 0
        total = len(train_pairs)
        per_example_params = []

        for input_grid, expected_output in train_pairs:
            try:
                # Track what parameters were used
                example_params = {}
                actual_output = input_grid.copy()

                for primitive in self.primitives:
                    # Resolve parameters for this example
                    resolved = {}
                    for param_name, param_value in primitive.parameters.items():
                        if isinstance(param_value, AdaptiveParameter):
                            resolved[param_name] = param_value.get_value(actual_output)
                        else:
                            resolved[param_name] = param_value

                    example_params[primitive.name] = resolved
                    actual_output = primitive.operation(actual_output, **resolved)

                per_example_params.append(example_params)

                if np.array_equal(actual_output, expected_output):
                    correct += 1

            except Exception as e:
                per_example_params.append({'error': str(e)})

        accuracy = correct / total if total > 0 else 0.0
        return accuracy, correct, total, per_example_params


class FluidComposer:
    """Composes primitives into adaptive programs based on learned patterns."""

    @staticmethod
    def create_color_marker_extraction_program(
        marker_color: int,
        colors_to_remove: Optional[List[int]] = None
    ) -> FluidProgram:
        """
        Create a program that:
        1. Finds the bbox of a marker color (adaptive per-example)
        2. Extracts that region
        3. Optionally removes specified colors

        This handles the common ARC pattern of "color X marks where to extract"
        """
        from .measurement_primitives import MeasurementPrimitives

        # Adaptive parameter: bbox of marker color (varies per example)
        def get_marker_bbox(grid: Array) -> Tuple[int, int, int, int]:
            objects = MeasurementPrimitives.find_objects_by_color(grid, background=-1)
            if marker_color in objects:
                bbox = MeasurementPrimitives.find_bounding_box(objects[marker_color])
                if bbox:
                    return bbox
            # Fallback: content bbox
            return MeasurementPrimitives.find_content_bounding_box(grid)

        bbox_param = AdaptiveParameter(
            extractor=get_marker_bbox,
            description=f"Bounding box of color {marker_color} objects"
        )

        # Primitive 1: Extract region at marker bbox
        def extract_bbox(grid: Array, bbox: Tuple[int, int, int, int]) -> Array:
            r, c, h, w = bbox
            return grid[r:r+h, c:c+w].copy()

        extract_primitive = FluidPrimitive(
            name="extract_marker_region",
            operation=extract_bbox,
            parameters={'bbox': bbox_param}
        )

        primitives = [extract_primitive]

        # Primitive 2 (optional): Remove colors
        if colors_to_remove:
            def remove_colors(grid: Array, colors: List[int]) -> Array:
                result = grid.copy()
                background = MeasurementPrimitives.find_background_color(grid)
                for color in colors:
                    result[result == color] = background
                return result

            remove_primitive = FluidPrimitive(
                name="remove_colors",
                operation=remove_colors,
                parameters={'colors': colors_to_remove}
            )
            primitives.append(remove_primitive)

        description = f"Extract region marked by color {marker_color}"
        if colors_to_remove:
            description += f", then remove colors {colors_to_remove}"

        return FluidProgram(primitives, description)


def demonstrate_fluid_intelligence():
    """Example showing how fluid programs adapt per-example."""
    print("=== Fluid Intelligence Example ===\n")

    # Create a program that extracts color-8 regions (varying sizes per example)
    program = FluidComposer.create_color_marker_extraction_program(
        marker_color=8,
        colors_to_remove=[8, 5, 7]
    )

    print(f"Program: {program.description}\n")
    print("This program will:")
    print("  1. Find color-8 bbox (ADAPTIVE - different per example)")
    print("  2. Extract that region (size varies based on step 1)")
    print("  3. Remove colors [8, 5, 7] (FIXED parameter)")
    print()
    print("Key insight: The bbox parameter is FLUID - it adapts to each")
    print("input, allowing the same rule to work on varying-sized regions!")


if __name__ == "__main__":
    demonstrate_fluid_intelligence()
