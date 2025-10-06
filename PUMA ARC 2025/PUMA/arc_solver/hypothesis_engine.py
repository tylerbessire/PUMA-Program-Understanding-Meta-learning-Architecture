
"""
An engine for generating, testing, and applying solution hypotheses to ARC tasks.

This module provides a framework for a more general reasoning process, where
different 'solution templates' can be tried against a task's training data.
"""

from __future__ import annotations
import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid import Array

# --- 1. Abstract Base Class for Solution Templates ---

class SolutionTemplate(abc.ABC):
    """An abstract base class for a potential solution strategy."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A unique name for this solution template."""
        pass

    @abc.abstractmethod
    def find_parameters(self, train_pairs: List[Tuple[Array, Array]]) -> Optional[Dict[str, Any]]:
        """
        Tries to find a consistent set of parameters that explains all training pairs.
        Returns a dictionary of parameters if successful, otherwise None.
        """
        pass

    @abc.abstractmethod
    def apply(self, grid: Array, params: Dict[str, Any]) -> Optional[Array]:
        """
        Applies the learned transformation with the given parameters to a single grid.
        Returns a new grid if successful, otherwise None.
        """
        pass

# --- 2. Concrete Implementation: Frame Rotation Template ---

class FrameRotationTemplate(SolutionTemplate):
    """A solution template based on finding and removing a colored frame."""

    @property
    def name(self) -> str:
        return "frame_rotation"

    def find_parameters(self, train_pairs: List[Tuple[Array, Array]]) -> Optional[Dict[str, Any]]:
        """Finds a consistent rotation, direction, and frame_color."""
        learned_transforms = []
        for inp, out in train_pairs:
            transform = self._find_single_transform(inp, out)
            if transform:
                learned_transforms.append(transform)

        if not learned_transforms or len(learned_transforms) != len(train_pairs):
            return None

        # Use the most common transform for robustness
        try:
            best_transform = max(set(learned_transforms), key=learned_transforms.count)
            return {
                "rotation_k": best_transform[0],
                "direction": best_transform[1],
                "frame_color": best_transform[2],
            }
        except (ValueError, TypeError):
            return None

    def apply(self, grid: Array, params: Dict[str, Any]) -> Optional[Array]:
        """Applies the learned frame rotation and extraction."""
        k = params.get("rotation_k")
        direction = params.get("direction")
        frame_color = params.get("frame_color")

        if k is None or direction is None or frame_color is None:
            return None

        rotated_grid = np.rot90(grid, k)
        try:
            bbox = self._find_bbox_of_color(rotated_grid, frame_color)
            if not bbox: return None
        except ValueError:
            return None

        candidates = self._get_adjacent_candidates(rotated_grid, bbox)
        raw_patch = candidates.get(direction)
        if raw_patch is None:
            return None

        # Apply the full two-stage cleaning for the final output
        solution = self._strip_background(raw_patch, frame_color)
        solution = self._trim_redundant_edges(solution)
        return solution

    def _find_single_transform(self, input_grid: Array, output_grid: Array) -> Optional[Tuple[int, str, int]]:
        """Finds the transform for a single input/output pair."""
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        frame_color_candidates = input_colors - output_colors

        for frame_color in frame_color_candidates:
            if frame_color == 0 and np.sum(input_grid == 0) > 0.5 * input_grid.size:
                continue

            for k in range(4):
                rotated_input = np.rot90(input_grid, k)
                try:
                    bbox = self._find_bbox_of_color(rotated_input, frame_color)
                    if not bbox: continue
                except ValueError:
                    continue

                candidates = self._get_adjacent_candidates(rotated_input, bbox)
                for direction, raw_patch in candidates.items():
                    # For training, use only simple background stripping to find a match
                    cleaned_patch = self._strip_background(raw_patch, frame_color)
                    if cleaned_patch.shape == output_grid.shape and np.array_equal(cleaned_patch, output_grid):
                        return (k, direction, frame_color)
        return None

    # --- Helper methods for this template ---
    def _find_bbox_of_color(self, grid: Array, color: int) -> Optional[Tuple[int, int, int, int]]:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return (r0, r1 + 1, c0, c1 + 1)

    def _get_adjacent_candidates(self, grid: Array, bbox: Tuple[int, int, int, int]) -> Dict[str, Array]:
        r0, r1, c0, c1 = bbox
        h, w = grid.shape
        candidates = {}
        if c1 < w: candidates['right'] = grid[r0:r1, c1:]
        if c0 > 0: candidates['left'] = grid[r0:r1, :c0]
        if r1 < h: candidates['below'] = grid[r1:, c0:c1]
        if r0 > 0: candidates['above'] = grid[:r0, c0:c1]
        return candidates

    def _strip_background(self, patch: Array, background_color: int) -> Array:
        if patch.size == 0: return patch
        if patch.shape[0] > 0:
            mask_rows = ~np.all(patch == background_color, axis=1)
            patch = patch[mask_rows, :]
        if patch.size == 0: return patch
        if patch.shape[1] > 0:
            mask_cols = ~np.all(patch == background_color, axis=0)
            patch = patch[:, mask_cols]
        return patch

    def _trim_redundant_edges(self, patch: Array) -> Array:
        if patch.size == 0: return patch
        while patch.shape[1] > 1 and np.array_equal(patch[:, -1], patch[:, -2]): patch = patch[:, :-1]
        while patch.shape[1] > 1 and np.array_equal(patch[:, 0], patch[:, 1]): patch = patch[:, 1:]
        while patch.shape[0] > 1 and np.array_equal(patch[-1, :], patch[-2, :]): patch = patch[:-1, :]
        while patch.shape[0] > 1 and np.array_equal(patch[0, :], patch[1, :]): patch = patch[1:, :]
        return patch

# --- 3. The Hypothesis Engine ---

class HypothesisEngine:
    """Manages and tests a list of solution templates."""

    def __init__(self):
        self._templates: List[SolutionTemplate] = []
        self._register_templates()

    def _register_templates(self):
        """Initializes all known solution templates."""
        self.register(FrameRotationTemplate())
        # Future templates can be added here, e.g.:
        # self.register(TilingTemplate())
        # self.register(ColorSwapTemplate())

    def register(self, template: SolutionTemplate):
        """Adds a new solution template to the engine."""
        self._templates.append(template)

    def solve(self, task: Dict[str, Any]) -> Optional[Array]:
        """Iterates through templates to find a solution."""
        train_pairs = []
        for pair in task.get("train", []):
            try:
                inp = np.array(pair["input"], dtype=int)
                out = np.array(pair["output"], dtype=int)
                train_pairs.append((inp, out))
            except Exception:
                continue
        
        if not train_pairs:
            return None

        for template in self._templates:
            try:
                params = template.find_parameters(train_pairs)
                if params:
                    # We found a winning hypothesis. Apply it to the test case.
                    test_input_grid = np.array(task["test"][0]["input"], dtype=int)
                    solution = template.apply(test_input_grid, params)
                    if solution is not None:
                        # For now, we return the first solution found.
                        return solution
            except Exception:
                # Best effort: if a template fails, try the next one.
                continue
        
        return None
