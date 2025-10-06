"""
Object Evolution Tracker for ARC Solver.

This module provides the core logic for identifying and tracking objects across
input/output grid pairs in an ARC task. It works by:
1.  Fingerprinting each object in the input and output grids using a set of
    stable features (shape, size, color, position).
2.  Solving the assignment problem to find the most likely correspondence
    between input and output objects.
3.  Calculating the "delta" or transformation for each matched pair.

This information is fundamental for inducing transformation rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .grid import Array
from .objects import connected_components


@dataclass(frozen=True)
class ObjectFingerprint:
    """A stable identifier for an object within a grid."""
    uid: int
    color: int
    size: int
    height: int
    width: int
    position: Tuple[float, float]
    shape_hash: int
    raw_object: Dict[str, Any] = field(repr=False)

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        return isinstance(other, ObjectFingerprint) and self.uid == other.uid


@dataclass
class ObjectEvolution:
    """Represents the transformation of a single object."""
    input_obj: ObjectFingerprint
    output_obj: Optional[ObjectFingerprint] = None
    delta: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0


class ObjectTracker:
    """
    Analyzes object transformations between input and output grids.
    """

    def __init__(self):
        pass

    def _fingerprint_objects(
        self,
        grid: Array,
        components: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ObjectFingerprint]:
        """Extracts and fingerprints objects from a grid or a precomputed component list."""

        objects = components if components is not None else connected_components(grid)
        fingerprints: List[ObjectFingerprint] = []

        for idx, obj in enumerate(objects):
            color = int(obj["color"])

            if "pixels" in obj:  # canonical format from connected_components
                top, left, height, width = obj["bbox"]
                pixels = obj["pixels"]
                ys = [p[0] for p in pixels]
                xs = [p[1] for p in pixels]
                centroid_r = top + np.mean(ys) - min(ys)
                centroid_c = left + np.mean(xs) - min(xs)
                mask = obj.get("mask")
                if mask is not None:
                    shape_hash = hash(tuple((mask != 0).flatten()))
                else:
                    rel_tiles = [(r - top, c - left) for r, c in pixels]
                    shape_hash = hash(tuple(sorted(rel_tiles)))
                size = len(pixels)

            else:  # detector-style component with explicit tile list
                tiles: List[Tuple[int, int]] = obj.get("tiles", [])
                bbox = obj.get("bbox")
                if bbox is None:
                    continue
                top, left, bottom, right = bbox
                height = bottom - top
                width = right - left
                if not tiles:
                    continue
                rel_tiles = [(r - top, c - left) for r, c in tiles]
                ys = [r for r, _ in rel_tiles]
                xs = [c for _, c in rel_tiles]
                centroid_r = top + float(np.mean(ys))
                centroid_c = left + float(np.mean(xs))
                shape_hash = hash(tuple(sorted(rel_tiles)))
                size = obj.get("area", len(tiles))

            position = (float(centroid_r), float(centroid_c))

            fingerprints.append(
                ObjectFingerprint(
                    uid=idx,
                    color=color,
                    size=int(size),
                    height=int(height),
                    width=int(width),
                    position=position,
                    shape_hash=shape_hash,
                    raw_object=obj,
                )
            )

        return fingerprints

    def _calculate_distance(self, in_obj: ObjectFingerprint, out_obj: ObjectFingerprint) -> float:
        """
        Calculates a weighted cost of matching an input object to an output object.
        Lower cost means a better match.
        """
        # Weights for different features. These can be tuned.
        W_SHAPE = 10.0
        W_COLOR = 5.0
        W_SIZE = 1.0
        W_POS = 0.1

        # Shape distance: binary (either same shape or not)
        shape_dist = 0.0 if in_obj.shape_hash == out_obj.shape_hash else 1.0

        # Color distance: binary
        color_dist = 0.0 if in_obj.color == out_obj.color else 1.0

        # Size distance: normalized absolute difference
        size_dist = abs(in_obj.size - out_obj.size) / max(in_obj.size, out_obj.size)

        # Position distance: Euclidean distance
        pos_dist = np.linalg.norm(np.array(in_obj.position) - np.array(out_obj.position))

        # Total weighted cost
        cost = (
            shape_dist * W_SHAPE +
            color_dist * W_COLOR +
            size_dist * W_SIZE +
            pos_dist * W_POS
        )
        return cost

    def track_evolutions(
        self,
        input_grid: Array,
        output_grid: Array,
        input_components: Optional[List[Dict[str, Any]]] = None,
        output_components: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ObjectEvolution]:
        """
        Tracks object evolutions from an input grid to an output grid.
        """
        input_objects = self._fingerprint_objects(input_grid, input_components)
        output_objects = self._fingerprint_objects(output_grid, output_components)

        if not input_objects or not output_objects:
            return []

        # Build the cost matrix for the assignment problem
        cost_matrix = np.zeros((len(input_objects), len(output_objects)))
        for i, in_obj in enumerate(input_objects):
            for j, out_obj in enumerate(output_objects):
                cost_matrix[i, j] = self._calculate_distance(in_obj, out_obj)

        # Solve the assignment problem using the Hungarian algorithm
        in_indices, out_indices = linear_sum_assignment(cost_matrix)

        evolutions = []
        for i, j in zip(in_indices, out_indices):
            in_obj = input_objects[i]
            out_obj = output_objects[j]
            cost = cost_matrix[i, j]

            # Calculate the delta
            delta = {}
            if in_obj.color != out_obj.color:
                delta['color'] = (in_obj.color, out_obj.color)
            if in_obj.size != out_obj.size:
                delta['size'] = (in_obj.size, out_obj.size)
            if in_obj.shape_hash != out_obj.shape_hash:
                delta['shape'] = 'changed'
            
            pos_delta = np.array(out_obj.position) - np.array(in_obj.position)
            if np.linalg.norm(pos_delta) > 0.1:
                delta['position'] = tuple(pos_delta)

            evolutions.append(
                ObjectEvolution(
                    input_obj=in_obj,
                    output_obj=out_obj,
                    delta=delta,
                    cost=cost
                )
            )
        
        return evolutions
