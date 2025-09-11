"""Hypothesis generation and testing for fluid intelligence in ARC tasks.

This module implements a lightweight hypothesis generation framework.  It is
not meant to be an exhaustive reasoning system but rather a scaffold that can
be extended in later phases.  The engine analyses training pairs and proposes
plausible transformations together with simple confidence estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid import Array


@dataclass
class Hypothesis:
    """Represents a hypothesis about task transformation."""

    description: str
    transformation_type: str  # "rotation", "color_swap", "pattern_fill", etc.
    confidence: float
    evidence: List[Dict[str, Any]]
    program_sketch: Optional[List[Tuple[str, Dict[str, Any]]]] = None


class HypothesisEngine:
    """Generates and tests hypotheses about ARC task transformations."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        """Generate multiple competing hypotheses about the task transformation."""
        hypotheses: List[Hypothesis] = []

        # 1. Geometric transformation hypotheses
        hypotheses.extend(self._generate_geometric_hypotheses(train_pairs))

        # 2. Color transformation hypotheses
        hypotheses.extend(self._generate_color_hypotheses(train_pairs))

        # 3. Pattern completion hypotheses
        hypotheses.extend(self._generate_pattern_hypotheses(train_pairs))

        # 4. Object manipulation hypotheses
        hypotheses.extend(self._generate_object_hypotheses(train_pairs))

        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def test_hypothesis(self, hypothesis: Hypothesis, train_pairs: List[Tuple[Array, Array]]) -> float:
        """Test hypothesis validity against training data.

        Returns a confidence score between 0 and 1 based on how many training
        pairs are perfectly explained by the hypothesis.
        """
        if not train_pairs:
            return 0.0
        matches = 0
        for inp, out in train_pairs:
            pred = self.apply(hypothesis, inp)
            if pred is not None and pred.shape == out.shape and np.array_equal(pred, out):
                matches += 1
        return matches / len(train_pairs)

    def refine_hypothesis(self, hypothesis: Hypothesis, feedback: Dict[str, Any]) -> Hypothesis:
        """Refine hypothesis based on test results.

        A very small refinement mechanism is provided for now: the confidence is
        updated if feedback contains a ``confidence`` field and any additional
        evidence is appended to the hypothesis' evidence list.
        """
        new_conf = float(feedback.get("confidence", hypothesis.confidence))
        new_evidence = hypothesis.evidence + [feedback.get("evidence", {})]
        return Hypothesis(
            description=hypothesis.description,
            transformation_type=hypothesis.transformation_type,
            confidence=new_conf,
            evidence=new_evidence,
            program_sketch=hypothesis.program_sketch,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def apply(self, hypothesis: Hypothesis, grid: Array) -> Optional[Array]:
        """Apply the hypothesis to a grid, returning the transformed grid."""
        try:
            if hypothesis.transformation_type == "rotation" and hypothesis.program_sketch:
                k = int(hypothesis.program_sketch[0][1].get("k", 0))
                return np.rot90(grid, k)
            if hypothesis.transformation_type == "color_swap" and hypothesis.program_sketch:
                mapping = hypothesis.program_sketch[0][1].get("mapping", {})
                result = grid.copy()
                for src, dst in mapping.items():
                    result[grid == src] = dst
                return result
            if hypothesis.transformation_type == "pattern_fill" and hypothesis.program_sketch:
                color = int(hypothesis.program_sketch[0][1].get("color", 0))
                return np.full_like(grid, color)
            if hypothesis.transformation_type == "object_translation" and hypothesis.program_sketch:
                params = hypothesis.program_sketch[0][1]
                dy = int(params.get("dy", 0))
                dx = int(params.get("dx", 0))
                h, w = grid.shape
                result = np.zeros_like(grid)
                ys, xs = np.nonzero(grid)
                ys_new = ys + dy
                xs_new = xs + dx
                if (
                    (ys_new < 0).any()
                    or (ys_new >= h).any()
                    or (xs_new < 0).any()
                    or (xs_new >= w).any()
                ):
                    return None
                result[ys_new, xs_new] = grid[ys, xs]
                return result
        except Exception:
            return None
        return None

    # Hypothesis generation subroutines ---------------------------------
    def _generate_geometric_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        rotations = [1, 2, 3]  # 90, 180, 270 degrees
        for k in rotations:
            evidence: List[Dict[str, Any]] = []
            matches = 0
            for idx, (inp, out) in enumerate(train_pairs):
                rotated = np.rot90(inp, k)
                match = rotated.shape == out.shape and np.array_equal(rotated, out)
                evidence.append({"pair": idx, "rotation": k * 90, "match": match})
                if match:
                    matches += 1
            confidence = matches / len(train_pairs)
            if confidence > 0:
                hyps.append(
                    Hypothesis(
                        description=f"Rotate input by {k * 90} degrees",
                        transformation_type="rotation",
                        confidence=confidence,
                        evidence=evidence,
                        program_sketch=[("rotate", {"k": k})],
                    )
                )
        return hyps

    def _generate_color_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        global_mapping: Dict[int, int] = {}
        evidence: List[Dict[str, Any]] = []
        consistent = True
        for idx, (inp, out) in enumerate(train_pairs):
            mapping: Dict[int, int] = {}
            for ci, co in zip(inp.flat, out.flat):
                if ci in mapping and mapping[ci] != int(co):
                    consistent = False
                    break
                if ci != co:
                    mapping[ci] = int(co)
            evidence.append({"pair": idx, "mapping": mapping})
            for k, v in mapping.items():
                if k in global_mapping and global_mapping[k] != v:
                    consistent = False
                    break
                global_mapping[k] = v
            if not consistent:
                break
        if consistent and global_mapping:
            hyps.append(
                Hypothesis(
                    description=f"Recolor using mapping {global_mapping}",
                    transformation_type="color_swap",
                    confidence=1.0,
                    evidence=evidence,
                    program_sketch=[("recolor", {"mapping": global_mapping})],
                )
            )
        return hyps

    def _generate_pattern_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        colors = [np.unique(out) for _, out in train_pairs]
        if colors and all(len(c) == 1 for c in colors):
            color = int(colors[0][0])
            evidence = [{"pair": idx, "color": int(c[0])} for idx, c in enumerate(colors)]
            hyps.append(
                Hypothesis(
                    description=f"Fill grid with color {color}",
                    transformation_type="pattern_fill",
                    confidence=1.0,
                    evidence=evidence,
                    program_sketch=[("fill", {"color": color})],
                )
            )
        return hyps

    def _find_translation(self, inp: Array, out: Array) -> Optional[Tuple[int, int]]:
        if inp.shape != out.shape:
            return None
        coords_in = np.argwhere(inp != 0)
        coords_out = np.argwhere(out != 0)
        if len(coords_in) == 0 or len(coords_out) == 0 or len(coords_in) != len(coords_out):
            return None
        shift = coords_out[0] - coords_in[0]
        h, w = inp.shape
        translated = np.zeros_like(inp)
        ys = coords_in[:, 0] + shift[0]
        xs = coords_in[:, 1] + shift[1]
        if (ys < 0).any() or (ys >= h).any() or (xs < 0).any() or (xs >= w).any():
            return None
        translated[ys, xs] = inp[coords_in[:, 0], coords_in[:, 1]]
        if np.array_equal(translated, out):
            return int(shift[0]), int(shift[1])
        return None

    def _generate_object_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        shifts: List[Tuple[int, int]] = []
        evidence: List[Dict[str, Any]] = []
        for idx, (inp, out) in enumerate(train_pairs):
            trans = self._find_translation(inp, out)
            evidence.append({"pair": idx, "shift": trans})
            if trans is not None:
                shifts.append(trans)
        if not shifts:
            return hyps
        common = shifts[0]
        if any(s != common for s in shifts):
            return hyps
        confidence = len(shifts) / len(train_pairs)
        hyps.append(
            Hypothesis(
                description=f"Translate object by {common}",
                transformation_type="object_translation",
                confidence=confidence,
                evidence=evidence,
                program_sketch=[("translate", {"dy": common[0], "dx": common[1]})],
            )
        )
        return hyps
