"""
Heuristic helpers for the ARC solver.

This module implements simple pattern inference routines used to constrain the
search space for program synthesis. Functions here try to detect if a single
transformation (e.g., rotation, recoloring, translation) explains all training
pairs. They also provide scoring and diversification utilities for candidate
programs.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional

from .grid import Array, eq, rotate90, flip, histogram, bg_color, to_array
from .dsl import apply_program

logger = logging.getLogger(__name__)

__all__ = [
    "infer_color_mapping",
    "match_rotation_reflection",
    "infer_translation",
    "consistent_program_single_step",
    "guess_output_shape",
    "score_candidate",
    "score_candidate_partial",
    "diversify_programs",
]


def infer_color_mapping(inp: Array, out: Array) -> Optional[Dict[int, int]]:
    """Try to infer a one-to-one color mapping between input and output grids.

    Returns a mapping dict if the output can be obtained by recoloring the
    input without changing shape or pixel positions. Returns None otherwise.
    """
    if inp.shape != out.shape:
        return None
    mapping: Dict[int, int] = {}
    for v_in, v_out in zip(inp.flatten(), out.flatten()):
        vi, vo = int(v_in), int(v_out)
        if vi in mapping and mapping[vi] != vo:
            return None
        mapping[vi] = vo
    return mapping


def match_rotation_reflection(inp: Array, out: Array) -> Optional[Tuple[str, Dict[str, int]]]:
    """Detect if out is a simple rotation, flip or transpose of inp.

    Returns the operation name and parameters if a match is found.
    """
    if inp.shape == out.shape:
        for k in range(4):
            if np.array_equal(rotate90(inp, k), out):
                return ("rotate", {"k": k})
        if np.array_equal(flip(inp, 0), out):
            return ("flip", {"axis": 0})
        if np.array_equal(flip(inp, 1), out):
            return ("flip", {"axis": 1})
        if np.array_equal(inp.T, out):
            return ("transpose", {})
    return None


def infer_translation(inp: Array, out: Array) -> Optional[Tuple[str, Dict[str, int]]]:
    """Infer a small translation between input and output.

    Checks displacements within a small range (default +/-3) and uses the
    output's background color for padding.
    """
    if inp.shape != out.shape:
        return None
    bg = bg_color(out)
    H, W = inp.shape
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifted = np.full_like(inp, bg)
            y0, y1 = max(0, dy), min(H, H + dy)
            x0, x1 = max(0, dx), min(W, W + dx)
            
            # Check bounds to avoid empty slices
            src_y0 = max(0, y0 - dy)
            src_y1 = min(H, y1 - dy)
            src_x0 = max(0, x0 - dx)
            src_x1 = min(W, x1 - dx)
            
            dst_y0 = y0 + (src_y0 - (y0 - dy))
            dst_y1 = y1 - ((y1 - dy) - src_y1)
            dst_x0 = x0 + (src_x0 - (x0 - dx))
            dst_x1 = x1 - ((x1 - dx) - src_x1)
            
            if src_y1 > src_y0 and src_x1 > src_x0 and dst_y1 > dst_y0 and dst_x1 > dst_x0:
                shifted[dst_y0:dst_y1, dst_x0:dst_x1] = inp[src_y0:src_y1, src_x0:src_x1]
            if np.array_equal(shifted, out):
                return ("translate", {"dy": dy, "dx": dx})
    return None


def consistent_program_single_step(pairs: List[Tuple[Array, Array]]) -> List[List[Tuple[str, Dict[str, int]]]]:
    """Try to find single-operation programs that fit all training pairs.

    Considers rotation/flip/transpose, translation, and recolor operations. Returns
    a list of candidate one-step programs.
    """
    cands: List[List[Tuple[str, Dict[str, int]]]] = []
    # Rotation / reflection / transpose
    ref = match_rotation_reflection(pairs[0][0], pairs[0][1])
    if ref is not None and all(match_rotation_reflection(a, b) == ref for a, b in pairs):
        cands.append([ref])
    # Translation
    tr = infer_translation(pairs[0][0], pairs[0][1])
    if tr is not None and all(infer_translation(a, b) == tr for a, b in pairs):
        cands.append([tr])
    # Recolor
    cm = infer_color_mapping(pairs[0][0], pairs[0][1])
    if cm is not None and all(infer_color_mapping(a, b) == cm for a, b in pairs):
        cands.append([("recolor", {"mapping": cm})])
    return cands


def guess_output_shape(pairs: List[Tuple[Array, Array]]) -> Optional[Tuple[int, int]]:
    """Infer a common output shape if train outputs all share the same shape."""
    shapes = [b.shape for _, b in pairs]
    if all(s == shapes[0] for s in shapes):
        return shapes[0]
    return None


def score_candidate(program: List[Tuple[str, Dict[str, int]]], train_pairs: List[Tuple[Array, Array]], expected_shape: Optional[Tuple[int, int]] = None) -> float:
    """Compute the proportion of train pairs exactly matched by the candidate program."""
    
    # Special handling for human reasoning candidates (using metadata flag)
    if (len(program) == 1 and program[0][1].get('_source') == 'human_reasoner'):
        # For human reasoning, use the verification score that was already computed
        # during hypothesis generation - don't re-run the expensive analysis
        verification_score = program[0][1].get('verification_score', 0.0)
        hypothesis_name = program[0][0]
        print(f"DEBUG: Scoring human reasoning '{hypothesis_name}': {verification_score:.3f}")
        return verification_score
    
    # Standard scoring for regular programs
    good = 0
    for a, b in train_pairs:
        try:
            out = apply_program(a, program)
            if expected_shape and out.shape != expected_shape:
                return 0.0 # Prune programs that produce the wrong shape
            good += int(eq(out, b))
        except Exception as exc:
            logger.warning("Program execution failed on training pair: %s", exc)
    return good / len(train_pairs)


def score_candidate_partial(program: List[Tuple[str, Dict[str, int]]], train_pairs: List[Tuple[Array, Array]]) -> float:
    """Compute a partial score for a candidate program."""
    scores = []
    for a, b in train_pairs:
        try:
            out = apply_program(a, program)
            
            # Pixel-wise accuracy
            if out.shape == b.shape:
                pixel_accuracy = np.sum(out == b) / b.size
            else:
                pixel_accuracy = 0

            # Shape similarity
            shape_similarity = min(out.size, b.size) / max(out.size, b.size)

            # Combined score
            score = 0.7 * pixel_accuracy + 0.3 * shape_similarity
            scores.append(score)
        except Exception as exc:
            logger.warning("Program execution failed on training pair: %s", exc)
            scores.append(0.0)
    return np.mean(scores)


def diversify_programs(programs: List[List[Tuple[str, Dict[str, int]]]]) -> List[List[Tuple[str, Dict[str, int]]]]:
    """Remove duplicate programs by string representation to diversify output."""
    seen = set()
    uniq: List[List[Tuple[str, Dict[str, int]]]] = []
    for p in programs:
        key = repr(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq