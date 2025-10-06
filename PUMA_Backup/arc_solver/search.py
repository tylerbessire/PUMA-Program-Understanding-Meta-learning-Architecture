"""
Program synthesis search for ARC tasks.

This module provides a very simple search over compositions of DSL operations.
Given a list of training input/output pairs, it attempts to find candidate
programs that exactly map all inputs to outputs. Programs are evaluated by
brute-force enumeration with a depth limit and heuristics.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
from .grid import Array, eq
from .dsl import OPS, apply_program
from .heuristics import consistent_program_single_step, score_candidate, diversify_programs


def enumerate_programs(depth: int = 2) -> List[List[Tuple[str, Dict[str, int]]]]:
    """Enumerate sequences of operations up to a fixed depth.

    The parameter space for each operation is constrained to small ranges to keep
    enumeration tractable. Identity is included for completeness.
    """
    # Parameter grids for each operation
    param_space = {
        "rotate": [{"k": k} for k in [0, 1, 2, 3]],
        "flip": [{"axis": a} for a in [0, 1]],
        "transpose": [{}],
        "translate": [
            {"dy": dy, "dx": dx}
            for dy in range(-2, 3)
            for dx in range(-2, 3)
        ],
        "crop": [
            {"top": t, "left": l, "height": h, "width": w}
            for t in range(3)
            for l in range(3)
            for h in range(1, 4)
            for w in range(1, 4)
        ],
        "pad": [
            {"out_h": h, "out_w": w}
            for h in range(5, 20)
            for w in range(5, 20)
        ],
        "recolor": [
            {"mapping": {i: j}}
            for i in range(10)
            for j in range(10)
            if i != j
        ],
        "find_color_region": [
            {"color": c} for c in range(10)
        ],
        "extract_marked_region": [
            {"marker_color": c} for c in [8, 9, 7, 6]
        ],
        "smart_crop_auto": [{}],
        "extract_symmetric_region": [{}],
        "extract_pattern_region": [
            {"marker_color": c} for c in [8, 9, 7, 6]
        ],
        "identity": [{}],
    }
    base_ops = [
        "rotate",
        "flip",
        "transpose",
        "translate",
        "crop",
        "pad",
        "recolor",
        "find_color_region",
        "extract_marked_region", 
        "smart_crop_auto",
        "extract_symmetric_region",
        "extract_pattern_region",
        "identity",
    ]
    # Generate all length-1 programs
    if depth == 1:
        return [[(op, params)] for op in base_ops for params in param_space[op]]
    # Generate length-2 programs
    programs: List[List[Tuple[str, Dict[str, int]]]] = []
    for op1 in base_ops:
        for p1 in param_space[op1]:
            for op2 in base_ops:
                for p2 in param_space[op2]:
                    programs.append([(op1, p1), (op2, p2)])
    return programs


def synthesize(train_pairs: List[Tuple[Array, Array]], max_programs: int = 256, expected_shape: Optional[Tuple[int, int]] = None) -> List[List[Tuple[str, Dict[str, int]]]]:
    """Synthesize a small set of candidate programs that fit the training pairs.

    First attempts to find single-operation programs via heuristics. If none
    succeed, enumerates two-step programs and scores them, keeping the best
    matches. Resulting programs are deduplicated.
    """
    # 1) Try heuristic single-step fits
    progs = consistent_program_single_step(train_pairs)
    # 2) If no single-step fit, brute force search depth-2 compositions
    if not progs:
        cand_progs = enumerate_programs(depth=2)
        scored = []
        for program in cand_progs:
            s = score_candidate(program, train_pairs, expected_shape=expected_shape)
            if s > 0.99:  # require exact matches
                scored.append((s, program))
        # sort by fit score (descending) and take top max_programs
        scored.sort(key=lambda x: -x[0])
        progs = [p for _, p in scored[:max_programs]]
    progs = diversify_programs(progs)
    return progs[:max_programs]


def predict_two(
    progs: List[List[Tuple[str, Dict[str, int]]]],
    test_inputs: List[Array],
    prefer_diverse: bool = False,
) -> List[List[Array]]:
    """Select up to two programs and apply them to each test input.

    Args:
        progs: Candidate programs ordered by preference.
        test_inputs: Test grids to which programs are applied.
        prefer_diverse: If ``True`` and multiple programs are available,
            attempt to ensure the two selected programs differ.

    Returns:
        A list ``[attempt_1, attempt_2]`` where each element is a list of
        output grids corresponding to ``test_inputs``.
    """
    if not progs:
        # Instead of just identity, try some smart fallbacks
        fallback_programs = [
            [("extract_pattern_region", {"marker_color": 8})],
            [("smart_crop_auto", {})],
            [("extract_marked_region", {"marker_color": 8})],
            [("find_color_region", {"color": 8})],
            [("identity", {})]
        ]
        picks = fallback_programs[:2]
    elif prefer_diverse and len(progs) > 1:
        picks = [progs[0], progs[1]]
    else:
        picks = progs[:2] if len(progs) >= 2 else [progs[0], progs[0]]

    attempts: List[List[Array]] = []
    for program in picks:
        outs: List[Array] = []
        for ti in test_inputs:
            try:
                outs.append(apply_program(ti, program))
            except Exception:
                # Better fallback strategy - try smart cropping before identity
                try:
                    outs.append(apply_program(ti, [("extract_pattern_region", {"marker_color": 8})]))
                except Exception:
                    try:
                        outs.append(apply_program(ti, [("smart_crop_auto", {})]))
                    except Exception:
                        try:
                            outs.append(apply_program(ti, [("extract_marked_region", {"marker_color": 8})]))
                        except Exception:
                            outs.append(ti)  # Final fallback to identity
        attempts.append(outs)

    # Ensure second attempt differs from the first using safe array comparison
    if len(attempts) == 2 and all(eq(a, b) for a, b in zip(attempts[0], attempts[1])):
        # Try a different fallback for second attempt
        outs: List[Array] = []
        for ti in test_inputs:
            try:
                outs.append(apply_program(ti, [("extract_marked_region", {"marker_color": 8})]))
            except Exception:
                try:
                    outs.append(apply_program(ti, [("find_color_region", {"color": 8})]))
                except Exception:
                    outs.append(np.copy(ti))
        attempts[1] = outs

    # [S:ALG v1] attempt-dedup=eq-fallback pass
    return attempts
