# [S:ALG v1] strategy=beam_search nodes_metric=on pass
import logging
from typing import List, Tuple, Dict, Any
from .grid import Array
from .dsl import OPS
from .heuristics import score_candidate
from .neural.sketches import generate_parameter_grid

logger = logging.getLogger(__name__)


def beam_search(
    train_pairs: List[Tuple[Array, Array]],
    beam_width: int = 10,
    depth: int = 2,
    max_expansions: int = 10000,
) -> Tuple[List[List[Tuple[str, Dict[str, Any]]]], Dict[str, int]]:
    """Beam search over DSL programs.

    Args:
        train_pairs: Training examples as ``[(input, output), ...]``.
        beam_width: Number of candidates kept per level.
        depth: Maximum program length.
        max_expansions: Safety limit on node expansions.

    Returns:
        A tuple ``(programs, stats)`` where ``programs`` is a list of candidate
        programs matching all training pairs exactly and ``stats`` contains
        observability metrics.
    """
    if beam_width <= 0 or depth <= 0:
        raise ValueError("beam_width and depth must be positive")

    beam: List[Tuple[List[Tuple[str, Dict[str, Any]]], float]] = [([], 1.0)]
    complete: List[List[Tuple[str, Dict[str, Any]]]] = []
    nodes_expanded = 0

    for _ in range(depth):
        expansions: List[Tuple[List[Tuple[str, Dict[str, Any]]], float]] = []
        for program, _ in beam:
            for op_name in OPS.keys():
                for params in generate_parameter_grid(op_name):
                    candidate = program + [(op_name, params)]
                    try:
                        score = score_candidate(candidate, train_pairs)
                    except Exception:
                        continue  # constraint violation
                    nodes_expanded += 1
                    if score >= 0.999:
                        complete.append(candidate)
                    else:
                        expansions.append((candidate, score))
                    if nodes_expanded >= max_expansions:
                        logger.warning(
                            "beam_search max expansions reached",
                            extra={"nodes_expanded": nodes_expanded},
                        )
                        break
                if nodes_expanded >= max_expansions:
                    break
            if nodes_expanded >= max_expansions:
                break
        expansions.sort(key=lambda x: x[1], reverse=True)
        beam = expansions[:beam_width]
        if not beam:
            break

    complete = complete[:beam_width]
    logger.info(
        "beam_search complete",
        extra={"nodes_expanded": nodes_expanded, "solutions": len(complete)},
    )
    return complete, {"nodes_expanded": nodes_expanded}