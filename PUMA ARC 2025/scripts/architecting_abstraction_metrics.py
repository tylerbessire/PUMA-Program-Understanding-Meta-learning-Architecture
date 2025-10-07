#!/usr/bin/env python3
"""Emit observability metrics for the architecting abstraction dossier."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

PROJECT_PYTHON_ROOT = Path(__file__).resolve().parents[1] / "PUMA"
if str(PROJECT_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PYTHON_ROOT))

# [S:OBS v1] cli=architecting_abstraction_metrics status=shipping pass
from arc_solver.utils import (
    DocumentMetricsError,
    compute_document_metrics,
    emit_document_metrics,
)

_LOGGER = logging.getLogger("arc.doc_metrics.cli")

_REQUIRED_HEADINGS: List[str] = [
    "Part I: Architectural Foundations for Large-Scale Libraries",
    "Chapter 1: Taming Complexity - Architectural Patterns for Monolithic Libraries",
    "Chapter 2: The Art of the API - Principles of Modern Library Design",
    "Chapter 3: Pythonic Structure at Scale",
    "Part II: Blueprint for a General-Purpose Library with 1,000 Functions & Chains",
    "Chapter 4: The DSL Decision - Internal vs. External Languages",
    "Chapter 5: Implementation of an Internal DSL via Fluent Interfaces",
    "Chapter 6: Implementation of an External DSL",
    "Part III: Blueprint for a Relational Frame Theory (RFT) Computational Library",
    "Chapter 7: Deconstructing Relational Frame Theory for Computation",
    "Chapter 8: An RFT-Driven Architecture - Modeling Relational Networks",
    "Chapter 9: A Fluent API for Relational Reasoning",
    "Part IV: Engineering for Longevity and Reliability",
    "Chapter 10: A Comprehensive Multi-Layered Testing Strategy",
    "Chapter 11: Lifecycle Management - Versioning, Documentation, and Community",
    "Conclusions",
]


def _default_doc_path() -> Path:
    return Path(__file__).resolve().parents[1] / "Architecting_Abstraction.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and emit structured metrics for the architecting abstraction dossier",
    )
    parser.add_argument(
        "doc_path",
        nargs="?",
        default=_default_doc_path(),
        type=Path,
        help="Path to the Markdown dossier (defaults to project copy).",
    )
    parser.add_argument(
        "--correlation-id",
        dest="correlation_id",
        default=None,
        help="Optional correlation identifier for log aggregation.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        metrics = compute_document_metrics(
            args.doc_path,
            expected_headings=_REQUIRED_HEADINGS,
        )
    except DocumentMetricsError as exc:
        _LOGGER.error(
            "architecting_abstraction_metrics_failed", extra={"error": str(exc)}
        )
        return 1
    emit_document_metrics(metrics, correlation_id=args.correlation_id)
    print(json.dumps(metrics.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
