"""Utility helpers for the ARC solver package."""

# [S:PKG v1] module=arc_solver.utils status=initialised pass

from .documentation import (
    DocumentMetrics,
    DocumentMetricsError,
    compute_document_metrics,
    emit_document_metrics,
)

__all__ = [
    "DocumentMetrics",
    "DocumentMetricsError",
    "compute_document_metrics",
    "emit_document_metrics",
]
