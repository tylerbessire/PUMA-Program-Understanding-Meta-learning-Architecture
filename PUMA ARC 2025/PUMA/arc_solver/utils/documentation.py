"""Utility helpers for validating long-form architectural documentation."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

# [S:OBS v1] emitter=doc_metrics_logger channel=structured pass
LOG = logging.getLogger("arc.doc_metrics")

_PROGRESS_MARKER_PATTERN = re.compile(r"\[S:[A-Z]+ v\d+\][^\n]* pass", re.IGNORECASE)
_HEADING_PATTERN = re.compile(r"^#{1,6} (.+)$", re.MULTILINE)


class DocumentMetricsError(RuntimeError):
    """Raised when documentation invariants fail validation."""


@dataclass(frozen=True)
class DocumentMetrics:
    """Computed statistics for a documentation artifact."""

    total_words: int
    total_characters: int
    heading_count: int
    progress_marker_count: int

    def as_dict(self) -> Dict[str, int]:
        """Return a JSON-serialisable view of the metrics."""

        return {
            "total_words": self.total_words,
            "total_characters": self.total_characters,
            "heading_count": self.heading_count,
            "progress_marker_count": self.progress_marker_count,
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _normalise_path(doc_path: Path) -> Path:
    resolved = doc_path.expanduser().resolve()
    try:
        resolved.relative_to(_repo_root())
    except ValueError as exc:  # pragma: no cover - defensive
        raise DocumentMetricsError(
            f"Document path {resolved} escapes repository root"  # noqa: EM102
        ) from exc
    if not resolved.is_file():
        raise DocumentMetricsError(f"Document path {resolved} is not a file")
    return resolved


def _load_text(doc_path: Path) -> str:
    try:
        return doc_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - IO failures are exceptional
        raise DocumentMetricsError(f"Failed to read document {doc_path}") from exc


def _validate_headings(text: str, expected_headings: Sequence[str] | None) -> List[str]:
    headings = [match.group(1).strip() for match in _HEADING_PATTERN.finditer(text)]
    if expected_headings:
        missing = [heading for heading in expected_headings if heading not in headings]
        if missing:
            raise DocumentMetricsError(
                "Missing required headings: " + ", ".join(sorted(missing))
            )
    return headings


def _count_words(text: str) -> int:
    return len([token for token in re.split(r"\s+", text.strip()) if token])


def _should_emit_metrics() -> bool:
    return os.getenv("ARC_ARCH_DOC_METRICS_ENABLED", "1") not in {"0", "false", "False"}


def compute_document_metrics(
    doc_path: Path | str,
    *,
    expected_headings: Sequence[str] | None = None,
) -> DocumentMetrics:
    """Compute validated metrics for the supplied documentation path."""

    path = _normalise_path(Path(doc_path))
    text = _load_text(path)
    headings = _validate_headings(text, expected_headings)
    progress_markers = _PROGRESS_MARKER_PATTERN.findall(text)
    metrics = DocumentMetrics(
        total_words=_count_words(text),
        total_characters=len(text),
        heading_count=len(headings),
        progress_marker_count=len(progress_markers),
    )
    if metrics.heading_count == 0:
        raise DocumentMetricsError("Documentation must include at least one heading")
    return metrics


def emit_document_metrics(metrics: DocumentMetrics, *, correlation_id: str | None = None) -> None:
    """Emit structured metrics to the shared logger if enabled."""

    if not _should_emit_metrics():
        LOG.debug(
            "architecting_abstraction_metrics_disabled",
            extra={"correlation_id": correlation_id},
        )
        return
    payload: Dict[str, object] = {
        "event": "architecting_abstraction_metrics",
        "metrics": metrics.as_dict(),
    }
    if correlation_id:
        payload["correlation_id"] = correlation_id
    LOG.info(json.dumps(payload, sort_keys=True))


__all__ = [
    "DocumentMetrics",
    "DocumentMetricsError",
    "compute_document_metrics",
    "emit_document_metrics",
]
