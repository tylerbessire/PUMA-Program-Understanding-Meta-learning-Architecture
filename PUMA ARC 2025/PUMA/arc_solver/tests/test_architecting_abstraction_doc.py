"""Regression tests for the architecting abstraction dossier."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from arc_solver.utils import (
    DocumentMetrics,
    compute_document_metrics,
    emit_document_metrics,
)

DOC_PATH = Path(__file__).resolve().parents[3] / "Architecting_Abstraction.md"
DOC_TEXT = DOC_PATH.read_text(encoding="utf-8")

EXPECTED_HEADINGS = [
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


@pytest.fixture(scope="session")
def doc_metrics() -> DocumentMetrics:
    return compute_document_metrics(DOC_PATH, expected_headings=EXPECTED_HEADINGS)


def test_document_exists() -> None:
    assert DOC_PATH.is_file(), "The architecting abstraction dossier must be present."


def test_metrics_counts_are_positive(doc_metrics) -> None:
    assert doc_metrics.total_words > 1000
    assert doc_metrics.heading_count >= len(EXPECTED_HEADINGS)
    assert doc_metrics.progress_marker_count >= 2


@given(st.sampled_from(EXPECTED_HEADINGS))
def test_each_heading_occurs_once(heading: str) -> None:
    assert DOC_TEXT.count(heading) == 1


def test_emit_document_metrics_logs_json(monkeypatch, caplog, doc_metrics) -> None:
    monkeypatch.delenv("ARC_ARCH_DOC_METRICS_ENABLED", raising=False)
    with caplog.at_level("INFO", logger="arc.doc_metrics"):
        emit_document_metrics(doc_metrics, correlation_id="ci-123")
    record = next(
        entry for entry in caplog.records if entry.name == "arc.doc_metrics"
    )
    payload = json.loads(record.message)
    assert payload["event"] == "architecting_abstraction_metrics"
    assert payload["metrics"]["total_words"] == doc_metrics.total_words
    assert payload["correlation_id"] == "ci-123"


def test_emit_document_metrics_respects_feature_flag(monkeypatch, caplog, doc_metrics) -> None:
    monkeypatch.setenv("ARC_ARCH_DOC_METRICS_ENABLED", "0")
    with caplog.at_level("DEBUG", logger="arc.doc_metrics"):
        emit_document_metrics(doc_metrics, correlation_id="ci-flagged")
    assert any(
        entry.message == "architecting_abstraction_metrics_disabled"
        for entry in caplog.records
    )
