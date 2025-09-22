"""Trace utilities to narrate the latest RFT run."""
# [S:OBS v1] telemetry=explain_last_run pass

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Context, Rule

_LAST_TRACE: Dict[str, Any] | None = None


def record_trace(
    context: Context,
    status: str,
    rules: Sequence[Rule],
    memory: Any,
) -> None:
    """Persist the latest execution trace for later explanation."""

    global _LAST_TRACE
    ranking_events = [
        event
        for event in context.history
        if event.get("phase") == "tracking" and event.get("status") == "ranking"
    ]
    promotions = [
        (event.get("rule"), event.get("score"))
        for event in context.history
        if event.get("phase") == "tracking" and event.get("status") == "promotion"
    ]
    _LAST_TRACE = {
        "status": status,
        "stuck_reason": context.stuck_reason,
        "top_hypotheses": ranking_events[-1]["top"] if ranking_events else [],
        "promotions": promotions,
        "metrics": dict(context.metrics),
        "rule_count": len(rules),
    }


def explain_last_run(context: Optional[Context] = None) -> str:
    """Return a human readable explanation of the latest RFT execution."""

    if _LAST_TRACE is None:
        return "No RFT runs recorded yet."

    trace = _LAST_TRACE.copy()
    lines = [f"Status: {trace['status']}"]
    if trace.get("stuck_reason"):
        lines.append(f"Stuck reason: {trace['stuck_reason']}")
    top = trace.get("top_hypotheses", [])
    if top:
        lines.append("Top hypotheses:")
        for desc, score in top:
            lines.append(f"  - {desc} (score={score:.2f})")
    promotions = trace.get("promotions", [])
    if promotions:
        lines.append("Promoted rules:")
        for name, score in promotions:
            lines.append(f"  - {name} (score={score:.2f})")
    metrics = trace.get("metrics", {})
    if metrics:
        metric_summary = ", ".join(f"{k}={v}" for k, v in sorted(metrics.items()))
        lines.append(f"Metrics: {metric_summary}")
    if context is not None and context.history:
        lines.append(f"Events recorded: {len(context.history)}")
    return "\n".join(lines)


__all__ = ["record_trace", "explain_last_run"]
