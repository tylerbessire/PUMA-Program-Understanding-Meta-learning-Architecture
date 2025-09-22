"""Coordinator that alternates pliance and tracking phases."""
# [S:CTRL v2] controller=rft_orchestrator pass

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

from .explain import record_trace
from .pliance import pliance_solve
from .tracking import HypothesisMemory, tracking_step
from .types import Context, Rule

logger = logging.getLogger("puma.rft.orchestrator")
logger.addHandler(logging.NullHandler())


def insert_after_stuck_rule(rules: Sequence[Rule], anchor: str | None, new_rule: Rule) -> List[Rule]:
    """Insert ``new_rule`` immediately after the anchor that last caused a stall."""

    updated = list(rules)
    if anchor is None:
        updated.append(new_rule)
        logger.debug(
            {
                "phase": "orchestrator",
                "event": "insert_rule",
                "anchor": None,
                "rule": new_rule.name,
                "reason": "no_anchor",
            }
        )
        return updated
    for idx, rule in enumerate(updated):
        if rule.name == anchor:
            updated.insert(idx + 1, new_rule)
            logger.debug(
                {
                    "phase": "orchestrator",
                    "event": "insert_rule",
                    "anchor": anchor,
                    "rule": new_rule.name,
                    "index": idx + 1,
                }
            )
            return updated
    updated.append(new_rule)
    logger.debug(
        {
            "phase": "orchestrator",
            "event": "insert_rule",
            "anchor": anchor,
            "rule": new_rule.name,
            "reason": "anchor_not_found",
        }
    )
    return updated


def solve_with_rft(
    context: Context,
    rules: Sequence[Rule],
    memory: HypothesisMemory,
    outer_budget: int | None = None,
) -> Tuple[Context, str]:
    """Run pliance and tracking with adaptive promotions."""

    current_rules = list(rules)
    for promoted in memory.promoted_rules():
        if all(existing.name != promoted.name for existing in current_rules):
            current_rules.append(promoted)
    if len(current_rules) > max(32, context.limits.tracking_budget * 2):
        logger.warning(
            {
                "phase": "orchestrator",
                "event": "rule_overflow",
                "count": len(current_rules),
            }
        )
    outer_budget = outer_budget or context.limits.outer_budget
    for outer in range(outer_budget):
        context.metric_inc("rft_outer_iterations")
        context.record_event({"phase": "orchestrator", "outer": outer, "status": "pliance"})
        context, status = pliance_solve(
            context,
            current_rules,
            max_steps=context.limits.pliance_steps,
        )
        if status == "GOAL":
            record_trace(context, status=status, rules=current_rules, memory=memory)
            return context, "DONE"

        if status not in {"PLIANCE_STUCK", "STEP_LIMIT"}:
            context.record_event({"phase": "orchestrator", "status": status})
            record_trace(context, status=status, rules=current_rules, memory=memory)
            return context, status

        context.record_event(
            {
                "phase": "orchestrator",
                "status": "tracking",
                "reason": context.stuck_reason,
            }
        )
        new_rule = tracking_step(
            context,
            current_rules,
            memory,
            budget=context.limits.tracking_budget,
            thresh=context.limits.thresh,
        )
        if new_rule is None:
            context.record_event(
                {
                    "phase": "orchestrator",
                    "status": "no_tracking_progress",
                    "reason": context.stuck_reason,
                }
            )
            record_trace(context, status="NO_TRACKING_PROGRESS", rules=current_rules, memory=memory)
            return context, "NO_TRACKING_PROGRESS"

        current_rules = insert_after_stuck_rule(current_rules, context.last_stuck_rule, new_rule)
        context.last_rule = new_rule.name
        context.last_stuck_rule = None
        context.record_event(
            {
                "phase": "orchestrator",
                "status": "rule_promoted",
                "rule": new_rule.name,
            }
        )
        context.metric_inc("tracking_promotions")

    record_trace(context, status="OUT_OF_BUDGET", rules=current_rules, memory=memory)
    return context, "OUT_OF_BUDGET"


__all__ = ["solve_with_rft", "insert_after_stuck_rule"]
