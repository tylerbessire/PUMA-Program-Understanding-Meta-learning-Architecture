"""Deterministic pliance interpreter for rule-following behavior."""
# [S:ALG v2] algo=pliance_inference complexity=O(steps*rules) pass

from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

from .types import Context, Rule

logger = logging.getLogger("puma.rft.pliance")
logger.addHandler(logging.NullHandler())


def _sorted_rules(rules: Sequence[Rule]) -> List[Rule]:
    return sorted(rules, key=lambda r: (r.priority, r.name))


def pliance_solve(
    context: Context,
    rules: Sequence[Rule],
    max_steps: int | None = None,
) -> Tuple[Context, str]:
    """Run deterministic pliance reasoning until completion or failure."""

    if max_steps is None:
        max_steps = context.limits.pliance_steps

    ordered = _sorted_rules(rules)
    for step in range(max_steps):
        context.metric_inc("pliance_iterations")
        if context.mark_state_seen():
            context.stuck_reason = "REPEAT_STATE"
            context.last_stuck_rule = context.last_rule
            loop_event = {
                "phase": "pliance",
                "step": step,
                "status": "loop",
                "reason": "REPEAT_STATE",
            }
            context.record_event(loop_event)
            logger.debug({"phase": "pliance", "step": step, "status": "loop"})
            return context, "PLIANCE_STUCK"

        fired_this_round = False
        if hasattr(context.state, "begin_iteration"):
            context.state.begin_iteration(step)

        for rule in ordered:
            context.metric_inc("pliance_rules_checked")
            should_fire = rule.applies(context)
            logger.debug(
                {
                    "phase": "pliance",
                    "step": step,
                    "rule": rule.name,
                    "status": "check",
                    "should_fire": should_fire,
                }
            )
            if not should_fire:
                continue

            new_state = rule.apply(context)
            if new_state is not None:
                context.state = new_state
            context.last_rule = rule.name
            fired_this_round = True
            context.metric_inc("pliance_rule_fires")
            event = {
                "phase": "pliance",
                "step": step,
                "rule": rule.name,
                "status": "fired",
            }
            context.record_event(event)
            logger.debug(event)
            if context.goal_test(context.state):
                context.stuck_reason = None
                context.last_stuck_rule = None
                completion = {
                    "phase": "pliance",
                    "step": step,
                    "rule": rule.name,
                    "status": "goal",
                }
                context.record_event(completion)
                return context, "GOAL"
            if context.stuck_reason == "FIXPOINT" and not context.goal_test(context.state):
                fixpoint_event = {
                    "phase": "pliance",
                    "step": step,
                    "rule": rule.name,
                    "status": "fixpoint",
                }
                context.record_event(fixpoint_event)
                logger.debug(fixpoint_event)
                return context, "PLIANCE_STUCK"

        if hasattr(context.state, "end_iteration"):
            context.state.end_iteration(step, fired_this_round)

        if not fired_this_round:
            context.stuck_reason = context.stuck_reason or "NO_RULE_APPLIED"
            context.last_stuck_rule = context.last_rule
            stuck_event = {
                "phase": "pliance",
                "step": step,
                "status": "stuck",
                "reason": context.stuck_reason,
            }
            context.record_event(stuck_event)
            logger.debug(stuck_event)
            return context, "PLIANCE_STUCK"

    context.stuck_reason = "STEP_LIMIT"
    context.last_stuck_rule = context.last_rule
    limit_event = {
        "phase": "pliance",
        "step": max_steps,
        "status": "limit",
        "reason": "STEP_LIMIT",
    }
    context.record_event(limit_event)
    logger.debug(limit_event)
    return context, "STEP_LIMIT"


__all__ = ["pliance_solve"]
