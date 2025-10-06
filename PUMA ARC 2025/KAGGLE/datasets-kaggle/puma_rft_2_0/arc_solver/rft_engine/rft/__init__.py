"""Relational Frame Theory inspired solving primitives for PUMA."""
# [S:DESIGN v1] package=rft_core pass

from .feature_flags import RFT_ENABLED, RFT_DEFAULT_LIMITS, build_limits
from .types import Entity, Relation, Rule, Context, Hypothesis, Limits
from .pliance import pliance_solve
from .tracking import (
    generate_variations,
    evaluate_hypothesis,
    promote_to_rule,
    HypothesisMemory,
    tracking_step,
)
from .orchestrator import solve_with_rft, insert_after_stuck_rule
from .explain import explain_last_run, record_trace

__all__ = [
    "RFT_ENABLED",
    "RFT_DEFAULT_LIMITS",
    "build_limits",
    "Entity",
    "Relation",
    "Rule",
    "Context",
    "Hypothesis",
    "Limits",
    "pliance_solve",
    "generate_variations",
    "evaluate_hypothesis",
    "promote_to_rule",
    "HypothesisMemory",
    "tracking_step",
    "solve_with_rft",
    "insert_after_stuck_rule",
    "explain_last_run",
    "record_trace",
]
