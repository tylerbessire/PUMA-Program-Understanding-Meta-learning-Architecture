"""Typed primitives shared across the RFT modules."""
# [S:API v2] module=types contracts=stable pass

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple
import json
import logging
import random

logger = logging.getLogger("puma.rft.types")
logger.addHandler(logging.NullHandler())


@dataclass(frozen=True)
class Entity:
    """Structured representation of an item participating in relations."""

    id: str
    type: str
    features: Dict[str, Any]


@dataclass(frozen=True)
class Relation:
    """Relational predicate over entities."""

    name: str
    arity: int
    eval: Callable[[Sequence[Entity], "Context"], float | bool]


@dataclass
class Rule:
    """Production rule evaluated by the pliance interpreter."""

    name: str
    priority: int
    preconditions: List[Any]
    effects: List[Any]
    applies_fn: Callable[[Any, "Context"], bool]
    apply_fn: Callable[[Any, "Context"], Any]

    def applies(self, context: "Context") -> bool:
        """Return ``True`` when the rule should fire for the given context."""

        try:
            result = bool(self.applies_fn(context.state, context))
            return result
        except Exception as exc:  # pragma: no cover - surfaced via logging
            logger.exception("rule_applies_error", extra={"rule": self.name, "error": str(exc)})
            return False

    def apply(self, context: "Context") -> Any:
        """Apply effects and return the next state."""

        state = self.apply_fn(context.state, context)
        return state


@dataclass
class Limits:
    """Tunable bounds for the RFT control flow."""

    pliance_steps: int
    tracking_budget: int
    thresh: float
    outer_budget: int


TraceEvent = Dict[str, Any]


@dataclass
class Context:
    """State and execution metadata for pliance and tracking."""

    state: Any
    history: List[TraceEvent]
    constraints: Dict[str, Any]
    goal_test: Callable[[Any], bool]
    limits: Limits
    rng: random.Random = field(default_factory=random.Random)
    metrics: MutableMapping[str, int] = field(default_factory=dict)
    stuck_reason: Optional[str] = None
    last_rule: Optional[str] = None
    last_stuck_rule: Optional[str] = None
    state_hash_fn: Optional[Callable[[Any], str]] = None
    _seen_signatures: set[str] = field(default_factory=set, init=False, repr=False)

    def record_event(self, event: TraceEvent) -> None:
        self.history.append(event)
        logger.debug("trace_event", extra={"event": event})

    def metric_inc(self, key: str, amount: int = 1) -> None:
        self.metrics[key] = self.metrics.get(key, 0) + amount

    def state_signature(self) -> str:
        """Return a stable hashable signature of the current state."""

        if self.state_hash_fn:
            return self.state_hash_fn(self.state)
        try:
            return json.dumps(self.state, sort_keys=True)
        except TypeError:
            return repr(self.state)

    def mark_state_seen(self) -> bool:
        """Record the current state signature and return ``True`` if seen before."""

        signature = self.state_signature()
        if signature in self._seen_signatures:
            return True
        self._seen_signatures.add(signature)
        return False

    def clone(self) -> "Context":
        """Create a shallow copy of the context with duplicated history and metrics."""

        cloned = Context(
            state=self.state,
            history=list(self.history),
            constraints=dict(self.constraints),
            goal_test=self.goal_test,
            limits=self.limits,
            rng=random.Random(self.rng.random()),
            metrics=dict(self.metrics),
            state_hash_fn=self.state_hash_fn,
        )
        cloned.stuck_reason = self.stuck_reason
        cloned.last_rule = self.last_rule
        cloned.last_stuck_rule = self.last_stuck_rule
        cloned._seen_signatures = set(self._seen_signatures)
        return cloned


@dataclass
class Hypothesis:
    """Candidate adaptive rule discovered during tracking."""

    description: str
    variation_of: str
    params: Dict[str, Any]
    score: float = 0.0
    evidence: Dict[str, List[Any]] = field(default_factory=lambda: {"pos": [], "neg": []})
    trace: Dict[str, Any] | None = None

    def signature(self) -> str:
        """Deterministic identity used in the hypothesis memory."""

        def _sanitize(value: Any) -> Any:
            if callable(value):
                return getattr(value, "__name__", repr(value))
            if isinstance(value, dict):
                return {k: _sanitize(v) for k, v in value.items()}
            if isinstance(value, list):
                sanitized = [_sanitize(v) for v in value]
                return tuple(sorted(sanitized, key=lambda item: repr(item)))
            if isinstance(value, tuple):
                sanitized = tuple(_sanitize(v) for v in value)
                return tuple(sorted(sanitized, key=lambda item: repr(item)))
            if isinstance(value, set):
                sanitized = [_sanitize(v) for v in value]
                return tuple(sorted(sanitized, key=lambda item: repr(item)))
            return value

        key = {
            "variation_of": self.variation_of,
            "params": _sanitize(self.params),
        }
        return json.dumps(key, sort_keys=True)


def ensure_iterable(obj: Any) -> Iterable[Any]:
    """Ensure ``obj`` is iterable, turning scalars into one-length tuples."""

    if isinstance(obj, (list, tuple, set)):
        return obj
    return (obj,)
