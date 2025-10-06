"""Tracking module for adaptive rule discovery."""
# [S:ALG v2] algo=tracking_search complexity=O(budget*eval_cost) pass

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Context, Hypothesis, Rule

logger = logging.getLogger("puma.rft.tracking")
logger.addHandler(logging.NullHandler())


@dataclass
class HypothesisMemory:
    """Memory of evaluated hypotheses with LRU eviction."""

    cap: int = 1000
    positives: "OrderedDict[str, Hypothesis]" = field(default_factory=OrderedDict)
    negatives: "OrderedDict[str, Hypothesis]" = field(default_factory=OrderedDict)

    def remember_positive(self, hyp: Hypothesis) -> None:
        signature = hyp.signature()
        self.positives[signature] = hyp
        self.positives.move_to_end(signature)
        while len(self.positives) > self.cap:
            evicted_sig, _ = self.positives.popitem(last=False)
            logger.debug({"phase": "tracking", "event": "evict_positive", "signature": evicted_sig})

    def remember_negative(self, hyp: Hypothesis) -> None:
        signature = hyp.signature()
        self.negatives[signature] = hyp
        self.negatives.move_to_end(signature)
        while len(self.negatives) > self.cap:
            evicted_sig, _ = self.negatives.popitem(last=False)
            logger.debug({"phase": "tracking", "event": "evict_negative", "signature": evicted_sig})

    def is_negative(self, hyp: Hypothesis) -> bool:
        return hyp.signature() in self.negatives

    def is_positive(self, hyp: Hypothesis) -> bool:
        return hyp.signature() in self.positives

    def promoted_rules(self) -> List[Rule]:
        promoted: List[Rule] = []
        for memory_rule in self.positives.values():
            rule = getattr(memory_rule, "promoted_rule", None)
            if isinstance(rule, Rule):
                promoted.append(rule)
        return promoted


def _dedupe_hypotheses(candidates: Iterable[Hypothesis], memory: HypothesisMemory) -> List[Hypothesis]:
    uniq: Dict[str, Hypothesis] = {}
    for hyp in candidates:
        sig = hyp.signature()
        if sig in uniq:
            continue
        if memory.is_positive(hyp) or memory.is_negative(hyp):
            continue
        uniq[sig] = hyp
    return list(uniq.values())


def generate_variations(context: Context, base_rules: Sequence[Rule], memory: HypothesisMemory) -> List[Hypothesis]:
    """Generate hypothesis variations by delegating to the state when available."""

    candidates: List[Hypothesis] = []
    if hasattr(context.state, "candidate_hypotheses"):
        state_candidates = context.state.candidate_hypotheses(context, base_rules, memory)
        candidates.extend(state_candidates)
    else:
        logger.debug({"phase": "tracking", "event": "no_candidate_hook"})
    return _dedupe_hypotheses(candidates, memory)


def evaluate_hypothesis(hypothesis: Hypothesis, context: Context) -> Tuple[float, Dict[str, object]]:
    """Evaluate a hypothesis using an explicit progress/simplicity score.

    The score adheres to the repository spec:

    ``progress = satisfied_targets / total_targets`` (reported by the state)
    ``simplicity = len(hypothesis.params.get("conditions", []))``
    ``score = progress - 0.1 * simplicity``

    The constant weight (0.1) is intentionally fixed for reproducibility and
    discourages overly complex promoted rules.
    """

    if not hasattr(context.state, "evaluate_hypothesis"):
        logger.debug({"phase": "tracking", "event": "no_eval_hook"})
        return -1.0, {"reason": "no_eval_hook"}

    trace = context.state.evaluate_hypothesis(hypothesis, context)
    progress = trace.get("progress", 0.0)
    simplicity = len(hypothesis.params.get("conditions", []))
    score = progress - 0.1 * simplicity
    trace.update({"simplicity": simplicity, "score": score})
    logger.debug({
        "phase": "tracking",
        "event": "hypothesis_evaluated",
        "hypothesis": hypothesis.description,
        "score": score,
    })
    return score, trace


def promote_to_rule(hypothesis: Hypothesis) -> Rule:
    """Promote a hypothesis to an executable rule."""

    promote_hook = hypothesis.params.get("promote_hook")
    if callable(promote_hook):
        rule = promote_hook(hypothesis)
    elif hasattr(hypothesis, "promote_hook") and callable(getattr(hypothesis, "promote_hook")):
        rule = hypothesis.promote_hook(hypothesis)
    else:
        raise ValueError(f"Hypothesis lacks promote hook: {hypothesis.description}")
    setattr(hypothesis, "promoted_rule", rule)
    return rule


def tracking_step(
    context: Context,
    base_rules: Sequence[Rule],
    memory: HypothesisMemory,
    budget: int,
    thresh: float,
) -> Optional[Rule]:
    """Run a single tracking step within the provided budget."""

    context.metric_inc("tracking_steps")
    candidates = generate_variations(context, base_rules, memory)
    if not candidates:
        logger.debug({"phase": "tracking", "event": "no_candidates"})
        context.metrics["tracking_negatives_size"] = len(memory.negatives)
        context.metrics["tracking_positive_cache"] = len(memory.positives)
        return None

    scored: List[Hypothesis] = []
    for idx, hypothesis in enumerate(candidates):
        if idx >= budget:
            break
        if memory.is_negative(hypothesis):
            continue
        score, trace = evaluate_hypothesis(hypothesis, context)
        hypothesis.score = score
        hypothesis.trace = trace
        if score > 0:
            hypothesis.evidence["pos"].append(trace)
        else:
            hypothesis.evidence["neg"].append(trace)
        scored.append(hypothesis)
        context.metric_inc("tracking_hypotheses_evaluated")
        logger.debug({
            "phase": "tracking",
            "hypothesis_id": hypothesis.signature(),
            "score": score,
        })

    if not scored:
        context.metrics["tracking_negatives_size"] = len(memory.negatives)
        context.metrics["tracking_positive_cache"] = len(memory.positives)
        return None

    scored.sort(key=lambda h: h.score, reverse=True)
    top_three = scored[:3]
    context.record_event(
        {
            "phase": "tracking",
            "status": "ranking",
            "top": [(h.description, h.score) for h in top_three],
        }
    )

    winner = scored[0]
    if winner.score > thresh:
        new_rule = promote_to_rule(winner)
        memory.remember_positive(winner)
        context.metrics["tracking_positive_cache"] = len(memory.positives)
        context.metrics["tracking_negatives_size"] = len(memory.negatives)
        logger.debug({
            "phase": "tracking",
            "event": "promoted",
            "hypothesis": winner.description,
            "score": winner.score,
        })
        context.record_event(
            {
                "phase": "tracking",
                "status": "promotion",
                "rule": new_rule.name,
                "score": winner.score,
            }
        )
        return new_rule

    for hyp in scored:
        if hyp.score <= 0:
            memory.remember_negative(hyp)
            logger.debug(
                {
                    "phase": "tracking",
                    "event": "negative_cache",
                    "hypothesis": hyp.description,
                    "score": hyp.score,
                }
            )
    context.metrics["tracking_negatives_size"] = len(memory.negatives)
    return None


__all__ = [
    "HypothesisMemory",
    "generate_variations",
    "evaluate_hypothesis",
    "promote_to_rule",
    "tracking_step",
]
