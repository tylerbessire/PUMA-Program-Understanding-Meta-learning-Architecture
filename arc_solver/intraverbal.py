"""Intraverbal chaining support for program synthesis sequences."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


class IntraverbalChainer:
    """Track operation bigrams and provide sequence scores."""

    _START = "__START__"
    _END = "__END__"

    def __init__(self, smoothing: float = 0.05) -> None:
        self._transitions: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._smoothing = max(0.0, min(1.0, smoothing))

    # ------------------------------------------------------------------
    def reinforce(self, program: Iterable[Tuple[str, Dict[str, int]]], reward: float) -> None:
        """Update transition statistics using the observed program and reward."""

        reward = max(0.0, min(1.0, float(reward)))
        prev = self._START
        for op_name, _ in program:
            self._update_transition(prev, op_name, reward)
            prev = op_name
        self._update_transition(prev, self._END, reward)

    def _update_transition(self, prev: str, nxt: str, reward: float) -> None:
        bucket = self._transitions.setdefault(prev, {})
        stats = bucket.setdefault(nxt, {"count": 0.0, "mean_reward": 0.0})
        stats["count"] += 1.0
        stats["mean_reward"] += (reward - stats["mean_reward"]) / stats["count"]
        if self._smoothing and stats["count"] > 1.0:
            stats["mean_reward"] *= (1.0 - self._smoothing)

    # ------------------------------------------------------------------
    def score_sequence(self, program: Iterable[Tuple[str, Dict[str, int]]]) -> float:
        """Return average transition quality for the given program."""

        prev = self._START
        total = 0.0
        steps = 0
        for op_name, _ in program:
            total += self._transition_score(prev, op_name)
            prev = op_name
            steps += 1
        total += self._transition_score(prev, self._END)
        steps += 1
        if steps == 0:
            return 0.0
        return total / float(steps)

    def _transition_score(self, prev: str, nxt: str) -> float:
        bucket = self._transitions.get(prev)
        if not bucket:
            return 0.0
        stats = bucket.get(nxt)
        if stats:
            return max(0.0, min(1.0, float(stats.get("mean_reward", 0.0))))
        # unseen edge: return small prior based on bucket average to encourage exploration
        mean = 0.0
        count = 0
        for values in bucket.values():
            mean += float(values.get("mean_reward", 0.0))
            count += 1
        return mean / count if count else 0.0

    # ------------------------------------------------------------------
    def suggested_next(self, previous: str, top_k: int = 3) -> List[str]:
        """Return next operations sorted by learned preference."""

        bucket = self._transitions.get(previous)
        if not bucket:
            return []
        ranked = sorted(
            ((op, stats.get("mean_reward", 0.0)) for op, stats in bucket.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        return [op for op, _ in ranked[:top_k] if op not in {self._END}]

