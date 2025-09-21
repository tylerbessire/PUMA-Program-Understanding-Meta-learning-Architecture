"""Adaptive tacting module for behavioural guidance.

This module implements a lightweight system that emits symbolic descriptors
("tacts") for a given ARC task and keeps reinforcement-weighted statistics on
how useful those descriptors were for selecting DSL operations.  The goal is to
provide a functional analogue of tacting behaviour from Skinnerian verbal
operants without introducing heavy learning infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set
from collections import defaultdict


Tact = str


@dataclass
class TactProfile:
    """Container for emitted tacts and cached feature references."""

    tokens: Set[Tact]
    features: Mapping[str, float]


class TactSystem:
    """Generate tact tokens and track their reinforcement statistics."""

    _SEED_HINTS: Dict[Tact, Sequence[str]] = {
        "shape_preserved": ("identity", "rotate", "flip", "translate"),
        "shape_changed": ("crop", "pad", "smart_crop_auto"),
        "likely_recolor": ("recolor",),
        "likely_rotation": ("rotate",),
        "likely_reflection": ("flip", "transpose"),
        "likely_translation": ("translate",),
        "likely_crop": ("crop", "extract_content_region", "smart_crop_auto"),
        "likely_pad": ("pad",),
        "palette_shift": ("recolor", "color_map"),
        "low_color_complexity": ("identity", "recolor"),
        "high_color_complexity": ("extract_distinct_regions", "find_color_region"),
        "objects_increase": ("tile", "pad"),
        "objects_decrease": ("crop", "extract_content_region"),
        "size_shrink": ("crop", "smart_crop_auto"),
        "size_grow": ("pad", "tile"),
    }

    def __init__(self, decay: float = 0.02) -> None:
        self._token_operation_stats: Dict[Tact, MutableMapping[str, Dict[str, float]]] = {}
        self._decay = max(0.0, min(1.0, decay))

    # ------------------------------------------------------------------
    # Tact emission
    # ------------------------------------------------------------------
    def emit(self, features: Mapping[str, float]) -> TactProfile:
        """Return a :class:`TactProfile` for the provided feature mapping."""

        tokens: Set[Tact] = set()

        num_pairs = int(round(float(features.get("num_train_pairs", 0))))
        tokens.add(f"train_pairs:{num_pairs}")

        shape_preserved = bool(features.get("shape_preserved", False))
        tokens.add("shape_preserved" if shape_preserved else "shape_changed")

        size_ratio = float(features.get("size_ratio_mean", 1.0))
        if size_ratio < 0.95:
            tokens.add("size_shrink")
        elif size_ratio > 1.05:
            tokens.add("size_grow")

        for key in ("likely_recolor", "likely_rotation", "likely_reflection",
                    "likely_translation", "likely_crop", "likely_pad"):
            if float(features.get(key, 0.0)) > 0.3:
                tokens.add(key)

        input_colors = float(features.get("input_colors_mean", 0.0))
        if input_colors <= 3:
            tokens.add("low_color_complexity")
        elif input_colors >= 6:
            tokens.add("high_color_complexity")

        if bool(features.get("has_color_mapping", False)):
            tokens.add("palette_shift")

        input_objs = float(features.get("input_objects_mean", 0.0))
        output_objs = float(features.get("output_objects_mean", 0.0))
        if output_objs > input_objs + 0.5:
            tokens.add("objects_increase")
        elif output_objs + 0.5 < input_objs:
            tokens.add("objects_decrease")

        return TactProfile(tokens=tokens, features=features)

    # ------------------------------------------------------------------
    # Guidance / reinforcement
    # ------------------------------------------------------------------
    def suggest_operations(self, tacts: Iterable[Tact], top_k: int = 6) -> List[str]:
        """Return high-confidence operations suggested by the given tacts."""

        scores: Dict[str, float] = defaultdict(float)

        for token in tacts:
            stats = self._token_operation_stats.get(token)
            if stats:
                for op_name, stat in stats.items():
                    scores[op_name] += float(stat.get("mean_reward", 0.0))

            for hinted_op in self._SEED_HINTS.get(token, ()):  # pragma: no cover - deterministic mapping
                scores[hinted_op] += 0.25

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [op for op, _ in ranked[:top_k]]

    def reinforce(self, tacts: Iterable[Tact], operations: Iterable[str], reward: float) -> None:
        """Update tact-to-operation statistics using the reinforcement signal."""

        reward = max(0.0, min(1.0, float(reward)))
        for token in tacts:
            op_stats = self._token_operation_stats.setdefault(token, {})
            for op in operations:
                if not op:
                    continue
                stats = op_stats.setdefault(op, {"count": 0.0, "mean_reward": 0.0})
                stats["count"] += 1.0
                stats["mean_reward"] += (reward - stats["mean_reward"]) / stats["count"]
                if self._decay and stats["count"] > 1.0:
                    stats["mean_reward"] *= (1.0 - self._decay)

    def known_tacts(self) -> Set[Tact]:
        """Return the set of all known tact tokens."""

        return set(self._token_operation_stats.keys())

