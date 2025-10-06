"""Minimal grid harness for exercising RFT pliance and tracking."""
# [S:HARNESS v1] fixture=grid_demo pass

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..feature_flags import build_limits
from ..types import Context, Hypothesis, Limits, Rule

logger = logging.getLogger("puma.rft.demo.grid")
logger.addHandler(logging.NullHandler())

Coord = Tuple[int, int]
Grid = List[List[str]]


def _clone_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


@dataclass
class GridState:
    """2D grid with target cells that must become black."""

    grid: Grid
    targets: Set[Coord]
    seed_color: object = 0
    metadata: Dict[str, object] = field(default_factory=dict)
    fixpoint_triggered: bool = False

    def clone(self) -> "GridState":
        return GridState(
            grid=_clone_grid(self.grid),
            targets=set(self.targets),
            seed_color=self.seed_color,
            metadata=dict(self.metadata),
            fixpoint_triggered=self.fixpoint_triggered,
        )

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def begin_iteration(self, step: int) -> None:
        self.metadata["iteration"] = step
        self.metadata["changed"] = False

    def end_iteration(self, step: int, fired: bool) -> None:
        self.metadata["changed_last"] = self.metadata.get("changed", False)

    def mark_changed(self) -> None:
        self.metadata["changed"] = True

    def in_bounds(self, coord: Coord) -> bool:
        r, c = coord
        return 0 <= r < self.height and 0 <= c < self.width

    def get(self, coord: Coord) -> str:
        r, c = coord
        return self.grid[r][c]

    def set(self, coord: Coord, value: str) -> None:
        r, c = coord
        self.grid[r][c] = value
        self.mark_changed()

    def progress(self) -> float:
        if not self.targets:
            return 1.0
        satisfied = sum(1 for coord in self.targets if self.get(coord) == self.seed_color)
        return satisfied / len(self.targets)

    def unsatisfied_targets(self) -> List[Coord]:
        return [coord for coord in sorted(self.targets) if self.get(coord) != self.seed_color]

    def satisfied_sources(self) -> List[Coord]:
        coords: List[Coord] = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r][c] == self.seed_color:
                    coords.append((r, c))
        return coords

    def neighbor_offsets(self) -> List[Coord]:
        return [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    def candidate_hypotheses(
        self,
        context: Context,
        base_rules: Sequence[Rule],
        memory: object,
    ) -> List[Hypothesis]:
        candidates: List[Hypothesis] = []
        seen: Set[Tuple[Coord, int, Tuple[Tuple[str, str], ...]]] = set()
        for target in self.unsatisfied_targets():
            for dr, dc in self.neighbor_offsets():
                source = (target[0] - dr, target[1] - dc)
                if not self.in_bounds(source):
                    continue
                if self.get(source) != self.seed_color:
                    continue
                conditions: List[Dict[str, str]] = []
                target_color = self.get(target)
                if target_color != self.seed_color:
                    conditions.append({"target_color": target_color})
                cond_key = tuple(sorted(tuple(sorted(cond.items())) for cond in conditions))
                key = ((dr, dc), 1, cond_key)
                if key in seen:
                    continue
                seen.add(key)
                desc = f"propagate {self.seed_color} by ({dr},{dc})"
                params = {
                    "direction": (dr, dc),
                    "radius": 1,
                    "conditions": conditions,
                    "priority": 30,
                    "name": f"track_{dr}_{dc}_{len(conditions)}",
                    "promote_hook": promote_grid_rule,
                }
                candidates.append(Hypothesis(desc, variation_of="R1", params=params))
                if not conditions:
                    continue
                # Condition-free alternative to test scope variation.
                alt_key = ((dr, dc), 1, tuple())
                if alt_key not in seen:
                    alt_params = dict(params)
                    alt_params["conditions"] = []
                    alt_params["name"] = f"track_{dr}_{dc}_broad"
                    seen.add(alt_key)
                    candidates.append(
                        Hypothesis(
                            f"propagate {self.seed_color} by ({dr},{dc}) regardless of color",
                            variation_of="R1",
                            params=alt_params,
                        )
                    )
        # Radius-based variations for diagonals or gaps.
        for target in self.unsatisfied_targets():
            for dr, dc in self.neighbor_offsets():
                source = (target[0] - 2 * dr, target[1] - 2 * dc)
                if not self.in_bounds(source):
                    continue
                if self.get(source) != self.seed_color:
                    continue
                params = {
                    "direction": (dr, dc),
                    "radius": 2,
                    "conditions": [],
                    "priority": 40,
                    "name": f"track_radius2_{dr}_{dc}",
                    "promote_hook": promote_grid_rule,
                }
                key = ((dr, dc), 2, tuple())
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    Hypothesis(
                        f"propagate {self.seed_color} by ({dr},{dc}) with radius 2",
                        variation_of="R1",
                        params=params,
                    )
                )
        return candidates

    def _conditions_met(self, source: Coord, target: Coord, conditions: Sequence[Dict[str, str]]) -> bool:
        for cond in conditions:
            src_color = cond.get("source_color")
            if src_color is not None and self.get(source) != src_color:
                return False
            tgt_color = cond.get("target_color")
            if tgt_color is not None and self.get(target) != tgt_color:
                return False
        return True

    def propagate(
        self,
        direction: Coord,
        radius: int = 1,
        conditions: Sequence[Dict[str, str]] | None = None,
        mutate: bool = True,
    ) -> int:
        conditions = conditions or []
        changed = 0
        while True:
            to_paint: List[Coord] = []
            for r, c in self.satisfied_sources():
                target = (r + direction[0] * radius, c + direction[1] * radius)
                if not self.in_bounds(target):
                    continue
                if self.get(target) == self.seed_color:
                    continue
                if not self._conditions_met((r, c), target, conditions):
                    continue
                to_paint.append(target)
            if not to_paint:
                break
            changed += len(to_paint)
            for coord in to_paint:
                self.set(coord, self.seed_color)
            if not mutate:
                break
        return changed

    def evaluate_hypothesis(self, hypothesis: Hypothesis, context: Context) -> Dict[str, object]:
        clone = self.clone()
        direction = tuple(hypothesis.params.get("direction", (0, 1)))
        radius = int(hypothesis.params.get("radius", 1))
        conditions = hypothesis.params.get("conditions", [])
        changed = clone.propagate(direction, radius=radius, conditions=conditions, mutate=True)
        return {
            "progress": clone.progress(),
            "changed": changed,
            "direction": direction,
            "radius": radius,
            "conditions": conditions,
        }

    def apply_rule(self, direction: Coord, radius: int, conditions: Sequence[Dict[str, str]]) -> int:
        changed = self.propagate(direction, radius=radius, conditions=conditions, mutate=True)
        if changed == 0:
            self.fixpoint_triggered = True
        else:
            self.fixpoint_triggered = False
        return changed

    def state_signature(self) -> str:
        rows = ["|".join(str(cell) for cell in row) for row in self.grid]
        rows.append("|".join(f"{r}:{c}" for r, c in sorted(self.targets)))
        return "#".join(rows)


def promote_grid_rule(hypothesis: Hypothesis) -> Rule:
    direction = tuple(hypothesis.params["direction"])
    radius = int(hypothesis.params.get("radius", 1))
    conditions = list(hypothesis.params.get("conditions", []))
    priority = int(hypothesis.params.get("priority", 50))
    name = hypothesis.params.get("name", f"track_{direction[0]}_{direction[1]}")

    def applies_fn(state: GridState, context: Context) -> bool:
        if not isinstance(state, GridState):
            return False
        clone = state.clone()
        changed = clone.propagate(direction, radius=radius, conditions=conditions, mutate=True)
        return changed > 0

    def apply_fn(state: GridState, context: Context) -> GridState:
        state.apply_rule(direction, radius, conditions)
        return state

    return Rule(
        name=name,
        priority=priority,
        preconditions=[{"direction": direction, "radius": radius}],
        effects=[{"paint": hypothesis.params.get("conditions", [])}],
        applies_fn=applies_fn,
        apply_fn=apply_fn,
    )


def make_base_rules() -> List[Rule]:
    """Return the intentionally incomplete base rule-set."""

    def r1_applies(state: GridState, context: Context) -> bool:
        return isinstance(state, GridState) and state.clone().propagate((0, 1), mutate=True) > 0

    def r1_apply(state: GridState, context: Context) -> GridState:
        state.apply_rule((0, 1), 1, [])
        return state

    def r2_applies(state: GridState, context: Context) -> bool:
        if not isinstance(state, GridState):
            return False
        if context.goal_test(state):
            return False
        return not state.clone().propagate((0, 1), mutate=True) > 0 and not state.fixpoint_triggered

    def r2_apply(state: GridState, context: Context) -> GridState:
        state.fixpoint_triggered = True
        context.stuck_reason = "FIXPOINT"
        return state

    return [
        Rule("R1_horizontal_black", priority=10, preconditions=[], effects=[], applies_fn=r1_applies, apply_fn=r1_apply),
        Rule("R2_fixpoint", priority=90, preconditions=[], effects=[], applies_fn=r2_applies, apply_fn=r2_apply),
    ]


def goal_test(state: GridState) -> bool:
    return state.progress() >= 1.0


def build_context(grid: Grid, targets: Iterable[Coord], limits: Optional[Limits] = None) -> Tuple[Context, List[Rule]]:
    limits = limits or build_limits()
    seed_color: object = 0
    if grid and isinstance(grid[0][0], str):
        seed_color = "black"
    state = GridState(grid=_clone_grid(grid), targets=set(targets), seed_color=seed_color)
    context = Context(
        state=state,
        history=[],
        constraints={"type": "grid_demo"},
        goal_test=goal_test,
        limits=limits,
        state_hash_fn=lambda s: s.state_signature() if isinstance(s, GridState) else str(s),
    )
    return context, make_base_rules()


__all__ = ["GridState", "promote_grid_rule", "make_base_rules", "build_context", "goal_test"]
