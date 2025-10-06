"""Relational decomposition utilities for ARC grids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .grid import Array
from .object_reasoning import ARCObject, ObjectExtractor


BinaryArgs = Tuple[int, int]


@dataclass(frozen=True)
class RelationalFact:
    """Represents a symbolic fact derived from a grid."""

    predicate: str
    args: Tuple[int, ...]
    attributes: Dict[str, object] = field(default_factory=dict)

    def key(self) -> Tuple[str, Tuple[int, ...]]:
        return (self.predicate, self.args)


class FactSet:
    """Utility wrapper that deduplicates facts while preserving order."""

    def __init__(self) -> None:
        self._facts: List[RelationalFact] = []
        self._keys: Set[Tuple[str, Tuple[int, ...]]] = set()

    def add(self, fact: RelationalFact) -> None:
        key = fact.key()
        if key not in self._keys:
            self._facts.append(fact)
            self._keys.add(key)

    def extend(self, facts: Iterable[RelationalFact]) -> None:
        for fact in facts:
            self.add(fact)

    def __iter__(self):
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    def to_list(self) -> List[RelationalFact]:
        return list(self._facts)


class RelationalDecomposer:
    """Convert grids into relational fact sets with RFT entailment."""

    def __init__(self, connectivity: str = "n4", ignore_color: Optional[int] = 0):
        self.connectivity = connectivity
        self.ignore_color = ignore_color
        self.extractor = ObjectExtractor()

    def decompose(self, grid: Array) -> Tuple[List[ARCObject], List[RelationalFact]]:
        objects = self.extractor.extract_objects(grid, ignore_color=self.ignore_color or 0)
        facts = FactSet()

        facts.extend(self._emit_object_facts(objects))
        facts.extend(self._emit_shape_facts(objects))
        facts.extend(self._emit_pairwise_relations(objects))

        self._apply_mutual_entailment(facts)
        self._apply_transitive_entailment(facts)

        return objects, facts.to_list()

    # ------------------------------------------------------------------
    # Fact emitters
    # ------------------------------------------------------------------

    def _emit_object_facts(self, objects: Sequence[ARCObject]) -> Iterable[RelationalFact]:
        for obj in objects:
            attributes = {
                "color": obj.color,
                "size": obj.size,
                "shape": obj.shape_type,
                "bbox": obj.bounding_box,
            }
            yield RelationalFact("object", (obj.id,), attributes)

    def _emit_shape_facts(self, objects: Sequence[ARCObject]) -> Iterable[RelationalFact]:
        for obj in objects:
            yield RelationalFact("center", (obj.id,), {"value": obj.center})
            yield RelationalFact("width", (obj.id,), {"value": obj.width})
            yield RelationalFact("height", (obj.id,), {"value": obj.height})

    def _emit_pairwise_relations(self, objects: Sequence[ARCObject]) -> Iterable[RelationalFact]:
        relations: List[RelationalFact] = []
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1 :]:
                for pred in ("left_of", "right_of", "above", "below", "touching", "overlaps"):
                    if self._holds_relation(obj_a, obj_b, pred):
                        relations.append(RelationalFact(pred, (obj_a.id, obj_b.id)))
        return relations

    # ------------------------------------------------------------------
    # RFT entailment rules
    # ------------------------------------------------------------------

    _MUTUAL_INVERSES: Dict[str, str] = {
        "left_of": "right_of",
        "right_of": "left_of",
        "above": "below",
        "below": "above",
    }

    _TRANSITIVE_RELATIONS: Set[str] = {"left_of", "right_of", "above", "below"}

    def _apply_mutual_entailment(self, facts: FactSet) -> None:
        to_add: List[RelationalFact] = []
        for fact in facts:
            if fact.predicate in self._MUTUAL_INVERSES and len(fact.args) >= 2:
                inverse_pred = self._MUTUAL_INVERSES[fact.predicate]
                inverse_args = (fact.args[1], fact.args[0]) + fact.args[2:]
                to_add.append(
                    RelationalFact(
                        inverse_pred,
                        inverse_args,
                        {**fact.attributes, "derived": "mutual"},
                    )
                )
        facts.extend(to_add)

    def _apply_transitive_entailment(self, facts: FactSet) -> None:
        adjacency: Dict[str, Dict[int, Set[int]]] = {
            rel: {} for rel in self._TRANSITIVE_RELATIONS
        }
        for fact in facts:
            if fact.predicate in self._TRANSITIVE_RELATIONS and len(fact.args) >= 2:
                src, dst = fact.args[:2]
                adjacency.setdefault(fact.predicate, {}).setdefault(src, set()).add(dst)

        derived: List[RelationalFact] = []
        for predicate, graph in adjacency.items():
            for start in graph:
                visited = set()
                stack = [start]
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    for nxt in graph.get(node, set()):
                        if nxt not in visited:
                            stack.append(nxt)
                        if nxt != start:
                            derived.append(
                                RelationalFact(
                                    predicate,
                                    (start, nxt),
                                    {"derived": "transitive"},
                                )
                            )
        facts.extend(derived)

    # ------------------------------------------------------------------
    # Relation helpers
    # ------------------------------------------------------------------

    def _holds_relation(self, obj_a: ARCObject, obj_b: ARCObject, relation: str) -> bool:
        ax1, ay1, ax2, ay2 = obj_a.bounding_box
        bx1, by1, bx2, by2 = obj_b.bounding_box

        if relation == "left_of":
            return ay2 < by1
        if relation == "right_of":
            return by2 < ay1
        if relation == "above":
            return ax2 < bx1
        if relation == "below":
            return bx2 < ax1
        if relation == "overlaps":
            return not (ay2 < by1 or by2 < ay1 or ax2 < bx1 or bx2 < ax1)
        if relation == "touching":
            return self._manhattan_distance(obj_a, obj_b) == 1
        return False

    def _manhattan_distance(self, obj_a: ARCObject, obj_b: ARCObject) -> int:
        ax1, ay1, ax2, ay2 = obj_a.bounding_box
        bx1, by1, bx2, by2 = obj_b.bounding_box

        if ay2 < by1:
            horizontal = by1 - ay2
        elif by2 < ay1:
            horizontal = ay1 - by2
        else:
            horizontal = 0

        if ax2 < bx1:
            vertical = bx1 - ax2
        elif bx2 < ax1:
            vertical = ax1 - bx2
        else:
            vertical = 0

        return horizontal + vertical


__all__ = ["RelationalFact", "RelationalDecomposer"]
