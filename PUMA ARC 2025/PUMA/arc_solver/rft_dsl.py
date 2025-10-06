"""Relational Frame Theory oriented DSL primitives and interpreter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

from .grid import Array, bg_color
from .object_reasoning import ARCObject, ObjectExtractor


class PrimitiveCategory(str, Enum):
    """High-level categories for DSL primitives."""

    PERCEPTION = "perception"
    RELATION = "relation"
    TRANSFORMATION = "transformation"
    CONTROL = "control"


@dataclass
class Primitive:
    """Metadata describing a DSL primitive."""

    name: str
    category: PrimitiveCategory
    handler: Callable[["DSLContext", Dict[str, Any]], Any]
    doc: str = ""
    outputs: Literal["objects", "facts", "grid", "value", "none"] = "value"


@dataclass
class Instruction:
    """Single DSL instruction."""

    primitive: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DSLContext:
    """State carried across DSL execution."""

    grid: Array
    objects: List[ARCObject] = field(default_factory=list)
    working_objects: List[ARCObject] = field(default_factory=list)
    facts: List[Tuple[str, Tuple[Any, ...]]] = field(default_factory=list)
    values: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "DSLContext":
        return DSLContext(
            grid=self.grid.copy(),
            objects=list(self.objects),
            working_objects=list(self.working_objects),
            facts=list(self.facts),
            values=dict(self.values),
        )


def _ensure_iterable(value: Any) -> Iterable:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, np.ndarray)):
        return value
    return [value]


def _matches_criteria(obj: ARCObject, criteria: Mapping[str, Any]) -> bool:
    for key, expected in criteria.items():
        attr_name = key if key != "shape" else "shape_type"
        if getattr(obj, attr_name) != expected:
            return False
    return True


def _object_property(obj: ARCObject, name: str) -> Any:
    if name == "color":
        return obj.color
    if name == "size":
        return obj.size
    if name == "shape":
        return obj.shape_type
    if name == "bbox":
        return obj.bounding_box
    return getattr(obj, name)


def _inverse_relation(name: str) -> Optional[str]:
    opposites = {
        "left_of": "right_of",
        "right_of": "left_of",
        "above": "below",
        "below": "above",
    }
    return opposites.get(name)


def _binary_relation(obj_a: ARCObject, obj_b: ARCObject, name: str) -> Optional[str]:
    if name == "touches":
        return "touches" if set(obj_a.positions) & set(obj_b.positions) else None
    if name == "same_color":
        return "same_color" if obj_a.color == obj_b.color else None
    if name == "left_of":
        return "left_of" if obj_a.bounding_box[1] < obj_b.bounding_box[1] else None
    if name == "right_of":
        return "right_of" if obj_a.bounding_box[1] > obj_b.bounding_box[1] else None
    if name == "above":
        return "above" if obj_a.bounding_box[0] < obj_b.bounding_box[0] else None
    if name == "below":
        return "below" if obj_a.bounding_box[0] > obj_b.bounding_box[0] else None
    return None


def _compare(left: Any, right: Any, op: str) -> bool:
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "gt":
        return left > right
    if op == "lt":
        return left < right
    return False


def _translate_objects(grid: Array, objects: Sequence[ARCObject], dy: int, dx: int, fill: int) -> Array:
    result = grid.copy()
    for obj in objects:
        for r, c in obj.positions:
            result[r, c] = fill
    for obj in objects:
        for r, c in obj.positions:
            nr, nc = r + dy, c + dx
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                result[nr, nc] = obj.color
    return result


def _apply_fill(objects: Sequence[ARCObject], grid: Array, fill: int) -> Array:
    result = np.full_like(grid, fill)
    for obj in objects:
        for r, c in obj.positions:
            if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                result[r, c] = obj.color
    return result


class DSLInterpreter:
    """Execute instructions built from primitives."""

    def __init__(self) -> None:
        self.primitives: Dict[str, Primitive] = {}
        self._register_primitives()

    def execute(self, instructions: Sequence[Instruction], ctx: DSLContext) -> DSLContext:
        for inst in instructions:
            primitive = self.primitives.get(inst.primitive)
            if primitive is None:
                raise ValueError(f"Unknown primitive: {inst.primitive}")
            primitive.handler(ctx, inst.arguments)
        return ctx

    def _register_primitives(self) -> None:
        self._register(
            Primitive(
                name="find_objects",
                category=PrimitiveCategory.PERCEPTION,
                handler=self._handle_find_objects,
                doc="Extract connected components as objects.",
                outputs="objects",
            )
        )
        self._register(
            Primitive(
                name="filter_objects",
                category=PrimitiveCategory.PERCEPTION,
                handler=self._handle_filter_objects,
                doc="Filter objects by attribute criteria.",
                outputs="objects",
            )
        )
        self._register(
            Primitive(
                name="get_property",
                category=PrimitiveCategory.PERCEPTION,
                handler=self._handle_get_property,
                doc="Store object property values in context.",
                outputs="value",
            )
        )
        self._register(
            Primitive(
                name="get_relation",
                category=PrimitiveCategory.RELATION,
                handler=self._handle_get_relation,
                doc="Derive pairwise relations between objects.",
                outputs="facts",
            )
        )
        self._register(
            Primitive(
                name="assert_fact",
                category=PrimitiveCategory.RELATION,
                handler=self._handle_assert_fact,
                doc="Assert custom fact into context.",
                outputs="none",
            )
        )
        self._register(
            Primitive(
                name="compare",
                category=PrimitiveCategory.RELATION,
                handler=self._handle_compare,
                doc="Compare context values and store results.",
                outputs="value",
            )
        )
        self._register(
            Primitive(
                name="move",
                category=PrimitiveCategory.TRANSFORMATION,
                handler=self._handle_move,
                doc="Translate selected objects.",
                outputs="grid",
            )
        )
        self._register(
            Primitive(
                name="retain_selection",
                category=PrimitiveCategory.TRANSFORMATION,
                handler=self._handle_retain_selection,
                doc="Mask grid to retain selected objects.",
                outputs="grid",
            )
        )

    def _register(self, primitive: Primitive) -> None:
        self.primitives[primitive.name] = primitive

    def _handle_find_objects(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> List[ARCObject]:
        ignore_color = kwargs.get("ignore_color", bg_color(ctx.grid))
        extractor = ObjectExtractor()
        objects = extractor.extract_objects(ctx.grid, ignore_color=ignore_color)
        ctx.objects = objects
        ctx.working_objects = objects
        return objects

    def _handle_filter_objects(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> List[ARCObject]:
        criteria = kwargs.get("criteria", {})
        candidates = kwargs.get("objects", ctx.working_objects or ctx.objects)
        selection = [obj for obj in candidates if _matches_criteria(obj, criteria)]
        ctx.working_objects = selection
        return selection

    def _handle_get_property(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> List[Any]:
        property_name = kwargs["property"]
        objects = kwargs.get("objects", ctx.working_objects or ctx.objects)
        values = [_object_property(obj, property_name) for obj in objects]
        ctx.values[property_name] = values
        return values

    def _handle_get_relation(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> List[Tuple[int, int, str]]:
        rel_type = kwargs["relation"]
        objects = kwargs.get("objects", ctx.working_objects or ctx.objects)
        facts: List[Tuple[int, int, str]] = []
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i + 1 :]:
                relation = _binary_relation(obj_a, obj_b, rel_type)
                if relation:
                    facts.append((obj_a.id, obj_b.id, relation))
                    ctx.facts.append((rel_type, (obj_a.id, obj_b.id)))
                    inverse = _inverse_relation(rel_type)
                    if inverse:
                        ctx.facts.append((inverse, (obj_b.id, obj_a.id)))
        return facts

    def _handle_assert_fact(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> None:
        fact_name = kwargs["name"]
        arguments = tuple(kwargs.get("args", []))
        ctx.facts.append((fact_name, arguments))

    def _handle_compare(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> List[bool]:
        left = kwargs["left"]
        right = kwargs["right"]
        op = kwargs.get("op", "eq")
        alias = kwargs.get("alias", op)
        comparisons = [_compare(l_val, r_val, op) for l_val, r_val in zip(_ensure_iterable(left), _ensure_iterable(right))]
        ctx.values[alias] = comparisons
        return comparisons

    def _handle_move(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> Array:
        dy = kwargs.get("dy", 0)
        dx = kwargs.get("dx", 0)
        fill = kwargs.get("fill", bg_color(ctx.grid))
        objects = kwargs.get("objects", ctx.working_objects or ctx.objects)
        objects = list(objects) if objects else []
        ctx.grid = _translate_objects(ctx.grid, objects, dy, dx, fill)
        return ctx.grid

    def _handle_retain_selection(self, ctx: DSLContext, kwargs: Dict[str, Any]) -> Array:
        fill = kwargs.get("fill", bg_color(ctx.grid))
        objects = kwargs.get("objects", ctx.working_objects or ctx.objects)
        objects = list(objects) if objects else []
        ctx.grid = _apply_fill(objects, ctx.grid, fill)
        return ctx.grid


__all__ = [
    "DSLContext",
    "DSLInterpreter",
    "Instruction",
    "Primitive",
    "PrimitiveCategory",
]
