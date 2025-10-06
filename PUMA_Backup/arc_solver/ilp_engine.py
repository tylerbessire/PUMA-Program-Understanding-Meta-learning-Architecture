"""Simple ILP-inspired program induction for ARC relational facts.

This module implements a lightweight attribute-based rule learner that operates
on ARC objects extracted from input/output grids. It provides a deterministic
hook for learning selection rules that can be expressed in the DSL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .grid import Array, bg_color
from .object_reasoning import ARCObject, ObjectExtractor
from .rft_dsl import DSLContext, DSLInterpreter, Instruction


@dataclass
class ILPExample:
    """Training example consisting of input and target output grids."""

    input_grid: Array
    output_grid: Array


@dataclass
class ILPProgram:
    """Learned rule expressed as attribute criteria."""

    criteria: Dict[str, object]
    ignore_color: int = 0

    def matches(self, obj: ARCObject) -> bool:
        for key, expected in self.criteria.items():
            attr_name = key if key != "shape" else "shape_type"
            if getattr(obj, attr_name) != expected:
                return False
        return True

    def apply_to_grid(self, grid: Array) -> Array:
        extractor = ObjectExtractor()
        objects = extractor.extract_objects(grid, ignore_color=self.ignore_color)
        selected = [obj for obj in objects if self.matches(obj)]
        result = np.full_like(grid, self.ignore_color)
        for obj in selected:
            for r, c in obj.positions:
                result[r, c] = obj.color
        return result

    def to_dsl_instructions(self, ignore_color: Optional[int] = None) -> List[Instruction]:
        ignore = self.ignore_color if ignore_color is None else ignore_color
        criteria = dict(self.criteria)
        return [
            Instruction("find_objects", {"ignore_color": ignore}),
            Instruction("filter_objects", {"criteria": criteria}),
            Instruction("retain_selection", {"fill": ignore}),
        ]

    def run_via_dsl(self, grid: Array) -> Array:
        ctx = DSLContext(grid=grid.copy())
        interpreter = DSLInterpreter()
        interpreter.execute(self.to_dsl_instructions(ignore_color=self.ignore_color), ctx)
        return ctx.grid


class SimpleAttributeILP:
    """Learn simple attribute-equality rules from ARC examples."""

    SUPPORTED_ATTRIBUTES = ("color", "shape_type")

    def __init__(self, extractor: Optional[ObjectExtractor] = None, ignore_color: Optional[int] = None) -> None:
        self.extractor = extractor or ObjectExtractor()
        self.ignore_color = bg_color(np.zeros((1, 1), dtype=np.int16)) if ignore_color is None else ignore_color

    def induce(self, examples: Sequence[ILPExample]) -> Optional[ILPProgram]:
        if not examples:
            return None

        attribute_values: Dict[str, Optional[object]] = {attr: None for attr in self.SUPPORTED_ATTRIBUTES}
        negative_values: Dict[str, Set[object]] = {attr: set() for attr in self.SUPPORTED_ATTRIBUTES}

        for example in examples:
            input_objs = self.extractor.extract_objects(example.input_grid, ignore_color=self.ignore_color)
            output_objs = self.extractor.extract_objects(example.output_grid, ignore_color=self.ignore_color)

            positives = self._match_output_to_input(input_objs, output_objs)
            if not positives:
                return None
            positive_ids = {obj.id for obj in positives}
            negatives = [obj for obj in input_objs if obj.id not in positive_ids]

            for attr in self.SUPPORTED_ATTRIBUTES:
                pos_values = {getattr(obj, attr) for obj in positives}
                if len(pos_values) == 1:
                    value = pos_values.pop()
                    if attribute_values[attr] is None:
                        attribute_values[attr] = value
                    elif attribute_values[attr] != value:
                        attribute_values[attr] = None
                else:
                    attribute_values[attr] = None
                negative_values[attr].update(getattr(obj, attr) for obj in negatives)

        criteria: Dict[str, object] = {}
        for attr, value in attribute_values.items():
            if value is None:
                continue
            if value in negative_values[attr]:
                continue
            key = "shape" if attr == "shape_type" else attr
            criteria[key] = value

        if not criteria:
            return None
        return ILPProgram(criteria, ignore_color=self.ignore_color)

    def _match_output_to_input(
        self, input_objs: Sequence[ARCObject], output_objs: Sequence[ARCObject]
    ) -> List[ARCObject]:
        matched: List[ARCObject] = []
        used: Set[int] = set()
        for out_obj in output_objs:
            candidate = self._find_matching_input_object(input_objs, out_obj, used)
            if candidate is not None:
                matched.append(candidate)
                used.add(candidate.id)
        return matched

    def _find_matching_input_object(
        self,
        input_objs: Sequence[ARCObject],
        out_obj: ARCObject,
        used_ids: Set[int],
    ) -> Optional[ARCObject]:
        for obj in input_objs:
            if obj.id in used_ids:
                continue
            if obj.color != out_obj.color:
                continue
            if obj.shape_type != out_obj.shape_type:
                continue
            if obj.size != out_obj.size:
                continue
            if obj.positions == out_obj.positions:
                return obj
        return None


__all__ = ["ILPExample", "ILPProgram", "SimpleAttributeILP"]
