"""
Pliance Rule Engine for ARC tasks.

Defines a shared rule model (selectors, relations, actions) that supports
both automated rule generation and LLM-authored updates. Includes rule
application, violation logging, and repair routines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum

import numpy as np

from .grid import Array
from .object_reasoning import ARCObject, ObjectExtractor


class RuleConfidence(Enum):
    """Confidence levels for rules."""
    HIGH = 0.9
    MEDIUM = 0.7
    LOW = 0.5
    PROVISIONAL = 0.3


@dataclass
class ObjectSelector:
    """Selects objects based on attributes."""
    color: Optional[int] = None
    shape_type: Optional[str] = None
    size_min: Optional[int] = None
    size_max: Optional[int] = None
    tags: Set[str] = field(default_factory=set)
    position: Optional[str] = None  # 'border', 'interior', 'top', 'bottom', 'left', 'right'
    custom_predicate: Optional[str] = None  # For LLM-defined conditions

    def matches(self, obj: ARCObject) -> bool:
        """Check if object matches this selector."""
        if self.color is not None and obj.color != self.color:
            return False

        if self.shape_type is not None and obj.shape_type != self.shape_type:
            return False

        if self.size_min is not None and obj.size < self.size_min:
            return False

        if self.size_max is not None and obj.size > self.size_max:
            return False

        if self.tags:
            # Object must have all required tags (would need tag tracking)
            pass

        if self.position:
            if self.position == 'border' and not obj.descriptors.get('touches_border', False):
                return False
            elif self.position == 'interior' and obj.descriptors.get('touches_border', False):
                return False
            elif self.position == 'top' and obj.descriptors.get('distance_top', 100) > 0:
                return False
            elif self.position == 'bottom' and obj.descriptors.get('distance_bottom', 100) > 0:
                return False
            elif self.position == 'left' and obj.descriptors.get('distance_left', 100) > 0:
                return False
            elif self.position == 'right' and obj.descriptors.get('distance_right', 100) > 0:
                return False

        return True


@dataclass
class SpatialRelation:
    """Defines a spatial relation between objects."""
    relation_type: str  # 'left_of', 'right_of', 'above', 'below', 'inside', 'touching', 'aligned_h', 'aligned_v'
    target_selector: ObjectSelector
    distance_min: Optional[float] = None
    distance_max: Optional[float] = None


@dataclass
class RuleAction:
    """Action to apply when rule fires."""
    action_type: str  # 'recolor', 'move', 'copy', 'delete', 'replace', 'fill_region', 'extract'
    parameters: Dict[str, Any] = field(default_factory=dict)

    def apply(self, grid: Array, obj: ARCObject, context: Dict[str, Any]) -> Array:
        """Apply this action to the grid."""
        result = grid.copy()

        if self.action_type == 'recolor':
            new_color = self.parameters.get('color', 0)
            for pos in obj.positions:
                result[pos] = new_color

        elif self.action_type == 'move':
            dr = self.parameters.get('dr', 0)
            dc = self.parameters.get('dc', 0)

            # Clear original
            for pos in obj.positions:
                result[pos] = 0

            # Place at new position
            for pos in obj.positions:
                new_r = pos[0] + dr
                new_c = pos[1] + dc
                if 0 <= new_r < result.shape[0] and 0 <= new_c < result.shape[1]:
                    result[new_r, new_c] = obj.color

        elif self.action_type == 'delete':
            for pos in obj.positions:
                result[pos] = 0

        elif self.action_type == 'copy':
            # Copy to specified offset
            dr = self.parameters.get('dr', 0)
            dc = self.parameters.get('dc', 0)
            copies = self.parameters.get('copies', 1)

            for i in range(1, copies + 1):
                for pos in obj.positions:
                    new_r = pos[0] + i * dr
                    new_c = pos[1] + i * dc
                    if 0 <= new_r < result.shape[0] and 0 <= new_c < result.shape[1]:
                        result[new_r, new_c] = obj.color

        elif self.action_type == 'fill_region':
            # Fill bounding box with pattern
            bbox = obj.bounding_box
            fill_color = self.parameters.get('color', obj.color)
            for r in range(bbox[0], bbox[2] + 1):
                for c in range(bbox[1], bbox[3] + 1):
                    if 0 <= r < result.shape[0] and 0 <= c < result.shape[1]:
                        result[r, c] = fill_color

        return result


@dataclass
class PlianceRule:
    """A pliance rule: selector + relation + action."""
    rule_id: str
    name: str
    selector: ObjectSelector
    relations: List[SpatialRelation] = field(default_factory=list)
    action: Optional[RuleAction] = None
    confidence: float = 0.7
    provenance: str = "automated"  # 'automated', 'llm', 'manual', 'hybrid'
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def matches(self, obj: ARCObject, grid: Array, all_objects: List[ARCObject]) -> bool:
        """Check if this rule applies to the given object."""
        if not self.enabled:
            return False

        # Check selector
        if not self.selector.matches(obj):
            return False

        # Check relations
        for relation in self.relations:
            if not self._check_relation(obj, relation, all_objects):
                return False

        return True

    def _check_relation(
        self,
        obj: ARCObject,
        relation: SpatialRelation,
        all_objects: List[ARCObject]
    ) -> bool:
        """Check if spatial relation is satisfied."""
        obj_center = obj.center

        for other in all_objects:
            if other.id == obj.id:
                continue

            if not relation.target_selector.matches(other):
                continue

            other_center = other.center

            # Calculate distance
            distance = np.sqrt((obj_center[0] - other_center[0])**2 + (obj_center[1] - other_center[1])**2)

            # Check distance constraints
            if relation.distance_min is not None and distance < relation.distance_min:
                continue
            if relation.distance_max is not None and distance > relation.distance_max:
                continue

            # Check relation type
            if relation.relation_type == 'left_of':
                if obj_center[1] >= other_center[1]:
                    continue
            elif relation.relation_type == 'right_of':
                if obj_center[1] <= other_center[1]:
                    continue
            elif relation.relation_type == 'above':
                if obj_center[0] >= other_center[0]:
                    continue
            elif relation.relation_type == 'below':
                if obj_center[0] <= other_center[0]:
                    continue
            elif relation.relation_type == 'touching':
                # Check if any positions are adjacent
                touching = False
                for pos1 in obj.positions:
                    for pos2 in other.positions:
                        if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1:
                            touching = True
                            break
                    if touching:
                        break
                if not touching:
                    continue

            # If we get here, relation is satisfied
            return True

        # No matching object found for this relation
        return False

    def apply(self, grid: Array, obj: ARCObject, context: Dict[str, Any]) -> Array:
        """Apply this rule's action to the grid."""
        if self.action is None:
            return grid

        return self.action.apply(grid, obj, context)


@dataclass
class RuleViolation:
    """Records a rule application failure."""
    rule_id: str
    object_id: int
    violation_type: str  # 'failed_to_apply', 'incorrect_output', 'exception'
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    error_message: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleApplicationResult:
    """Result of applying rules to a grid."""
    output_grid: Array
    rules_applied: List[Tuple[str, int]]  # (rule_id, object_id)
    violations: List[RuleViolation]
    success_rate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlianceEngine:
    """Rule engine for applying and managing pliance rules."""

    def __init__(self):
        self.rules: Dict[str, PlianceRule] = {}
        self.extractor = ObjectExtractor()
        self.violation_log: List[RuleViolation] = []
        self._next_rule_id = 0

    def add_rule(self, rule: PlianceRule) -> None:
        """Add a rule to the engine."""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def emit_provisional_rule(
        self,
        name: str,
        selector: ObjectSelector,
        action: RuleAction,
        relations: List[SpatialRelation] = None,
        confidence: float = 0.5,
        provenance: str = "automated",
        metadata: Dict[str, Any] = None
    ) -> PlianceRule:
        """Emit a provisional rule from analysis."""
        rule = PlianceRule(
            rule_id=f"rule_{self._next_rule_id}",
            name=name,
            selector=selector,
            relations=relations or [],
            action=action,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata or {}
        )
        self._next_rule_id += 1
        self.add_rule(rule)
        return rule

    def apply_rules(
        self,
        grid: Array,
        rules_to_apply: Optional[List[str]] = None,
        context: Dict[str, Any] = None
    ) -> RuleApplicationResult:
        """Apply pliance rules to a grid."""
        context = context or {}
        result_grid = grid.copy()

        # Extract objects
        objects = self.extractor.extract_objects(grid)

        rules_applied = []
        violations = []

        # Determine which rules to apply
        if rules_to_apply is None:
            active_rules = [r for r in self.rules.values() if r.enabled]
        else:
            active_rules = [self.rules[rid] for rid in rules_to_apply if rid in self.rules]

        # Sort by confidence (apply high-confidence rules first)
        active_rules.sort(key=lambda r: r.confidence, reverse=True)

        # Apply rules
        for rule in active_rules:
            for obj in objects:
                try:
                    if rule.matches(obj, result_grid, objects):
                        # Apply rule
                        result_grid = rule.apply(result_grid, obj, context)
                        rules_applied.append((rule.rule_id, obj.id))

                except Exception as e:
                    # Log violation
                    violation = RuleViolation(
                        rule_id=rule.rule_id,
                        object_id=obj.id,
                        violation_type='exception',
                        error_message=str(e),
                        context={'rule': rule.name, 'object': obj.id}
                    )
                    violations.append(violation)
                    self.violation_log.append(violation)

        # Calculate success rate
        total_attempts = len(rules_applied) + len(violations)
        success_rate = len(rules_applied) / total_attempts if total_attempts > 0 else 0.0

        return RuleApplicationResult(
            output_grid=result_grid,
            rules_applied=rules_applied,
            violations=violations,
            success_rate=success_rate,
            metadata={'num_objects': len(objects), 'num_rules': len(active_rules)}
        )

    def validate_rules(
        self,
        train_pairs: List[Tuple[Array, Array]]
    ) -> Dict[str, Dict[str, Any]]:
        """Validate rules against training examples."""
        results = {}

        for rule_id, rule in self.rules.items():
            correct = 0
            total = 0
            violations = []

            for input_grid, expected_output in train_pairs:
                result = self.apply_rules(input_grid, rules_to_apply=[rule_id])

                total += 1
                if np.array_equal(result.output_grid, expected_output):
                    correct += 1
                else:
                    violation = RuleViolation(
                        rule_id=rule_id,
                        object_id=-1,
                        violation_type='incorrect_output',
                        expected=expected_output.tolist(),
                        actual=result.output_grid.tolist()
                    )
                    violations.append(violation)

            results[rule_id] = {
                'accuracy': correct / total if total > 0 else 0.0,
                'correct': correct,
                'total': total,
                'violations': violations
            }

        return results

    def repair_rule(
        self,
        rule_id: str,
        repair_action: Dict[str, Any]
    ) -> bool:
        """Repair a rule based on suggested action."""
        if rule_id not in self.rules:
            return False

        rule = self.rules[rule_id]
        action_type = repair_action.get('action')

        if action_type == 'adjust_confidence':
            new_confidence = repair_action.get('new_confidence', rule.confidence * 0.8)
            rule.confidence = max(0.1, min(1.0, new_confidence))

        elif action_type == 'disable':
            rule.enabled = False

        elif action_type == 'modify_selector':
            # Update selector attributes
            updates = repair_action.get('updates', {})
            for key, value in updates.items():
                if hasattr(rule.selector, key):
                    setattr(rule.selector, key, value)

        elif action_type == 'modify_action':
            # Update action parameters
            updates = repair_action.get('updates', {})
            rule.action.parameters.update(updates)

        elif action_type == 'add_relation':
            # Add new spatial relation
            relation_data = repair_action.get('relation')
            if relation_data:
                relation = SpatialRelation(**relation_data)
                rule.relations.append(relation)

        else:
            return False

        rule.metadata['last_repair'] = repair_action
        return True

    def get_rule_metrics(self) -> Dict[str, Any]:
        """Get metrics about rule performance."""
        total_rules = len(self.rules)
        enabled_rules = sum(1 for r in self.rules.values() if r.enabled)

        by_provenance = defaultdict(int)
        by_confidence = defaultdict(int)

        for rule in self.rules.values():
            by_provenance[rule.provenance] += 1

            if rule.confidence >= 0.8:
                by_confidence['high'] += 1
            elif rule.confidence >= 0.6:
                by_confidence['medium'] += 1
            else:
                by_confidence['low'] += 1

        return {
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'by_provenance': dict(by_provenance),
            'by_confidence': dict(by_confidence),
            'total_violations': len(self.violation_log)
        }

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export rules in serializable format."""
        return [
            {
                'rule_id': r.rule_id,
                'name': r.name,
                'selector': asdict(r.selector),
                'relations': [asdict(rel) for rel in r.relations],
                'action': asdict(r.action) if r.action else None,
                'confidence': r.confidence,
                'provenance': r.provenance,
                'metadata': r.metadata,
                'enabled': r.enabled
            }
            for r in self.rules.values()
        ]

    def import_rules(self, rules_data: List[Dict[str, Any]]) -> None:
        """Import rules from serialized format."""
        for rule_data in rules_data:
            selector = ObjectSelector(**rule_data['selector'])

            relations = []
            for rel_data in rule_data.get('relations', []):
                target_sel = ObjectSelector(**rel_data.pop('target_selector'))
                relations.append(SpatialRelation(target_selector=target_sel, **rel_data))

            action = None
            if rule_data.get('action'):
                action = RuleAction(**rule_data['action'])

            rule = PlianceRule(
                rule_id=rule_data['rule_id'],
                name=rule_data['name'],
                selector=selector,
                relations=relations,
                action=action,
                confidence=rule_data['confidence'],
                provenance=rule_data['provenance'],
                metadata=rule_data.get('metadata', {}),
                enabled=rule_data.get('enabled', True)
            )

            self.add_rule(rule)

            # Update ID counter
            rule_num = int(rule.rule_id.split('_')[1])
            self._next_rule_id = max(self._next_rule_id, rule_num + 1)

    def save(self, filepath: str) -> None:
        """Save rules to file."""
        data = {
            'rules': self.export_rules(),
            'violation_log': [asdict(v) for v in self.violation_log],
            'metrics': self.get_rule_metrics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load rules from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.import_rules(data.get('rules', []))

        # Reconstruct violation log
        self.violation_log = [
            RuleViolation(**v) for v in data.get('violation_log', [])
        ]
