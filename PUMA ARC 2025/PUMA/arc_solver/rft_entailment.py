"""
Mutual and Combinatorial Entailment for RFT Rules.

Implements automatic rule inference based on relational properties:
- Mutual entailment: if A > B, then B < A
- Combinatorial entailment: if A > B and B > C, then A > C
- Frame coordination: maintaining consistency across derived relations
"""

from __future__ import annotations

from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


@dataclass
class RelationalRule:
    """A relational rule with entailment properties."""
    rule_id: str
    relation_type: str  # 'greater_than', 'smaller_than', 'left_of', 'right_of', etc.
    subject_id: str
    object_id: str
    confidence: float = 1.0
    derived: bool = False  # Whether this was inferred via entailment
    source_rules: List[str] = field(default_factory=list)  # Rules used to derive this


class EntailmentEngine:
    """Engine for deriving rules via mutual and combinatorial entailment."""

    def __init__(self):
        """Initialize entailment engine."""
        self.rules: Dict[str, RelationalRule] = {}
        self._next_rule_id = 0

        # Define mutual entailments (bidirectional opposites)
        self.mutual_relations = {
            'greater_than': 'smaller_than',
            'smaller_than': 'greater_than',
            'left_of': 'right_of',
            'right_of': 'left_of',
            'above': 'below',
            'below': 'above',
            'inside': 'contains',
            'contains': 'inside',
            'larger_than': 'smaller_than',
            'wider_than': 'narrower_than',
            'narrower_than': 'wider_than',
            'taller_than': 'shorter_than',
            'shorter_than': 'taller_than'
        }

        # Define transitive relations (support combinatorial entailment)
        self.transitive_relations = {
            'greater_than', 'smaller_than',
            'left_of', 'right_of',
            'above', 'below',
            'larger_than',
            'wider_than', 'narrower_than',
            'taller_than', 'shorter_than'
        }

    def add_rule(
        self,
        relation_type: str,
        subject_id: str,
        object_id: str,
        confidence: float = 1.0
    ) -> RelationalRule:
        """Add a base rule and derive entailments.

        Args:
            relation_type: Type of relation
            subject_id: Subject object ID
            object_id: Object object ID
            confidence: Rule confidence

        Returns:
            The added rule
        """
        # Create base rule
        rule = RelationalRule(
            rule_id=f"rule_{self._next_rule_id}",
            relation_type=relation_type,
            subject_id=subject_id,
            object_id=object_id,
            confidence=confidence,
            derived=False
        )
        self._next_rule_id += 1

        self.rules[rule.rule_id] = rule

        # Apply mutual entailment
        self._derive_mutual_entailment(rule)

        # Apply combinatorial entailment
        self._derive_combinatorial_entailment(rule)

        return rule

    def _derive_mutual_entailment(self, base_rule: RelationalRule) -> None:
        """Derive mutual entailment from a base rule.

        Example: if A > B, then B < A

        Args:
            base_rule: Base rule to derive from
        """
        opposite_relation = self.mutual_relations.get(base_rule.relation_type)

        if not opposite_relation:
            return  # No mutual relation defined

        # Check if opposite already exists
        existing = self._find_rule(
            opposite_relation,
            base_rule.object_id,
            base_rule.subject_id
        )

        if existing:
            # Already exists, update confidence if needed
            if base_rule.confidence > existing.confidence:
                existing.confidence = base_rule.confidence
            return

        # Create mutual entailment
        derived_rule = RelationalRule(
            rule_id=f"rule_{self._next_rule_id}",
            relation_type=opposite_relation,
            subject_id=base_rule.object_id,
            object_id=base_rule.subject_id,
            confidence=base_rule.confidence,
            derived=True,
            source_rules=[base_rule.rule_id]
        )
        self._next_rule_id += 1

        self.rules[derived_rule.rule_id] = derived_rule

    def _derive_combinatorial_entailment(self, new_rule: RelationalRule) -> None:
        """Derive combinatorial entailments from a new rule.

        Example: if A > B and B > C, then A > C

        Args:
            new_rule: Newly added rule
        """
        if new_rule.relation_type not in self.transitive_relations:
            return  # Not a transitive relation

        # Find chains where new_rule can be combined

        # Type 1: new_rule.subject -> new_rule.object, find object -> ?
        for existing_rule in list(self.rules.values()):
            if (existing_rule.relation_type == new_rule.relation_type and
                existing_rule.subject_id == new_rule.object_id):

                # Chain: new_rule.subject -> new_rule.object -> existing_rule.object
                # Derive: new_rule.subject -> existing_rule.object

                # Check if already exists
                if not self._find_rule(
                    new_rule.relation_type,
                    new_rule.subject_id,
                    existing_rule.object_id
                ):
                    # Create derived rule
                    combined_confidence = min(new_rule.confidence, existing_rule.confidence)

                    derived_rule = RelationalRule(
                        rule_id=f"rule_{self._next_rule_id}",
                        relation_type=new_rule.relation_type,
                        subject_id=new_rule.subject_id,
                        object_id=existing_rule.object_id,
                        confidence=combined_confidence * 0.9,  # Slight reduction for derived
                        derived=True,
                        source_rules=[new_rule.rule_id, existing_rule.rule_id]
                    )
                    self._next_rule_id += 1

                    self.rules[derived_rule.rule_id] = derived_rule

                    # Recursively derive more entailments
                    self._derive_mutual_entailment(derived_rule)

        # Type 2: ? -> new_rule.subject, find new_rule.object -> ?
        for existing_rule in list(self.rules.values()):
            if (existing_rule.relation_type == new_rule.relation_type and
                existing_rule.object_id == new_rule.subject_id):

                # Chain: existing_rule.subject -> new_rule.subject -> new_rule.object
                # Derive: existing_rule.subject -> new_rule.object

                # Check if already exists
                if not self._find_rule(
                    new_rule.relation_type,
                    existing_rule.subject_id,
                    new_rule.object_id
                ):
                    # Create derived rule
                    combined_confidence = min(new_rule.confidence, existing_rule.confidence)

                    derived_rule = RelationalRule(
                        rule_id=f"rule_{self._next_rule_id}",
                        relation_type=new_rule.relation_type,
                        subject_id=existing_rule.subject_id,
                        object_id=new_rule.object_id,
                        confidence=combined_confidence * 0.9,
                        derived=True,
                        source_rules=[existing_rule.rule_id, new_rule.rule_id]
                    )
                    self._next_rule_id += 1

                    self.rules[derived_rule.rule_id] = derived_rule

                    # Recursively derive more entailments
                    self._derive_mutual_entailment(derived_rule)

    def _find_rule(
        self,
        relation_type: str,
        subject_id: str,
        object_id: str
    ) -> Optional[RelationalRule]:
        """Find existing rule matching criteria.

        Args:
            relation_type: Relation type
            subject_id: Subject ID
            object_id: Object ID

        Returns:
            Matching rule or None
        """
        for rule in self.rules.values():
            if (rule.relation_type == relation_type and
                rule.subject_id == subject_id and
                rule.object_id == object_id):
                return rule

        return None

    def query_relation(
        self,
        relation_type: str,
        subject_id: str,
        object_id: str
    ) -> Optional[RelationalRule]:
        """Query for a specific relation.

        Args:
            relation_type: Relation type
            subject_id: Subject ID
            object_id: Object ID

        Returns:
            Rule if exists, None otherwise
        """
        return self._find_rule(relation_type, subject_id, object_id)

    def get_all_relations_for_object(self, object_id: str) -> List[RelationalRule]:
        """Get all relations involving an object.

        Args:
            object_id: Object ID

        Returns:
            List of rules involving the object
        """
        return [
            rule for rule in self.rules.values()
            if rule.subject_id == object_id or rule.object_id == object_id
        ]

    def verify_consistency(self) -> Dict[str, Any]:
        """Verify consistency of derived rules.

        Returns:
            Consistency report
        """
        issues = []
        warnings = []

        # Check for contradictions
        for rule in self.rules.values():
            # Check if opposite relation exists with different confidence
            opposite_relation = self.mutual_relations.get(rule.relation_type)

            if opposite_relation:
                opposite_rule = self._find_rule(
                    opposite_relation,
                    rule.object_id,
                    rule.subject_id
                )

                if opposite_rule:
                    # Should have same confidence (within tolerance)
                    if abs(rule.confidence - opposite_rule.confidence) > 0.1:
                        warnings.append({
                            'type': 'confidence_mismatch',
                            'rule1': rule.rule_id,
                            'rule2': opposite_rule.rule_id,
                            'diff': abs(rule.confidence - opposite_rule.confidence)
                        })

        # Check for circular dependencies in transitive relations
        for rule in self.rules.values():
            if rule.relation_type in self.transitive_relations:
                # Check if there's a cycle
                if self._has_cycle(rule.subject_id, rule.object_id, rule.relation_type):
                    issues.append({
                        'type': 'circular_dependency',
                        'rule': rule.rule_id,
                        'relation': rule.relation_type
                    })

        return {
            'consistent': len(issues) == 0,
            'num_rules': len(self.rules),
            'num_derived': sum(1 for r in self.rules.values() if r.derived),
            'issues': issues,
            'warnings': warnings
        }

    def _has_cycle(
        self,
        start_id: str,
        end_id: str,
        relation_type: str,
        visited: Optional[Set[str]] = None
    ) -> bool:
        """Check for circular dependencies.

        Args:
            start_id: Starting object ID
            end_id: Ending object ID
            relation_type: Relation type to follow
            visited: Set of visited IDs

        Returns:
            True if cycle detected
        """
        if visited is None:
            visited = {start_id}

        if start_id == end_id and len(visited) > 1:
            return True  # Found cycle

        # Find all rules where end_id is subject
        for rule in self.rules.values():
            if (rule.relation_type == relation_type and
                rule.subject_id == end_id and
                rule.object_id not in visited):

                new_visited = visited | {rule.object_id}

                if self._has_cycle(start_id, rule.object_id, relation_type, new_visited):
                    return True

        return False

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules in serializable format.

        Returns:
            List of rule dictionaries
        """
        return [
            {
                'rule_id': rule.rule_id,
                'relation_type': rule.relation_type,
                'subject_id': rule.subject_id,
                'object_id': rule.object_id,
                'confidence': rule.confidence,
                'derived': rule.derived,
                'source_rules': rule.source_rules
            }
            for rule in self.rules.values()
        ]

    def get_entailment_stats(self) -> Dict[str, Any]:
        """Get statistics about entailments.

        Returns:
            Statistics dictionary
        """
        base_rules = [r for r in self.rules.values() if not r.derived]
        derived_rules = [r for r in self.rules.values() if r.derived]

        mutual_derived = [
            r for r in derived_rules
            if len(r.source_rules) == 1  # Mutual entailment has 1 source
        ]

        combinatorial_derived = [
            r for r in derived_rules
            if len(r.source_rules) > 1  # Combinatorial has multiple sources
        ]

        return {
            'total_rules': len(self.rules),
            'base_rules': len(base_rules),
            'derived_rules': len(derived_rules),
            'mutual_entailments': len(mutual_derived),
            'combinatorial_entailments': len(combinatorial_derived),
            'avg_confidence': np.mean([r.confidence for r in self.rules.values()]),
            'avg_derived_confidence': np.mean([r.confidence for r in derived_rules]) if derived_rules else 0.0
        }


def integrate_entailment_with_pliance(pliance_engine, inventory) -> EntailmentEngine:
    """Integrate entailment engine with pliance rules.

    Args:
        pliance_engine: PlianceEngine instance
        inventory: ObjectInventory instance

    Returns:
        EntailmentEngine with rules from pliance
    """
    entailment = EntailmentEngine()

    # Extract relational facts from inventory
    schema = inventory.get_rule_friendly_schema()

    for obj_id, obj_data in schema.get('objects', {}).items():
        attrs = obj_data.get('attributes', {})

        # Add size-based relations
        size = attrs.get('size')
        if size:
            for other_id, other_data in schema.get('objects', {}).items():
                if other_id == obj_id:
                    continue

                other_size = other_data.get('attributes', {}).get('size')
                if other_size:
                    if size > other_size:
                        entailment.add_rule('larger_than', obj_id, other_id, confidence=1.0)
                    elif size < other_size:
                        entailment.add_rule('smaller_than', obj_id, other_id, confidence=1.0)

        # Add spatial relations from descriptors
        descriptors = obj_data.get('attributes', {})

        for other_id, other_data in schema.get('objects', {}).items():
            if other_id == obj_id:
                continue

            other_descriptors = other_data.get('attributes', {})

            # Position-based relations (simplified)
            # Would need actual position data for full implementation

    return entailment
