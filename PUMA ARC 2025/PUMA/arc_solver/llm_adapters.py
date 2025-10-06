"""
LLM Adapters for PUMA ARC Solver.

Provides adapters to convert internal data structures (object inventories,
rule failures, primitive catalogs, tracking state) into LLM-friendly formats
for reasoning tasks.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict

import numpy as np

from .grid import Array
from .object_inventory import ObjectInventory, ObjectDelta, ObjectEntry
from .pliance_engine import PlianceRule, RuleViolation, PlianceEngine
from .rft_tracking import RFTTracker, RuleConflict, RuleRepairTask


class ObjectInventoryAdapter:
    """Adapter to convert ObjectInventory to LLM-friendly format."""

    @staticmethod
    def to_llm_context(inventory: ObjectInventory, max_objects: int = 20) -> Dict[str, Any]:
        """Convert inventory to concise LLM context.

        Args:
            inventory: Object inventory to convert
            max_objects: Maximum objects to include (most important first)

        Returns:
            LLM-friendly dictionary with object summaries
        """
        schema = inventory.get_rule_friendly_schema()

        # Prioritize objects by occurrence count
        objects_with_priority = []
        for obj_id, obj_data in schema.get('objects', {}).items():
            priority = len(obj_data.get('occurrences', []))
            objects_with_priority.append((priority, obj_id, obj_data))

        objects_with_priority.sort(reverse=True, key=lambda x: x[0])

        # Take top N objects
        top_objects = objects_with_priority[:max_objects]

        # Format for LLM
        context = {
            'summary': {
                'total_objects': len(schema.get('objects', {})),
                'included_objects': len(top_objects),
                'patterns_found': len(schema.get('patterns', []))
            },
            'objects': []
        }

        for priority, obj_id, obj_data in top_objects:
            obj_summary = {
                'id': obj_id,
                'attributes': {
                    'color': obj_data['attributes'].get('color'),
                    'shape': obj_data['attributes'].get('shape_type'),
                    'size': obj_data['attributes'].get('size')
                },
                'seen_in': f"{len(obj_data['occurrences'])} example(s)",
                'tags': obj_data.get('tags', []),
                'variations': ObjectInventoryAdapter._summarize_variations(
                    obj_data.get('variations', {})
                )
            }
            context['objects'].append(obj_summary)

        # Add patterns
        context['patterns'] = schema.get('patterns', [])

        # Add transformation rules
        context['transformation_rules'] = schema.get('transformation_rules', [])

        return context

    @staticmethod
    def _summarize_variations(variations: Dict[str, List[Any]]) -> Dict[str, str]:
        """Summarize attribute variations."""
        summary = {}
        for attr, values in variations.items():
            if len(values) == 1:
                summary[attr] = 'stable'
            else:
                summary[attr] = f'varies ({len(values)} values)'
        return summary

    @staticmethod
    def format_deltas_for_prompt(deltas: List[ObjectDelta]) -> str:
        """Format object deltas as natural language for prompts."""
        if not deltas:
            return "No object transformations detected."

        lines = []
        lines.append("Object Transformations:")

        for i, delta in enumerate(deltas[:10], 1):  # Limit to 10 for brevity
            trans_type = delta.transformation_type
            changes_desc = []

            for attr, (old_val, new_val) in delta.changes.items():
                changes_desc.append(f"{attr}: {old_val} → {new_val}")

            changes_str = ", ".join(changes_desc) if changes_desc else "no attribute changes"

            lines.append(f"{i}. Object {delta.object_id}: {trans_type} ({changes_str})")

        return "\n".join(lines)


class RuleFailureAdapter:
    """Adapter to convert rule failures and violations to LLM-friendly format."""

    @staticmethod
    def to_llm_context(
        engine: PlianceEngine,
        violations: List[RuleViolation],
        max_violations: int = 10
    ) -> Dict[str, Any]:
        """Convert rule violations to LLM context.

        Args:
            engine: Pliance engine with rules
            violations: List of rule violations
            max_violations: Max violations to include

        Returns:
            LLM-friendly context with violation analysis
        """
        # Group violations by rule
        by_rule = {}
        for v in violations:
            by_rule.setdefault(v.rule_id, []).append(v)

        # Sort by frequency
        sorted_rules = sorted(by_rule.items(), key=lambda x: len(x[1]), reverse=True)

        context = {
            'summary': {
                'total_violations': len(violations),
                'affected_rules': len(by_rule),
                'most_common_type': RuleFailureAdapter._most_common_type(violations)
            },
            'violations': []
        }

        # Include top violations
        for rule_id, rule_violations in sorted_rules[:max_violations]:
            rule = engine.rules.get(rule_id)
            rule_name = rule.name if rule else rule_id

            violation_summary = {
                'rule_id': rule_id,
                'rule_name': rule_name,
                'count': len(rule_violations),
                'types': list(set(v.violation_type for v in rule_violations)),
                'examples': [
                    {
                        'type': v.violation_type,
                        'error': v.error_message,
                        'context': v.context
                    }
                    for v in rule_violations[:3]  # Top 3 examples
                ]
            }
            context['violations'].append(violation_summary)

        return context

    @staticmethod
    def _most_common_type(violations: List[RuleViolation]) -> str:
        """Find most common violation type."""
        if not violations:
            return "none"

        types = [v.violation_type for v in violations]
        return max(set(types), key=types.count)

    @staticmethod
    def format_for_prompt(violations: List[RuleViolation], engine: PlianceEngine) -> str:
        """Format violations as natural language for prompts."""
        if not violations:
            return "No rule violations detected."

        context = RuleFailureAdapter.to_llm_context(engine, violations)

        lines = []
        lines.append(f"Rule Violations Summary ({context['summary']['total_violations']} total):")

        for v_summary in context['violations'][:5]:  # Top 5
            lines.append(f"\n- Rule '{v_summary['rule_name']}': {v_summary['count']} violations")
            lines.append(f"  Types: {', '.join(v_summary['types'])}")

            if v_summary['examples']:
                lines.append("  Examples:")
                for ex in v_summary['examples'][:2]:
                    lines.append(f"    • {ex['type']}: {ex['error']}")

        return "\n".join(lines)


class ConflictAdapter:
    """Adapter to convert tracking conflicts to LLM-friendly format."""

    @staticmethod
    def to_llm_context(conflicts: List[RuleConflict]) -> Dict[str, Any]:
        """Convert conflicts to LLM context.

        Args:
            conflicts: List of rule conflicts

        Returns:
            LLM-friendly conflict analysis
        """
        # Sort by severity
        sorted_conflicts = sorted(conflicts, key=lambda c: c.severity, reverse=True)

        context = {
            'summary': {
                'total_conflicts': len(conflicts),
                'pending': sum(1 for c in conflicts if c.resolution_status == 'pending'),
                'resolved': sum(1 for c in conflicts if c.resolution_status == 'resolved'),
                'avg_severity': np.mean([c.severity for c in conflicts]) if conflicts else 0.0
            },
            'conflicts': []
        }

        for conflict in sorted_conflicts[:10]:  # Top 10
            conflict_summary = {
                'id': conflict.conflict_id,
                'severity': conflict.severity,
                'status': conflict.resolution_status,
                'conflicting_rules': conflict.conflicting_rules,
                'evidence_summary': {
                    rule: len(examples)
                    for rule, examples in conflict.evidence_for.items()
                },
                'notes': conflict.resolution_notes
            }
            context['conflicts'].append(conflict_summary)

        return context

    @staticmethod
    def format_for_prompt(conflicts: List[RuleConflict]) -> str:
        """Format conflicts as natural language for prompts."""
        if not conflicts:
            return "No rule conflicts detected."

        context = ConflictAdapter.to_llm_context(conflicts)

        lines = []
        lines.append(f"Rule Conflicts ({context['summary']['total_conflicts']} total, {context['summary']['pending']} pending):")

        for c in context['conflicts'][:5]:  # Top 5
            lines.append(f"\n- Conflict {c['id']} (severity: {c['severity']:.2f}):")
            lines.append(f"  Rules: {', '.join(c['conflicting_rules'])}")
            lines.append(f"  Evidence: {c['evidence_summary']}")
            if c['notes']:
                lines.append(f"  Notes: {c['notes']}")

        return "\n".join(lines)


class RepairTaskAdapter:
    """Adapter to convert repair tasks to LLM-friendly format."""

    @staticmethod
    def to_llm_context(tasks: List[RuleRepairTask]) -> Dict[str, Any]:
        """Convert repair tasks to LLM context.

        Args:
            tasks: List of repair tasks

        Returns:
            LLM-friendly task summary
        """
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        context = {
            'summary': {
                'total_tasks': len(tasks),
                'queued': sum(1 for t in tasks if t.status == 'queued'),
                'in_progress': sum(1 for t in tasks if t.status == 'in_progress'),
                'completed': sum(1 for t in tasks if t.status == 'completed')
            },
            'tasks': []
        }

        for task in sorted_tasks[:15]:  # Top 15
            task_summary = {
                'id': task.task_id,
                'rule': task.rule_id,
                'issue': task.issue_type,
                'priority': task.priority,
                'status': task.status,
                'suggested_fixes': task.suggested_fixes[:3]  # Top 3 fixes
            }
            context['tasks'].append(task_summary)

        return context

    @staticmethod
    def format_for_prompt(tasks: List[RuleRepairTask]) -> str:
        """Format repair tasks as natural language for prompts."""
        if not tasks:
            return "No repair tasks queued."

        context = RepairTaskAdapter.to_llm_context(tasks)

        lines = []
        lines.append(f"Repair Tasks ({context['summary']['queued']} queued, {context['summary']['completed']} completed):")

        for t in context['tasks'][:5]:  # Top 5
            lines.append(f"\n- Task {t['id']} (priority: {t['priority']:.2f}):")
            lines.append(f"  Rule: {t['rule']}, Issue: {t['issue']}")
            lines.append(f"  Status: {t['status']}")

            if t['suggested_fixes']:
                lines.append("  Suggested fixes:")
                for fix in t['suggested_fixes'][:2]:
                    lines.append(f"    • {fix.get('action', 'unknown')}")

        return "\n".join(lines)


class TrackingStateAdapter:
    """Adapter to convert full tracking state to LLM-friendly format."""

    @staticmethod
    def to_llm_context(tracker: RFTTracker) -> Dict[str, Any]:
        """Convert full tracking state to LLM context.

        Args:
            tracker: RFT tracker with full state

        Returns:
            Comprehensive LLM-friendly context
        """
        inventory_context = ObjectInventoryAdapter.to_llm_context(tracker.inventory)
        conflict_context = ConflictAdapter.to_llm_context(tracker.conflicts)
        repair_context = RepairTaskAdapter.to_llm_context(tracker.repair_queue)

        # Combine with metadata
        context = {
            'task_id': tracker.task_id,
            'inventory': inventory_context,
            'conflicts': conflict_context,
            'repairs': repair_context,
            'metadata': {
                'num_objects': len(tracker.inventory.entries),
                'num_facts': len(tracker.rft_analyzer.fact_database),
                'num_conflicts': len(tracker.conflicts),
                'pending_repairs': sum(1 for t in tracker.repair_queue if t.status == 'queued')
            }
        }

        return context

    @staticmethod
    def format_full_state_for_prompt(tracker: RFTTracker) -> str:
        """Format full tracking state as natural language."""
        lines = []
        lines.append(f"=== Tracking State for Task {tracker.task_id} ===\n")

        # Objects
        inv_context = ObjectInventoryAdapter.to_llm_context(tracker.inventory)
        lines.append(f"Objects: {inv_context['summary']['total_objects']} total")
        lines.append(f"Patterns: {inv_context['summary']['patterns_found']} found\n")

        # Conflicts
        if tracker.conflicts:
            lines.append(ConflictAdapter.format_for_prompt(tracker.conflicts))
            lines.append("")

        # Repair tasks
        if tracker.repair_queue:
            lines.append(RepairTaskAdapter.format_for_prompt(tracker.repair_queue))
            lines.append("")

        return "\n".join(lines)


class PrimitiveCatalogAdapter:
    """Adapter to format primitive operation catalog for LLM."""

    @staticmethod
    def get_catalog() -> Dict[str, Any]:
        """Get catalog of primitive operations with descriptions."""
        return {
            'geometric': {
                'rotate': 'Rotate grid by 90/180/270 degrees',
                'flip': 'Flip grid horizontally or vertically',
                'transpose': 'Transpose grid (swap rows and columns)',
                'scale': 'Scale grid up or down by integer factor'
            },
            'color': {
                'recolor': 'Change specific color to another color',
                'swap_colors': 'Swap two colors in the grid',
                'majority_color': 'Set cell to majority color in neighborhood',
                'gradient': 'Apply color gradient based on position'
            },
            'spatial': {
                'shift': 'Shift grid contents in a direction',
                'extract': 'Extract rectangular region from grid',
                'crop': 'Crop to bounding box of non-background',
                'pad': 'Add border padding to grid'
            },
            'object': {
                'move_object': 'Move object to new position',
                'copy_object': 'Copy object to multiple positions',
                'delete_object': 'Remove object from grid',
                'fill_region': 'Fill region with color or pattern'
            },
            'pattern': {
                'tile': 'Tile pattern across grid',
                'mirror': 'Mirror pattern across axis',
                'extend_line': 'Extend line in direction',
                'connect': 'Connect objects with line/path'
            },
            'logical': {
                'overlay': 'Overlay two grids with priority',
                'mask': 'Apply mask to show/hide regions',
                'conditional': 'Apply operation conditionally',
                'compose': 'Compose multiple operations'
            }
        }

    @staticmethod
    def format_for_prompt(categories: Optional[List[str]] = None) -> str:
        """Format primitive catalog as natural language.

        Args:
            categories: Specific categories to include (None = all)

        Returns:
            Formatted catalog string
        """
        catalog = PrimitiveCatalogAdapter.get_catalog()

        if categories:
            catalog = {k: v for k, v in catalog.items() if k in categories}

        lines = ["Available Primitive Operations:"]

        for category, operations in catalog.items():
            lines.append(f"\n{category.upper()}:")
            for op_name, description in operations.items():
                lines.append(f"  • {op_name}: {description}")

        return "\n".join(lines)
