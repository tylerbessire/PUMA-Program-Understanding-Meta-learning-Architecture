"""
LLM Fallback Behaviors for PUMA ARC Solver.

Defines fallback strategies when LLM access is unavailable, disabled,
or fails. Ensures solver degrades gracefully with heuristic-based
reasoning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

from .grid import Array
from .object_inventory import ObjectInventory
from .pliance_engine import PlianceEngine, ObjectSelector, RuleAction, PlianceRule
from .rft_tracking import RFTTracker, RuleConflict


logger = logging.getLogger(__name__)


class FallbackReasoner:
    """Heuristic-based reasoning when LLM is unavailable."""

    @staticmethod
    def analyze_inventory_fallback(inventory: ObjectInventory) -> Dict[str, Any]:
        """Fallback inventory analysis using heuristics.

        Args:
            inventory: Object inventory to analyze

        Returns:
            Analysis dict similar to LLM output but heuristic-based
        """
        schema = inventory.get_rule_friendly_schema()
        objects = schema.get('objects', {})
        patterns = schema.get('patterns', [])
        transformation_rules = schema.get('transformation_rules', [])

        observations = []
        key_patterns = []
        suggested_rules = []

        # Heuristic 1: Identify most common transformations
        if transformation_rules:
            most_common = max(transformation_rules, key=lambda r: r.get('example_count', 0))
            observations.append(
                f"Most common transformation: {most_common.get('transformation')} "
                f"({most_common.get('example_count')} examples)"
            )
            key_patterns.append(most_common.get('transformation'))

        # Heuristic 2: Identify stable vs changing attributes
        stable_colors = []
        changing_colors = []

        for obj_id, obj_data in objects.items():
            variations = obj_data.get('variations', {})
            color_variations = variations.get('color', [])

            if len(color_variations) == 1:
                stable_colors.append(color_variations[0])
            else:
                changing_colors.extend(color_variations)

        if stable_colors:
            observations.append(f"Stable colors: {set(stable_colors)}")

        if changing_colors:
            observations.append(f"Colors that change: {set(changing_colors)}")
            key_patterns.append("color_transformation")

        # Heuristic 3: Generate simple rules based on consistent patterns
        for pattern in patterns:
            if pattern.get('type') == 'consistent_transformation':
                trans_type = pattern.get('transformation')
                objects_affected = pattern.get('objects', [])

                if trans_type == 'recolored' and objects_affected:
                    # Suggest recoloring rule
                    suggested_rules.append({
                        'selector': {'tags': ['auto_detected']},
                        'action': {'action_type': 'recolor'},
                        'rationale': 'Consistent recoloring pattern detected'
                    })

                elif trans_type == 'moved' and objects_affected:
                    # Suggest movement rule
                    suggested_rules.append({
                        'selector': {'tags': ['auto_detected']},
                        'action': {'action_type': 'move'},
                        'rationale': 'Consistent movement pattern detected'
                    })

        # Default confidence based on pattern consistency
        confidence = 0.6 if len(patterns) > 0 else 0.3

        return {
            'observations': observations,
            'key_patterns': key_patterns,
            'suggested_rules': suggested_rules,
            'confidence': confidence,
            'method': 'heuristic_fallback'
        }

    @staticmethod
    def resolve_conflict_fallback(conflict: RuleConflict) -> Dict[str, Any]:
        """Fallback conflict resolution using heuristics.

        Args:
            conflict: Rule conflict to resolve

        Returns:
            Resolution strategy dict
        """
        # Heuristic: Choose rule with most evidence
        evidence_counts = {
            rule: len(examples)
            for rule, examples in conflict.evidence_for.items()
        }

        if evidence_counts:
            chosen_rule = max(evidence_counts.keys(), key=lambda r: evidence_counts[r])

            return {
                'conflict_id': conflict.conflict_id,
                'strategy': 'choose_best',
                'chosen_rule': chosen_rule,
                'rationale': f'Chosen based on evidence count ({evidence_counts[chosen_rule]} examples)',
                'confidence': 0.6,
                'method': 'heuristic_fallback'
            }

        # Fallback: disable if no clear winner
        return {
            'conflict_id': conflict.conflict_id,
            'strategy': 'disable',
            'rationale': 'No clear evidence for any rule',
            'confidence': 0.4,
            'method': 'heuristic_fallback'
        }

    @staticmethod
    def prioritize_search_fallback(
        task_features: Dict[str, Any],
        inventory_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback search prioritization using heuristics.

        Args:
            task_features: Task features
            inventory_summary: Inventory summary

        Returns:
            Search priorities dict
        """
        priorities = {
            'episodic_memory': 0.5,
            'sketch_search': 0.4,
            'rft_engine': 0.6,
            'beam_search': 0.5,
            'neural_guidance': 0.3,
            'pliance_rules': 0.5
        }

        # Heuristic adjustments based on task features
        num_objects = inventory_summary.get('total_objects', 0)
        patterns_found = inventory_summary.get('patterns_found', 0)

        # If many patterns detected, boost pliance rules and rft
        if patterns_found > 3:
            priorities['pliance_rules'] = 0.8
            priorities['rft_engine'] = 0.8

        # If few objects, boost sketch search
        if num_objects < 5:
            priorities['sketch_search'] = 0.7

        # If output size different from input, boost episodic memory
        if task_features.get('output_size_differs', False):
            priorities['episodic_memory'] = 0.7

        # Choose primary strategy (highest priority)
        primary_strategy = max(priorities.keys(), key=lambda k: priorities[k])

        # Suggest generic operations
        suggested_operations = ['rotate', 'flip', 'recolor', 'extract']

        return {
            'primary_strategy': primary_strategy,
            'method_priorities': priorities,
            'suggested_operations': suggested_operations,
            'reasoning': 'Heuristic-based priorities',
            'confidence': 0.5,
            'method': 'heuristic_fallback'
        }

    @staticmethod
    def suggest_repair_fallback(
        rule_id: str,
        issue_type: str,
        suggested_fixes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback repair suggestion using heuristics.

        Args:
            rule_id: Rule to repair
            issue_type: Type of issue
            suggested_fixes: Pre-suggested fixes

        Returns:
            Repair recommendation dict
        """
        # Simple heuristic: use first suggested fix
        if suggested_fixes:
            recommended_fix = suggested_fixes[0]
        else:
            # Default: reduce confidence
            recommended_fix = {
                'action': 'adjust_confidence',
                'parameters': {'new_confidence': 0.5},
                'expected_improvement': 'Reduce confidence to avoid over-application'
            }

        return {
            'recommended_fix': recommended_fix,
            'alternative_fixes': suggested_fixes[1:] if len(suggested_fixes) > 1 else [],
            'confidence': 0.5,
            'rationale': 'Default heuristic repair',
            'method': 'heuristic_fallback'
        }


class LLMFallbackWrapper:
    """Wrapper that adds fallback behavior to any LLM-based component."""

    def __init__(
        self,
        llm_component,
        enable_logging: bool = True
    ):
        """Initialize fallback wrapper.

        Args:
            llm_component: Component that uses LLM (should have .enabled attribute)
            enable_logging: Whether to log fallback events
        """
        self.llm_component = llm_component
        self.enable_logging = enable_logging
        self.fallback_count = 0

    def _log_fallback(self, method_name: str, reason: str) -> None:
        """Log fallback event."""
        if self.enable_logging:
            logger.warning(f"LLM fallback triggered in {method_name}: {reason}")
        self.fallback_count += 1

    def analyze_inventory(self, inventory: ObjectInventory) -> Dict[str, Any]:
        """Analyze inventory with fallback.

        Args:
            inventory: Object inventory

        Returns:
            Analysis dict (from LLM or fallback)
        """
        # Try LLM if enabled
        if hasattr(self.llm_component, 'enabled') and self.llm_component.enabled:
            try:
                if hasattr(self.llm_component, 'analyze_inventory'):
                    return self.llm_component.analyze_inventory(inventory)
            except Exception as e:
                self._log_fallback('analyze_inventory', str(e))

        # Fallback
        return FallbackReasoner.analyze_inventory_fallback(inventory)

    def resolve_conflict(self, conflict: RuleConflict) -> Dict[str, Any]:
        """Resolve conflict with fallback.

        Args:
            conflict: Rule conflict

        Returns:
            Resolution dict (from LLM or fallback)
        """
        # Try LLM if enabled
        if hasattr(self.llm_component, 'enabled') and self.llm_component.enabled:
            try:
                if hasattr(self.llm_component, 'resolve_conflict'):
                    return self.llm_component.resolve_conflict(conflict)
            except Exception as e:
                self._log_fallback('resolve_conflict', str(e))

        # Fallback
        return FallbackReasoner.resolve_conflict_fallback(conflict)

    def prioritize_search(
        self,
        task_features: Dict[str, Any],
        inventory_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prioritize search with fallback.

        Args:
            task_features: Task features
            inventory_summary: Inventory summary

        Returns:
            Priorities dict (from LLM or fallback)
        """
        # Try LLM if enabled
        if hasattr(self.llm_component, 'enabled') and self.llm_component.enabled:
            try:
                if hasattr(self.llm_component, 'prioritize_search'):
                    return self.llm_component.prioritize_search(task_features, inventory_summary)
            except Exception as e:
                self._log_fallback('prioritize_search', str(e))

        # Fallback
        return FallbackReasoner.prioritize_search_fallback(task_features, inventory_summary)

    def suggest_repair(
        self,
        rule_id: str,
        issue_type: str,
        suggested_fixes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Suggest repair with fallback.

        Args:
            rule_id: Rule to repair
            issue_type: Issue type
            suggested_fixes: Pre-suggested fixes

        Returns:
            Repair dict (from LLM or fallback)
        """
        # Try LLM if enabled
        if hasattr(self.llm_component, 'enabled') and self.llm_component.enabled:
            try:
                if hasattr(self.llm_component, 'suggest_repair'):
                    return self.llm_component.suggest_repair(rule_id, issue_type, suggested_fixes)
            except Exception as e:
                self._log_fallback('suggest_repair', str(e))

        # Fallback
        return FallbackReasoner.suggest_repair_fallback(rule_id, issue_type, suggested_fixes)

    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback statistics.

        Returns:
            Dict with fallback counts
        """
        return {
            'total_fallbacks': self.fallback_count,
            'llm_enabled': hasattr(self.llm_component, 'enabled') and self.llm_component.enabled
        }


def create_fallback_reasoner(llm_enabled: bool = True) -> LLMFallbackWrapper:
    """Create a fallback-capable reasoner.

    Args:
        llm_enabled: Whether to enable LLM (if False, always uses fallbacks)

    Returns:
        Fallback wrapper with dummy LLM component
    """
    # Create dummy component
    class DummyLLMComponent:
        def __init__(self, enabled: bool):
            self.enabled = enabled

    dummy = DummyLLMComponent(llm_enabled)
    return LLMFallbackWrapper(dummy)
