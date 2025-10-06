"""
Search & Fallback Fusion for PUMA ARC Solver.

Integrates LLM meta-reasoning results with EnhancedSearch to:
1. Apply search_priorities to weight candidate generators
2. Seed heuristics/beam search with LLM-suggested operations
3. Feed validated pliance rules into search pipeline
4. Trigger re-search after rule amendments
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

import numpy as np

from .grid import Array
from .dsl import OPS, apply_program
from .heuristics import score_candidate
from .pliance_engine import PlianceEngine, PlianceRule
from .rft_tracking import RFTTracker


class SearchPriorityWeighter:
    """Applies MetaReasoningResult priorities to weight search methods."""

    def __init__(self):
        self.default_weights = {
            'episodic_memory': 1.0,
            'sketch_search': 0.8,
            'rft_engine': 0.9,
            'beam_search': 1.0,
            'neural_guidance': 0.7,
            'pliance_rules': 0.85,
            'human_reasoning': 1.2,
            'memory_candidates': 1.0,
            'facts_guided': 0.9
        }

    def apply_priorities(
        self,
        base_candidates: Dict[str, List[Any]],
        priorities: Dict[str, float]
    ) -> List[Tuple[Any, float]]:
        """Apply priorities to weight and merge candidates.

        Args:
            base_candidates: Dict mapping method name to candidate list
            priorities: Dict mapping method name to priority (0.0-1.0)

        Returns:
            List of (candidate, weight) tuples sorted by weight
        """
        weighted_candidates = []

        for method, candidates in base_candidates.items():
            priority = priorities.get(method, self.default_weights.get(method, 0.5))

            for candidate in candidates:
                weighted_candidates.append((candidate, priority))

        # Sort by weight (descending)
        weighted_candidates.sort(key=lambda x: x[1], reverse=True)

        return weighted_candidates

    def sample_by_priority(
        self,
        weighted_candidates: List[Tuple[Any, float]],
        max_samples: int
    ) -> List[Any]:
        """Sample candidates proportional to priorities.

        Args:
            weighted_candidates: List of (candidate, weight) tuples
            max_samples: Maximum candidates to return

        Returns:
            Sampled candidates
        """
        if not weighted_candidates:
            return []

        if len(weighted_candidates) <= max_samples:
            return [c for c, w in weighted_candidates]

        # Normalize weights to probabilities
        candidates, weights = zip(*weighted_candidates)
        weights = np.array(weights)
        probs = weights / weights.sum()

        # Sample without replacement
        indices = np.random.choice(
            len(candidates),
            size=min(max_samples, len(candidates)),
            replace=False,
            p=probs
        )

        return [candidates[i] for i in indices]


class OperationSeeder:
    """Seeds search with LLM-suggested operations and parameters."""

    def __init__(self, operations_catalog: Optional[Dict[str, Any]] = None):
        """Initialize seeder.

        Args:
            operations_catalog: Catalog of available operations
        """
        self.catalog = operations_catalog or self._default_catalog()

    def _default_catalog(self) -> Dict[str, Any]:
        """Get default operations catalog."""
        return {
            'rotate': {'params': ['angle'], 'valid_angles': [90, 180, 270]},
            'flip': {'params': ['axis'], 'valid_axes': ['h', 'v']},
            'recolor': {'params': ['from_color', 'to_color'], 'color_range': range(0, 10)},
            'extract': {'params': ['x', 'y', 'w', 'h']},
            'scale': {'params': ['factor'], 'valid_factors': [2, 3, 4]},
            'shift': {'params': ['dx', 'dy']},
            'transpose': {'params': []},
            'crop': {'params': []},
            'tile': {'params': ['nx', 'ny']},
            'mirror': {'params': ['axis'], 'valid_axes': ['h', 'v']},
        }

    def seed_beam_search(
        self,
        suggested_operations: List[str],
        operation_parameters: Dict[str, Dict[str, Any]],
        train_pairs: List[Tuple[Array, Array]]
    ) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Generate initial programs for beam search based on LLM suggestions.

        Args:
            suggested_operations: LLM-suggested operation names
            operation_parameters: LLM-suggested parameters per operation
            train_pairs: Training pairs for validation

        Returns:
            List of seed programs
        """
        seed_programs = []

        # Single-step programs from suggestions
        for op_name in suggested_operations[:10]:  # Top 10
            if op_name not in OPS:
                continue

            # Get suggested parameters
            param_hints = operation_parameters.get(op_name, {})

            # Generate program with hints
            program = self._generate_program_with_hints(op_name, param_hints)

            if program:
                seed_programs.append(program)

        # Two-step compositions from top suggestions
        for i, op1 in enumerate(suggested_operations[:5]):
            for op2 in suggested_operations[i+1:6]:
                if op1 in OPS and op2 in OPS:
                    param1 = operation_parameters.get(op1, {})
                    param2 = operation_parameters.get(op2, {})

                    prog1 = self._generate_program_with_hints(op1, param1)
                    prog2 = self._generate_program_with_hints(op2, param2)

                    if prog1 and prog2:
                        seed_programs.append(prog1 + prog2)

        return seed_programs

    def _generate_program_with_hints(
        self,
        op_name: str,
        param_hints: Dict[str, Any]
    ) -> Optional[List[Tuple[str, Dict[str, int]]]]:
        """Generate a program step with parameter hints.

        Args:
            op_name: Operation name
            param_hints: Parameter hints from LLM

        Returns:
            Program step or None if invalid
        """
        if op_name not in OPS:
            return None

        op_info = self.catalog.get(op_name, {})
        params = {}

        # Apply hints where possible
        for param_name in op_info.get('params', []):
            if param_name in param_hints:
                params[param_name] = param_hints[param_name]
            else:
                # Use default/first valid value
                if param_name == 'angle':
                    params[param_name] = 90
                elif param_name == 'axis':
                    params[param_name] = 'h'
                elif param_name == 'factor':
                    params[param_name] = 2
                elif param_name in ['dx', 'dy', 'x', 'y', 'w', 'h']:
                    params[param_name] = 1

        return [(op_name, params)]

    def enhance_heuristic_search(
        self,
        base_candidates: List[List[Tuple[str, Dict[str, int]]]],
        suggested_operations: List[str],
        max_enhanced: int = 50
    ) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Enhance heuristic candidates with LLM suggestions.

        Args:
            base_candidates: Base heuristic candidates
            suggested_operations: LLM-suggested operations
            max_enhanced: Maximum enhanced candidates to generate

        Returns:
            Enhanced candidate list
        """
        enhanced = deepcopy(base_candidates)

        # Add suggested single-step operations
        for op_name in suggested_operations[:max_enhanced // 2]:
            if op_name in OPS:
                program = self._generate_program_with_hints(op_name, {})
                if program and program not in enhanced:
                    enhanced.append(program)

        return enhanced[:max_enhanced]


class PlianceRuleFusion:
    """Feeds validated pliance rules into search subsystems."""

    def __init__(self, engine: PlianceEngine):
        """Initialize fusion module.

        Args:
            engine: Pliance engine with rules
        """
        self.engine = engine

    def apply_rules_to_candidates(
        self,
        candidates: List[List[Tuple[str, Dict[str, int]]]],
        train_pairs: List[Tuple[Array, Array]]
    ) -> List[List[Tuple[str, Dict[str, int]]]]:
        """Apply pliance rules to generate new candidates.

        Args:
            candidates: Existing candidates
            train_pairs: Training pairs

        Returns:
            Enhanced candidates with rule applications
        """
        enhanced_candidates = deepcopy(candidates)

        # For each training pair, apply high-confidence rules
        high_confidence_rules = [
            r for r in self.engine.rules.values()
            if r.confidence >= 0.7 and r.enabled
        ]

        for rule in high_confidence_rules[:10]:  # Top 10 rules
            # Try to convert rule to program
            program = self._rule_to_program(rule)

            if program and program not in enhanced_candidates:
                enhanced_candidates.append(program)

        return enhanced_candidates

    def _rule_to_program(
        self,
        rule: PlianceRule
    ) -> Optional[List[Tuple[str, Dict[str, int]]]]:
        """Convert a pliance rule to a DSL program.

        Args:
            rule: Pliance rule

        Returns:
            Program or None if not convertible
        """
        if rule.action is None:
            return None

        action_type = rule.action.action_type
        params = rule.action.parameters

        # Map action types to DSL operations
        if action_type == 'recolor':
            color = params.get('color', 0)
            # Need to determine source color from selector
            from_color = rule.selector.color or 0
            return [('recolor', {'from_color': from_color, 'to_color': color})]

        elif action_type == 'move':
            dr = params.get('dr', 0)
            dc = params.get('dc', 0)
            return [('shift', {'dx': dc, 'dy': dr})]

        elif action_type == 'fill_region':
            color = params.get('color', 0)
            # Placeholder - would need more sophisticated mapping
            return None

        # Not directly convertible
        return None

    def inject_into_episodic_memory(
        self,
        episodic_retrieval,
        train_pairs: List[Tuple[Array, Array]]
    ) -> None:
        """Inject successful pliance rules into episodic memory.

        Args:
            episodic_retrieval: Episodic retrieval system
            train_pairs: Training pairs
        """
        # Validate rules against training pairs
        validation_results = self.engine.validate_rules(train_pairs)

        # Add high-accuracy rules to episodic memory
        for rule_id, results in validation_results.items():
            if results['accuracy'] >= 0.8:
                program = self._rule_to_program(self.engine.rules[rule_id])

                if program:
                    episodic_retrieval.add_successful_solution(
                        train_pairs,
                        [program],
                        metadata={'source': 'pliance_rule', 'rule_id': rule_id}
                    )


class AdaptiveResearcher:
    """Triggers re-search after rule amendments."""

    def __init__(self):
        self.search_history: List[Dict[str, Any]] = []

    def should_research(
        self,
        tracker: RFTTracker,
        last_search_result: Optional[List[Any]]
    ) -> bool:
        """Determine if re-search is needed based on tracking state.

        Args:
            tracker: RFT tracker with current state
            last_search_result: Result of last search

        Returns:
            True if re-search is warranted
        """
        # Re-search if:
        # 1. No previous search
        if last_search_result is None:
            return True

        # 2. Conflicts were resolved since last search
        pending_repairs = sum(1 for t in tracker.repair_queue if t.status == 'queued')
        if pending_repairs > 0:
            return True

        # 3. Last search yielded no candidates
        if not last_search_result:
            return True

        # 4. New objects discovered (inventory grew)
        if len(self.search_history) > 0:
            prev_obj_count = self.search_history[-1].get('object_count', 0)
            curr_obj_count = len(tracker.inventory.entries)

            if curr_obj_count > prev_obj_count:
                return True

        return False

    def research_with_updates(
        self,
        search_func,
        tracker: RFTTracker,
        train_pairs: List[Tuple[Array, Array]],
        **search_kwargs
    ) -> List[Any]:
        """Perform re-search with updated knowledge.

        Args:
            search_func: Search function to call
            tracker: RFT tracker with updated state
            train_pairs: Training pairs
            **search_kwargs: Additional search arguments

        Returns:
            Search results
        """
        # Record search attempt
        self.search_history.append({
            'object_count': len(tracker.inventory.entries),
            'conflict_count': len(tracker.conflicts),
            'pending_repairs': sum(1 for t in tracker.repair_queue if t.status == 'queued')
        })

        # Perform search with updated context
        results = search_func(train_pairs, **search_kwargs)

        return results


def integrate_search_fusion(
    enhanced_search,
    meta_reasoning_result: Optional[Any],
    pliance_engine: Optional[PlianceEngine] = None,
    tracker: Optional[RFTTracker] = None
) -> Dict[str, Any]:
    """Integrate search fusion components into EnhancedSearch.

    Args:
        enhanced_search: EnhancedSearch instance
        meta_reasoning_result: MetaReasoningResult from LLM
        pliance_engine: Optional pliance engine
        tracker: Optional RFT tracker

    Returns:
        Integration config to pass to search
    """
    config = {
        'use_priorities': False,
        'use_operation_seeding': False,
        'use_pliance_fusion': False,
        'use_adaptive_research': False
    }

    # Configure priority weighting
    if meta_reasoning_result:
        config['use_priorities'] = True
        config['priorities'] = meta_reasoning_result.search_priorities
        config['suggested_operations'] = meta_reasoning_result.suggested_operations
        config['operation_parameters'] = meta_reasoning_result.operation_parameters

    # Configure pliance fusion
    if pliance_engine:
        config['use_pliance_fusion'] = True
        config['pliance_engine'] = pliance_engine

    # Configure adaptive research
    if tracker:
        config['use_adaptive_research'] = True
        config['tracker'] = tracker

    return config
