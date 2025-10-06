"""
LLM Prompt Templates for PUMA ARC Solver.

Provides structured prompt templates for various reasoning tasks:
- Object inventory analysis
- Rule synthesis from examples
- Conflict resolution
- Search strategy prioritization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .llm_adapters import (
    ObjectInventoryAdapter,
    RuleFailureAdapter,
    ConflictAdapter,
    RepairTaskAdapter,
    PrimitiveCatalogAdapter
)


class PromptTemplates:
    """Collection of prompt templates for LLM reasoning."""

    @staticmethod
    def inventory_analysis_prompt(inventory_context: Dict[str, Any]) -> Dict[str, str]:
        """Generate prompt for analyzing object inventory.

        Args:
            inventory_context: Output from ObjectInventoryAdapter.to_llm_context()

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are an expert at analyzing visual patterns in ARC (Abstraction and Reasoning Corpus) tasks.
Your role is to examine object inventories from training examples and identify key patterns, transformations,
and relationships that will help solve the task.

Focus on:
1. Stable vs varying object attributes
2. Transformation patterns (movement, recoloring, resizing)
3. Spatial relationships between objects
4. Patterns that recur across multiple examples"""

        summary = inventory_context['summary']
        objects = inventory_context['objects']
        patterns = inventory_context.get('patterns', [])

        user = f"""Analyze this object inventory from an ARC task:

SUMMARY:
- Total objects tracked: {summary['total_objects']}
- Objects shown: {summary['included_objects']}
- Patterns detected: {summary['patterns_found']}

OBJECTS:
"""
        for obj in objects:
            attrs = obj['attributes']
            user += f"""
Object {obj['id']}:
  - Color: {attrs.get('color')}, Shape: {attrs.get('shape')}, Size: {attrs.get('size')}
  - Seen in: {obj['seen_in']}
  - Tags: {', '.join(obj.get('tags', []))}
  - Variations: {obj.get('variations', {})}
"""

        if patterns:
            user += "\nDETECTED PATTERNS:\n"
            for i, pattern in enumerate(patterns, 1):
                user += f"{i}. {pattern.get('type')}: {pattern}\n"

        user += """
Based on this inventory, provide:
1. Key observations about object transformations
2. Most important patterns to leverage
3. Suggested transformation rules to test
4. Confidence in these observations (0-1 scale)

Format your response as JSON:
{
  "observations": ["observation 1", "observation 2", ...],
  "key_patterns": ["pattern 1", "pattern 2", ...],
  "suggested_rules": [
    {"selector": {...}, "action": {...}, "rationale": "..."},
    ...
  ],
  "confidence": 0.0-1.0
}"""

        return {'system': system, 'user': user}

    @staticmethod
    def rule_synthesis_prompt(
        inventory_context: Dict[str, Any],
        transformation_rules: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate prompt for synthesizing pliance rules from patterns.

        Args:
            inventory_context: Object inventory context
            transformation_rules: Existing transformation rules

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are an expert at creating precise transformation rules for ARC tasks.
Given object inventories and observed transformation patterns, you generate formal pliance rules
that specify: which objects to select, what transformations to apply, and when to apply them.

Rules should be:
- Specific enough to avoid false matches
- General enough to work across all examples
- Ranked by confidence (how certain you are they'll work)"""

        user = f"""Create pliance rules for this ARC task.

OBJECT INVENTORY:
{len(inventory_context.get('objects', []))} objects tracked

EXISTING TRANSFORMATION PATTERNS:
"""
        for i, rule in enumerate(transformation_rules, 1):
            user += f"{i}. {rule.get('transformation')}: {rule.get('common_changes', {})}\n"

        user += """
Generate 3-5 pliance rules that capture the core transformation logic.

For each rule, specify:
- selector: Which objects it applies to (color, shape, size, position criteria)
- relations: Spatial relationships required (optional)
- action: Transformation to apply (recolor, move, copy, delete, fill_region)
- confidence: How confident you are (0.0-1.0)
- rationale: Why this rule makes sense

Format as JSON:
{
  "rules": [
    {
      "name": "descriptive_name",
      "selector": {"color": X, "shape_type": "Y", ...},
      "relations": [],
      "action": {"action_type": "...", "parameters": {...}},
      "confidence": 0.0-1.0,
      "rationale": "explanation"
    },
    ...
  ]
}"""

        return {'system': system, 'user': user}

    @staticmethod
    def conflict_resolution_prompt(
        conflict_context: Dict[str, Any],
        inventory_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate prompt for resolving rule conflicts.

        Args:
            conflict_context: Output from ConflictAdapter.to_llm_context()
            inventory_context: Object inventory for reference

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are an expert at resolving conflicts in transformation rules for ARC tasks.
When multiple rules conflict (predict different outcomes), you analyze the evidence for each
and suggest the best resolution strategy.

Resolution strategies:
1. Choose the rule with strongest evidence
2. Create conditional rule (apply different rules in different contexts)
3. Merge rules into a more general form
4. Mark one rule as exception to another
5. Disable low-confidence rule"""

        summary = conflict_context['summary']
        conflicts = conflict_context.get('conflicts', [])

        user = f"""Resolve these rule conflicts:

SUMMARY:
- Total conflicts: {summary['total_conflicts']}
- Pending: {summary['pending']}
- Average severity: {summary['avg_severity']:.2f}

CONFLICTS:
"""
        for conf in conflicts[:5]:  # Top 5
            user += f"""
Conflict {conf['id']} (severity {conf['severity']:.2f}):
  - Conflicting rules: {', '.join(conf['conflicting_rules'])}
  - Evidence: {conf['evidence_summary']}
  - Status: {conf['status']}
"""

        user += """
For each conflict, suggest a resolution strategy.

Format as JSON:
{
  "resolutions": [
    {
      "conflict_id": "...",
      "strategy": "choose_best|conditional|merge|exception|disable",
      "chosen_rule": "..." (if applicable),
      "condition": "..." (if conditional),
      "rationale": "explanation",
      "confidence": 0.0-1.0
    },
    ...
  ]
}"""

        return {'system': system, 'user': user}

    @staticmethod
    def search_prioritization_prompt(
        task_features: Dict[str, Any],
        inventory_summary: Dict[str, Any],
        available_methods: List[str]
    ) -> Dict[str, str]:
        """Generate prompt for prioritizing search strategies.

        Args:
            task_features: Extracted task features
            inventory_summary: Summary of object inventory
            available_methods: List of available search methods

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are a meta-reasoner that decides which search strategies to prioritize for ARC tasks.
Given task characteristics and object patterns, you predict which search methods will be most effective.

Available methods:
- episodic_memory: Use similar past solutions
- sketch_search: Generate grid structure sketches
- rft_engine: Apply relational transformation rules
- beam_search: Explore operation sequences
- neural_guidance: Use learned operation predictions
- pliance_rules: Apply formal transformation rules"""

        user = f"""Prioritize search strategies for this ARC task:

TASK CHARACTERISTICS:
"""
        for key, value in task_features.items():
            user += f"  - {key}: {value}\n"

        user += f"""
OBJECT INVENTORY SUMMARY:
  - Total objects: {inventory_summary.get('total_objects', 0)}
  - Patterns found: {inventory_summary.get('patterns_found', 0)}

AVAILABLE METHODS:
{', '.join(available_methods)}

Recommend:
1. Primary strategy (most promising method)
2. Weight/priority for each method (0.0-1.0)
3. Suggested operation sequence to try first
4. Overall confidence in recommendations

Format as JSON:
{{
  "primary_strategy": "method_name",
  "method_priorities": {{
    "episodic_memory": 0.0-1.0,
    "sketch_search": 0.0-1.0,
    "rft_engine": 0.0-1.0,
    "beam_search": 0.0-1.0,
    "neural_guidance": 0.0-1.0,
    "pliance_rules": 0.0-1.0
  }},
  "suggested_operations": ["op1", "op2", "op3", ...],
  "reasoning": "explanation of strategy choice",
  "confidence": 0.0-1.0
}}"""

        return {'system': system, 'user': user}

    @staticmethod
    def repair_task_prompt(
        task_context: Dict[str, Any],
        rule_id: str,
        issue_type: str,
        suggested_fixes: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate prompt for repair task execution.

        Args:
            task_context: Context about the repair task
            rule_id: ID of rule to repair
            issue_type: Type of issue (conflict, low_confidence, inconsistent_application)
            suggested_fixes: Pre-suggested fixes from repair system

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are an expert at debugging and repairing transformation rules for ARC tasks.
Given a rule that's failing or conflicting, you analyze the issue and recommend specific fixes
to improve the rule's accuracy and consistency."""

        user = f"""Repair this transformation rule:

RULE: {rule_id}
ISSUE TYPE: {issue_type}

SUGGESTED FIXES:
"""
        for i, fix in enumerate(suggested_fixes, 1):
            user += f"{i}. Action: {fix.get('action')}\n"
            if 'reason' in fix:
                user += f"   Reason: {fix['reason']}\n"

        user += f"""
CONTEXT:
{task_context}

Recommend the best fix to apply.

Format as JSON:
{{
  "recommended_fix": {{
    "action": "adjust_confidence|modify_selector|modify_action|add_relation|disable",
    "parameters": {{}},
    "expected_improvement": "description"
  }},
  "alternative_fixes": [...],
  "confidence": 0.0-1.0,
  "rationale": "explanation"
}}"""

        return {'system': system, 'user': user}

    @staticmethod
    def primitive_selection_prompt(
        task_description: str,
        inventory_summary: Dict[str, Any],
        primitive_categories: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Generate prompt for selecting relevant primitive operations.

        Args:
            task_description: Natural language description of task patterns
            inventory_summary: Summary of object inventory
            primitive_categories: Specific categories to consider

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are an expert at selecting primitive operations to solve ARC tasks.
Given a task description and object patterns, you identify which primitive operations
are most likely to be useful and in what sequence they should be applied."""

        catalog_str = PrimitiveCatalogAdapter.format_for_prompt(primitive_categories)

        user = f"""Select relevant primitive operations for this task:

TASK PATTERNS:
{task_description}

OBJECT SUMMARY:
  - Total objects: {inventory_summary.get('total_objects', 0)}
  - Patterns: {inventory_summary.get('patterns_found', 0)}

{catalog_str}

Recommend:
1. Top 5-7 most relevant primitives
2. Suggested sequence to apply them
3. Parameter hints for each primitive

Format as JSON:
{{
  "selected_primitives": [
    {{
      "operation": "primitive_name",
      "category": "geometric|color|spatial|object|pattern|logical",
      "relevance": 0.0-1.0,
      "parameter_hints": {{}}
    }},
    ...
  ],
  "suggested_sequence": ["op1", "op2", "op3", ...],
  "reasoning": "explanation"
}}"""

        return {'system': system, 'user': user}

    @staticmethod
    def multi_step_reasoning_prompt(
        inventory_context: Dict[str, Any],
        conflict_context: Dict[str, Any],
        repair_context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate comprehensive multi-step reasoning prompt.

        Combines inventory analysis, conflict resolution, and repair planning
        into a single reasoning session.

        Args:
            inventory_context: Object inventory
            conflict_context: Rule conflicts
            repair_context: Repair tasks

        Returns:
            Dict with 'system' and 'user' prompts
        """
        system = """You are a comprehensive meta-reasoner for ARC tasks. You analyze object inventories,
resolve rule conflicts, and plan repairs in an integrated way to develop the best solution strategy.

Your analysis should be holistic:
1. Understand core transformation patterns from objects
2. Identify and resolve conflicting hypotheses
3. Prioritize repairs that will have most impact
4. Recommend an integrated solution approach"""

        inv_summary = inventory_context['summary']
        conf_summary = conflict_context['summary']
        repair_summary = repair_context['summary']

        user = f"""Perform integrated analysis of this ARC task:

OBJECT INVENTORY:
  - {inv_summary['total_objects']} objects tracked
  - {inv_summary['patterns_found']} patterns detected
  - Top objects: {len(inventory_context.get('objects', []))} shown

CONFLICTS:
  - {conf_summary['total_conflicts']} conflicts
  - {conf_summary['pending']} pending resolution
  - Avg severity: {conf_summary['avg_severity']:.2f}

REPAIR QUEUE:
  - {repair_summary['total_tasks']} tasks
  - {repair_summary['queued']} queued
  - {repair_summary['completed']} completed

Provide integrated recommendations:

1. PATTERN ANALYSIS: What are the core transformation patterns?
2. CONFLICT RESOLUTION: How should conflicts be resolved?
3. REPAIR PRIORITIES: Which repairs are most critical?
4. SOLUTION STRATEGY: Recommended approach to solve this task
5. CONFIDENCE: Overall confidence in recommendations (0.0-1.0)

Format as JSON:
{{
  "pattern_analysis": {{
    "core_patterns": ["pattern1", "pattern2", ...],
    "key_objects": ["obj_id1", "obj_id2", ...],
    "transformation_type": "description"
  }},
  "conflict_resolutions": [
    {{"conflict_id": "...", "strategy": "...", "rationale": "..."}},
    ...
  ],
  "repair_priorities": [
    {{"task_id": "...", "priority": 0.0-1.0, "rationale": "..."}},
    ...
  ],
  "solution_strategy": {{
    "approach": "description",
    "steps": ["step1", "step2", ...],
    "fallback": "if primary approach fails"
  }},
  "confidence": 0.0-1.0
}}"""

        return {'system': system, 'user': user}
