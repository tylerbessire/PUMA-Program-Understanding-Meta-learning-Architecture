"""
Extended RFT tracking that ingests object history, resolves conflicts,
and queues rule repair tasks.

Integrates ObjectInventory with RFT analysis to maintain a comprehensive
tracking state per task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict

import numpy as np

from .grid import Array
from .object_inventory import ObjectInventory, ObjectDelta, ObjectEntry
from .rft import RelationalFrameAnalyzer, RelationalFact


@dataclass
class RuleConflict:
    """Represents a conflict between competing rules or observations."""
    conflict_id: str
    conflicting_rules: List[str]  # Rule IDs that conflict
    evidence_for: Dict[str, List[str]]  # rule_id -> [example_ids]
    evidence_against: Dict[str, List[str]]
    severity: float  # 0.0 to 1.0
    resolution_status: str  # 'pending', 'resolved', 'ignored'
    resolution_notes: str = ""


@dataclass
class RuleRepairTask:
    """A queued task to repair or refine a rule."""
    task_id: str
    rule_id: str
    issue_type: str  # 'conflict', 'low_confidence', 'inconsistent_application', 'missing_cases'
    priority: float  # Higher = more urgent
    suggested_fixes: List[Dict[str, Any]]
    status: str  # 'queued', 'in_progress', 'completed', 'failed'


@dataclass
class TrackingState:
    """Complete tracking state for a task."""
    task_id: str
    inventory: ObjectInventory
    relational_facts: Dict[str, List[RelationalFact]]
    conflicts: List[RuleConflict]
    repair_queue: List[RuleRepairTask]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: int = 1


class RFTTracker:
    """Extended RFT tracking with object history integration."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.inventory = ObjectInventory()
        self.rft_analyzer = RelationalFrameAnalyzer()
        self.conflicts: List[RuleConflict] = []
        self.repair_queue: List[RuleRepairTask] = []
        self._next_conflict_id = 0
        self._next_task_id = 0

    def ingest_training_pairs(self, train_pairs: List[Tuple[Array, Array]]) -> None:
        """Ingest training pairs into both inventory and RFT analysis."""
        # Build object inventory
        self.inventory.build_from_train_pairs(train_pairs)

        # Perform RFT analysis
        relational_facts = self.rft_analyzer.analyze(train_pairs)

        # Cross-reference: find conflicts between object deltas and RFT facts
        self._detect_conflicts(relational_facts)

        # Queue repair tasks for any conflicts or inconsistencies
        self._queue_repairs()

    def _detect_conflicts(self, relational_facts: Dict[str, List[RelationalFact]]) -> None:
        """Detect conflicts between object history and relational facts."""

        # Conflict Type 1: Object deltas disagree with transformation facts
        self._detect_transformation_conflicts(relational_facts.get('transformation', []))

        # Conflict Type 2: Spatial facts inconsistent across examples
        self._detect_spatial_conflicts(relational_facts.get('spatial', []))

        # Conflict Type 3: Multiple transformations apply to same object
        self._detect_multiple_transformation_conflicts()

    def _detect_transformation_conflicts(self, transformation_facts: List[RelationalFact]) -> None:
        """Detect conflicts in transformation rules."""
        # Group transformations by subject signature
        by_subject = defaultdict(list)
        for fact in transformation_facts:
            by_subject[fact.subject].append(fact)

        # Look for conflicting transformations
        for subject, facts in by_subject.items():
            if len(facts) > 1:
                # Check if all transformations agree
                objects = [f.object for f in facts]
                if len(set(objects)) > 1:
                    # Conflict: same subject transforms to different objects
                    conflict = RuleConflict(
                        conflict_id=f"conflict_{self._next_conflict_id}",
                        conflicting_rules=[f"transform_{i}" for i in range(len(facts))],
                        evidence_for={f"transform_{i}": [str(f.metadata)] for i, f in enumerate(facts)},
                        evidence_against={},
                        severity=0.7,
                        resolution_status='pending'
                    )
                    self._next_conflict_id += 1
                    self.conflicts.append(conflict)

    def _detect_spatial_conflicts(self, spatial_facts: List[RelationalFact]) -> None:
        """Detect conflicts in spatial relationships."""
        # Group by subject-object pairs
        by_pair = defaultdict(list)
        for fact in spatial_facts:
            key = (fact.subject, fact.object)
            by_pair[key].append(fact)

        # Look for inconsistent spatial relations
        for pair, facts in by_pair.items():
            if len(facts) > 1:
                # Check direction consistency
                vectors = [f.direction_vector for f in facts if f.direction_vector is not None]
                if len(vectors) > 1:
                    # Calculate variance in directions
                    avg_vector = np.mean(vectors, axis=0)
                    deviations = [np.linalg.norm(v - avg_vector) for v in vectors]
                    avg_deviation = np.mean(deviations)

                    if avg_deviation > 0.5:  # Threshold for inconsistency
                        conflict = RuleConflict(
                            conflict_id=f"conflict_{self._next_conflict_id}",
                            conflicting_rules=[f"spatial_{i}" for i in range(len(facts))],
                            evidence_for={f"spatial_{i}": [str(f.metadata)] for i, f in enumerate(facts)},
                            evidence_against={},
                            severity=0.5,
                            resolution_status='pending'
                        )
                        self._next_conflict_id += 1
                        self.conflicts.append(conflict)

    def _detect_multiple_transformation_conflicts(self) -> None:
        """Detect when multiple transformations apply to same persistent object."""
        # Group deltas by object_id
        by_object = defaultdict(list)
        for entry in self.inventory.entries.values():
            for delta in entry.transformations:
                by_object[delta.object_id].append(delta)

        # Look for conflicting transformation types
        for obj_id, deltas in by_object.items():
            if len(deltas) > 1:
                trans_types = [d.transformation_type for d in deltas]
                if len(set(trans_types)) > 1:
                    # Multiple transformation types for same object
                    conflict = RuleConflict(
                        conflict_id=f"conflict_{self._next_conflict_id}",
                        conflicting_rules=[f"{obj_id}_{t}" for t in trans_types],
                        evidence_for={t: [str(d.changes)] for d, t in zip(deltas, trans_types)},
                        evidence_against={},
                        severity=0.6,
                        resolution_status='pending',
                        resolution_notes=f"Object {obj_id} has multiple transformation types: {set(trans_types)}"
                    )
                    self._next_conflict_id += 1
                    self.conflicts.append(conflict)

    def _queue_repairs(self) -> None:
        """Queue repair tasks based on detected conflicts and low-confidence rules."""

        # Repair tasks for conflicts
        for conflict in self.conflicts:
            if conflict.resolution_status == 'pending':
                task = RuleRepairTask(
                    task_id=f"repair_{self._next_task_id}",
                    rule_id=conflict.conflict_id,
                    issue_type='conflict',
                    priority=conflict.severity,
                    suggested_fixes=self._suggest_conflict_fixes(conflict),
                    status='queued'
                )
                self._next_task_id += 1
                self.repair_queue.append(task)

        # Repair tasks for low-confidence transformations
        for entry in self.inventory.entries.values():
            for delta in entry.transformations:
                if delta.confidence < 0.6:
                    task = RuleRepairTask(
                        task_id=f"repair_{self._next_task_id}",
                        rule_id=f"{delta.object_id}_{delta.transformation_type}",
                        issue_type='low_confidence',
                        priority=0.4,
                        suggested_fixes=[
                            {'action': 'gather_more_evidence', 'target': delta.object_id},
                            {'action': 'verify_with_llm', 'delta': asdict(delta)}
                        ],
                        status='queued'
                    )
                    self._next_task_id += 1
                    self.repair_queue.append(task)

        # Sort by priority
        self.repair_queue.sort(key=lambda t: t.priority, reverse=True)

    def _suggest_conflict_fixes(self, conflict: RuleConflict) -> List[Dict[str, Any]]:
        """Suggest possible fixes for a conflict."""
        fixes = []

        # Fix 1: Choose rule with most evidence
        evidence_counts = {rule: len(examples) for rule, examples in conflict.evidence_for.items()}
        if evidence_counts:
            best_rule = max(evidence_counts.keys(), key=lambda r: evidence_counts[r])
            fixes.append({
                'action': 'prefer_rule',
                'rule': best_rule,
                'reason': f'Most evidence ({evidence_counts[best_rule]} examples)'
            })

        # Fix 2: Create conditional rule
        fixes.append({
            'action': 'create_conditional',
            'condition': 'context_dependent',
            'rules': list(conflict.conflicting_rules)
        })

        # Fix 3: Request LLM analysis
        fixes.append({
            'action': 'llm_analysis',
            'conflict_details': {
                'rules': conflict.conflicting_rules,
                'evidence': conflict.evidence_for
            }
        })

        return fixes

    def resolve_conflict(self, conflict_id: str, resolution: str, notes: str = "") -> bool:
        """Mark a conflict as resolved."""
        for conflict in self.conflicts:
            if conflict.conflict_id == conflict_id:
                conflict.resolution_status = 'resolved'
                conflict.resolution_notes = notes

                # Update repair tasks
                for task in self.repair_queue:
                    if task.rule_id == conflict_id:
                        task.status = 'completed'

                return True
        return False

    def get_next_repair_task(self) -> Optional[RuleRepairTask]:
        """Get the next highest-priority repair task."""
        for task in self.repair_queue:
            if task.status == 'queued':
                task.status = 'in_progress'
                return task
        return None

    def complete_repair_task(self, task_id: str, success: bool, notes: str = "") -> bool:
        """Mark a repair task as completed."""
        for task in self.repair_queue:
            if task.task_id == task_id:
                task.status = 'completed' if success else 'failed'
                if not success:
                    task.suggested_fixes.append({'action': 'manual_review', 'notes': notes})
                return True
        return False

    def get_tracking_state(self) -> TrackingState:
        """Get complete tracking state for serialization."""
        return TrackingState(
            task_id=self.task_id,
            inventory=self.inventory,
            relational_facts={'facts': self.rft_analyzer.fact_database},
            conflicts=self.conflicts,
            repair_queue=self.repair_queue,
            metadata={
                'num_objects': len(self.inventory.entries),
                'num_conflicts': len(self.conflicts),
                'pending_repairs': sum(1 for t in self.repair_queue if t.status == 'queued')
            }
        )

    def export_for_llm(self) -> Dict[str, Any]:
        """Export tracking state in LLM-friendly format."""
        schema = self.inventory.get_rule_friendly_schema()

        # Add RFT patterns
        rft_patterns = self.rft_analyzer.find_relation_patterns()
        schema['rft_patterns'] = rft_patterns

        # Add conflicts and repair suggestions
        schema['conflicts'] = [
            {
                'id': c.conflict_id,
                'rules': c.conflicting_rules,
                'severity': c.severity,
                'status': c.resolution_status,
                'notes': c.resolution_notes
            }
            for c in self.conflicts
        ]

        schema['repair_tasks'] = [
            {
                'id': t.task_id,
                'rule': t.rule_id,
                'issue': t.issue_type,
                'priority': t.priority,
                'suggested_fixes': t.suggested_fixes,
                'status': t.status
            }
            for t in self.repair_queue
        ]

        return schema

    def save(self, filepath: str) -> None:
        """Save tracking state to file."""
        state = self.get_tracking_state()

        # Convert to serializable format
        data = {
            'task_id': state.task_id,
            'inventory': self.inventory.get_rule_friendly_schema(),
            'relational_facts': [
                {
                    'relation': f.relation,
                    'subject': f.subject,
                    'object': f.object,
                    'metadata': f.metadata,
                    'confidence': f.confidence,
                    'direction_vector': f.direction_vector.tolist() if f.direction_vector is not None else None
                }
                for f in state.relational_facts.get('facts', [])
            ],
            'conflicts': [asdict(c) for c in state.conflicts],
            'repair_queue': [asdict(t) for t in state.repair_queue],
            'metadata': state.metadata,
            'version': state.version
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load tracking state from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.task_id = data['task_id']

        # Load inventory
        inventory_path = Path(filepath).parent / f"{self.task_id}_inventory.json"
        if inventory_path.exists():
            self.inventory.load(str(inventory_path))

        # Reconstruct conflicts
        self.conflicts = [
            RuleConflict(**c) for c in data.get('conflicts', [])
        ]

        # Reconstruct repair queue
        self.repair_queue = [
            RuleRepairTask(**t) for t in data.get('repair_queue', [])
        ]

        # Update ID counters
        if self.conflicts:
            max_id = max(int(c.conflict_id.split('_')[1]) for c in self.conflicts)
            self._next_conflict_id = max_id + 1

        if self.repair_queue:
            max_id = max(int(t.task_id.split('_')[1]) for t in self.repair_queue)
            self._next_task_id = max_id + 1


class TaskTrackingCache:
    """Cache for storing and retrieving tracking states across reruns."""

    def __init__(self, cache_dir: str = ".arc_tracking_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def save_tracking_state(self, tracker: RFTTracker) -> None:
        """Save tracking state to cache."""
        filepath = self.cache_dir / f"{tracker.task_id}.json"
        tracker.save(str(filepath))

        # Also save inventory separately
        inventory_path = self.cache_dir / f"{tracker.task_id}_inventory.json"
        tracker.inventory.save(str(inventory_path))

    def load_tracking_state(self, task_id: str) -> Optional[RFTTracker]:
        """Load tracking state from cache."""
        filepath = self.cache_dir / f"{task_id}.json"

        if not filepath.exists():
            return None

        tracker = RFTTracker(task_id)
        tracker.load(str(filepath))

        return tracker

    def has_cached_state(self, task_id: str) -> bool:
        """Check if tracking state exists in cache."""
        filepath = self.cache_dir / f"{task_id}.json"
        return filepath.exists()

    def clear_cache(self, task_id: Optional[str] = None) -> None:
        """Clear cache for specific task or all tasks."""
        if task_id:
            filepath = self.cache_dir / f"{task_id}.json"
            inventory_path = self.cache_dir / f"{task_id}_inventory.json"

            if filepath.exists():
                filepath.unlink()
            if inventory_path.exists():
                inventory_path.unlink()
        else:
            # Clear all
            for file in self.cache_dir.glob("*.json"):
                file.unlink()
