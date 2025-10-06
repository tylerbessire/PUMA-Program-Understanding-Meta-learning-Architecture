"""
Persistent object inventory and tracking for ARC tasks.

Builds a dictionary of objects across train pairs, tracks attribute deltas,
and maintains persistent IDs for objects that evolve across examples.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json

from .grid import Array
from .object_reasoning import ARCObject, ObjectExtractor, SpatialAnalyzer


@dataclass
class ObjectDelta:
    """Tracks changes to an object between input and output."""
    object_id: str  # Persistent ID across the task
    input_attributes: Dict[str, Any]
    output_attributes: Dict[str, Any]
    changes: Dict[str, Tuple[Any, Any]]  # attribute -> (old_value, new_value)
    transformation_type: str  # 'moved', 'recolored', 'resized', 'deleted', 'created', 'unchanged'
    confidence: float


@dataclass
class ObjectEntry:
    """A persistent entry in the object inventory."""
    persistent_id: str  # Unique ID for this object across the task
    first_seen_pair: int  # Training pair index where first observed
    occurrences: List[Tuple[int, str]]  # [(pair_index, 'input'|'output')]
    canonical_attributes: Dict[str, Any]  # Most common/stable attributes
    attribute_variations: Dict[str, Set[Any]]  # Variations observed across pairs
    transformations: List[ObjectDelta]  # All deltas involving this object
    tags: Set[str]  # Semantic tags: 'placeholder', 'anchor', 'pattern', etc.


class ObjectInventory:
    """Maintains a persistent dictionary of objects across a task."""

    def __init__(self):
        self.extractor = ObjectExtractor()
        self.analyzer = SpatialAnalyzer()
        self.entries: Dict[str, ObjectEntry] = {}  # persistent_id -> ObjectEntry
        self._next_id = 0

    def build_from_train_pairs(self, train_pairs: List[Tuple[Array, Array]]) -> None:
        """Build inventory from training pairs, tracking objects and their evolution."""
        for pair_idx, (input_grid, output_grid) in enumerate(train_pairs):
            self._process_pair(pair_idx, input_grid, output_grid)

    def _process_pair(self, pair_idx: int, input_grid: Array, output_grid: Array) -> None:
        """Process a single training pair, matching and tracking objects."""
        # Extract objects from both grids
        input_objects = self.extractor.extract_objects(input_grid)
        output_objects = self.extractor.extract_objects(output_grid)

        # Match input objects to existing inventory entries
        input_matches = self._match_objects_to_inventory(
            input_objects, pair_idx, 'input', input_grid
        )

        # Match output objects to existing inventory entries
        output_matches = self._match_objects_to_inventory(
            output_objects, pair_idx, 'output', output_grid
        )

        # Find deltas between matched input/output objects
        deltas = self._compute_deltas(
            input_objects, output_objects, input_matches, output_matches, pair_idx
        )

        # Update inventory with deltas
        for delta in deltas:
            if delta.object_id in self.entries:
                self.entries[delta.object_id].transformations.append(delta)

    def _match_objects_to_inventory(
        self,
        objects: List[ARCObject],
        pair_idx: int,
        stage: str,
        grid: Array
    ) -> Dict[int, str]:
        """Match extracted objects to persistent inventory entries.

        Returns:
            Mapping from object.id (local) to persistent_id (inventory)
        """
        matches: Dict[int, str] = {}

        for obj in objects:
            # Try to find existing entry that matches this object
            persistent_id = self._find_matching_entry(obj, pair_idx, stage, grid)

            if persistent_id:
                # Update existing entry
                entry = self.entries[persistent_id]
                entry.occurrences.append((pair_idx, stage))
                self._update_entry_attributes(entry, obj)
            else:
                # Create new entry
                persistent_id = self._create_new_entry(obj, pair_idx, stage)

            matches[obj.id] = persistent_id

        return matches

    def _find_matching_entry(
        self,
        obj: ARCObject,
        pair_idx: int,
        stage: str,
        grid: Array
    ) -> Optional[str]:
        """Find existing inventory entry that matches this object."""

        # Compute signature for this object
        signature = self._compute_object_signature(obj, grid)

        # Find entries with compatible signatures
        candidates = []
        for pid, entry in self.entries.items():
            # Check if attributes are compatible
            if self._signatures_compatible(signature, entry.canonical_attributes):
                # Calculate similarity score
                score = self._compute_match_score(obj, entry)
                candidates.append((pid, score))

        # Return best match if score is high enough
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_id, best_score = candidates[0]

            if best_score > 0.7:  # Threshold for accepting a match
                return best_id

        return None

    def _compute_object_signature(self, obj: ARCObject, grid: Array) -> Dict[str, Any]:
        """Compute a signature dictionary for an object."""
        return {
            'color': obj.color,
            'shape_type': obj.shape_type,
            'size': obj.size,
            'width': obj.width,
            'height': obj.height,
            'aspect_ratio': obj.width / obj.height if obj.height > 0 else 0,
            'symmetry_h': obj.descriptors.get('symmetry_horizontal', False),
            'symmetry_v': obj.descriptors.get('symmetry_vertical', False),
            'touches_border': obj.descriptors.get('touches_border', False),
        }

    def _signatures_compatible(self, sig1: Dict[str, Any], sig2: Dict[str, Any]) -> bool:
        """Check if two signatures are compatible (could be same object)."""
        # Core attributes must match
        if sig1.get('shape_type') != sig2.get('shape_type'):
            return False

        # Size can vary slightly
        size_diff = abs(sig1.get('size', 0) - sig2.get('size', 0))
        if size_diff > 5:  # Allow small variations
            return False

        # Color often changes, so don't require match
        return True

    def _compute_match_score(self, obj: ARCObject, entry: ObjectEntry) -> float:
        """Compute similarity score between object and inventory entry."""
        score = 0.0
        total_weight = 0.0

        canon = entry.canonical_attributes

        # Shape type (high weight)
        if obj.shape_type == canon.get('shape_type'):
            score += 3.0
        total_weight += 3.0

        # Size (medium weight)
        size_diff = abs(obj.size - canon.get('size', 0))
        size_similarity = max(0, 1.0 - size_diff / max(obj.size, canon.get('size', 1)))
        score += 2.0 * size_similarity
        total_weight += 2.0

        # Color (low weight, often changes)
        if obj.color == canon.get('color'):
            score += 0.5
        total_weight += 0.5

        # Position similarity (medium weight)
        # Check if object appears in similar relative position
        pos_score = self._compute_position_similarity(obj, entry)
        score += 1.5 * pos_score
        total_weight += 1.5

        return score / total_weight if total_weight > 0 else 0.0

    def _compute_position_similarity(self, obj: ARCObject, entry: ObjectEntry) -> float:
        """Compute positional similarity score."""
        # Simple heuristic: check if object is in similar region
        # More sophisticated version would track typical positions

        # For now, just check if it touches the same borders
        obj_borders = set()
        if obj.descriptors.get('distance_top', 100) == 0:
            obj_borders.add('top')
        if obj.descriptors.get('distance_left', 100) == 0:
            obj_borders.add('left')
        if obj.descriptors.get('distance_bottom', 100) == 0:
            obj_borders.add('bottom')
        if obj.descriptors.get('distance_right', 100) == 0:
            obj_borders.add('right')

        canon = entry.canonical_attributes
        entry_borders = set()
        if canon.get('distance_top', 100) == 0:
            entry_borders.add('top')
        if canon.get('distance_left', 100) == 0:
            entry_borders.add('left')
        if canon.get('distance_bottom', 100) == 0:
            entry_borders.add('bottom')
        if canon.get('distance_right', 100) == 0:
            entry_borders.add('right')

        if not obj_borders and not entry_borders:
            return 0.5  # Neither touches borders

        if obj_borders == entry_borders:
            return 1.0  # Same border pattern

        # Partial overlap
        overlap = len(obj_borders & entry_borders)
        union = len(obj_borders | entry_borders)
        return overlap / union if union > 0 else 0.0

    def _create_new_entry(self, obj: ARCObject, pair_idx: int, stage: str) -> str:
        """Create a new inventory entry for this object."""
        persistent_id = f"obj_{self._next_id}"
        self._next_id += 1

        signature = self._compute_object_signature(obj, np.zeros((1, 1)))  # Grid not needed for new entry

        entry = ObjectEntry(
            persistent_id=persistent_id,
            first_seen_pair=pair_idx,
            occurrences=[(pair_idx, stage)],
            canonical_attributes=signature,
            attribute_variations={k: {v} for k, v in signature.items()},
            transformations=[],
            tags=self._infer_initial_tags(obj)
        )

        self.entries[persistent_id] = entry
        return persistent_id

    def _update_entry_attributes(self, entry: ObjectEntry, obj: ARCObject) -> None:
        """Update entry with new observation of the object."""
        signature = self._compute_object_signature(obj, np.zeros((1, 1)))

        # Track variations
        for key, value in signature.items():
            if key not in entry.attribute_variations:
                entry.attribute_variations[key] = set()
            entry.attribute_variations[key].add(value)

        # Update canonical attributes (use most common values)
        # For now, simple average/mode
        for key in signature:
            variations = entry.attribute_variations.get(key, set())
            if len(variations) == 1:
                entry.canonical_attributes[key] = list(variations)[0]
            # For numeric values, use average
            elif all(isinstance(v, (int, float)) for v in variations):
                entry.canonical_attributes[key] = sum(variations) / len(variations)

    def _infer_initial_tags(self, obj: ARCObject) -> Set[str]:
        """Infer semantic tags for a new object."""
        tags = set()

        # Shape-based tags
        if obj.shape_type == 'rectangle' and obj.size > 10:
            tags.add('placeholder')

        if obj.shape_type == 'single':
            tags.add('marker')

        # Position-based tags
        if obj.descriptors.get('touches_border'):
            tags.add('border_object')
        else:
            tags.add('interior_object')

        # Pattern tags
        if obj.descriptors.get('symmetry_horizontal') or obj.descriptors.get('symmetry_vertical'):
            tags.add('symmetric')

        return tags

    def _compute_deltas(
        self,
        input_objects: List[ARCObject],
        output_objects: List[ARCObject],
        input_matches: Dict[int, str],
        output_matches: Dict[int, str],
        pair_idx: int
    ) -> List[ObjectDelta]:
        """Compute deltas between input and output objects."""
        deltas = []

        # Create reverse mapping: persistent_id -> local object
        input_by_pid = {pid: obj for obj in input_objects for obj_id, pid in input_matches.items() if obj.id == obj_id}
        output_by_pid = {pid: obj for obj in output_objects for obj_id, pid in output_matches.items() if obj.id == obj_id}

        # Find all persistent IDs that appear in either input or output
        all_pids = set(input_matches.values()) | set(output_matches.values())

        for pid in all_pids:
            input_obj = input_by_pid.get(pid)
            output_obj = output_by_pid.get(pid)

            if input_obj and output_obj:
                # Object exists in both: compute changes
                delta = self._compute_object_delta(pid, input_obj, output_obj, pair_idx)
                deltas.append(delta)

            elif input_obj and not output_obj:
                # Object deleted
                delta = ObjectDelta(
                    object_id=pid,
                    input_attributes=self._compute_object_signature(input_obj, np.zeros((1, 1))),
                    output_attributes={},
                    changes={},
                    transformation_type='deleted',
                    confidence=0.9
                )
                deltas.append(delta)

            elif output_obj and not input_obj:
                # Object created
                delta = ObjectDelta(
                    object_id=pid,
                    input_attributes={},
                    output_attributes=self._compute_object_signature(output_obj, np.zeros((1, 1))),
                    changes={},
                    transformation_type='created',
                    confidence=0.9
                )
                deltas.append(delta)

        return deltas

    def _compute_object_delta(
        self,
        pid: str,
        input_obj: ARCObject,
        output_obj: ARCObject,
        pair_idx: int
    ) -> ObjectDelta:
        """Compute delta for an object that exists in both input and output."""
        input_sig = self._compute_object_signature(input_obj, np.zeros((1, 1)))
        output_sig = self._compute_object_signature(output_obj, np.zeros((1, 1)))

        changes = {}
        for key in set(input_sig.keys()) | set(output_sig.keys()):
            input_val = input_sig.get(key)
            output_val = output_sig.get(key)

            if input_val != output_val:
                changes[key] = (input_val, output_val)

        # Classify transformation type
        transformation_type = self._classify_transformation(changes, input_obj, output_obj)

        return ObjectDelta(
            object_id=pid,
            input_attributes=input_sig,
            output_attributes=output_sig,
            changes=changes,
            transformation_type=transformation_type,
            confidence=0.85
        )

    def _classify_transformation(
        self,
        changes: Dict[str, Tuple[Any, Any]],
        input_obj: ARCObject,
        output_obj: ARCObject
    ) -> str:
        """Classify the type of transformation based on changes."""
        if not changes:
            return 'unchanged'

        if 'color' in changes and len(changes) == 1:
            return 'recolored'

        if 'size' in changes or 'width' in changes or 'height' in changes:
            return 'resized'

        # Check for movement (positions changed)
        if input_obj.bounding_box != output_obj.bounding_box:
            # If size and color unchanged, it's a move
            if input_obj.size == output_obj.size and input_obj.color == output_obj.color:
                return 'moved'

        # Multiple changes
        return 'transformed'

    def get_rule_friendly_schema(self) -> Dict[str, Any]:
        """Export inventory in a rule-friendly format for LLM and heuristics."""
        schema = {
            'objects': {},
            'patterns': [],
            'transformation_rules': []
        }

        # Export object entries
        for pid, entry in self.entries.items():
            schema['objects'][pid] = {
                'id': pid,
                'first_seen': entry.first_seen_pair,
                'occurrences': entry.occurrences,
                'attributes': entry.canonical_attributes,
                'variations': {k: list(v) for k, v in entry.attribute_variations.items()},
                'tags': list(entry.tags)
            }

        # Extract common patterns
        patterns = self._extract_patterns()
        schema['patterns'] = patterns

        # Extract transformation rules
        rules = self._extract_transformation_rules()
        schema['transformation_rules'] = rules

        return schema

    def _extract_patterns(self) -> List[Dict[str, Any]]:
        """Extract common patterns from the inventory."""
        patterns = []

        # Pattern 1: Objects that always transform the same way
        consistent_transformers = defaultdict(list)
        for pid, entry in self.entries.items():
            if len(entry.transformations) >= 2:
                # Check if transformations are consistent
                trans_types = [t.transformation_type for t in entry.transformations]
                if len(set(trans_types)) == 1:  # All same type
                    consistent_transformers[trans_types[0]].append(pid)

        for trans_type, pids in consistent_transformers.items():
            if len(pids) >= 2:
                patterns.append({
                    'type': 'consistent_transformation',
                    'transformation': trans_type,
                    'objects': pids,
                    'confidence': 0.8
                })

        # Pattern 2: Objects with stable attributes
        stable_objects = []
        for pid, entry in self.entries.items():
            # Check if all occurrences have same attributes
            stable = True
            for attr, variations in entry.attribute_variations.items():
                if len(variations) > 1:
                    stable = False
                    break

            if stable and len(entry.occurrences) >= 2:
                stable_objects.append(pid)

        if stable_objects:
            patterns.append({
                'type': 'stable_objects',
                'objects': stable_objects,
                'confidence': 0.9
            })

        return patterns

    def _extract_transformation_rules(self) -> List[Dict[str, Any]]:
        """Extract transformation rules from deltas."""
        rules = []

        # Group deltas by transformation type
        by_type = defaultdict(list)
        for entry in self.entries.values():
            for delta in entry.transformations:
                by_type[delta.transformation_type].append(delta)

        # Create rules for each transformation type
        for trans_type, deltas in by_type.items():
            if len(deltas) >= 2:  # Need multiple examples
                # Find common change patterns
                common_changes = self._find_common_changes(deltas)

                if common_changes:
                    rules.append({
                        'transformation': trans_type,
                        'common_changes': common_changes,
                        'example_count': len(deltas),
                        'confidence': min(0.95, 0.5 + 0.1 * len(deltas))
                    })

        return rules

    def _find_common_changes(self, deltas: List[ObjectDelta]) -> Dict[str, Any]:
        """Find changes that are common across multiple deltas."""
        if not deltas:
            return {}

        # Count attribute changes
        change_counts = defaultdict(lambda: defaultdict(int))

        for delta in deltas:
            for attr, (old_val, new_val) in delta.changes.items():
                change_counts[attr][(old_val, new_val)] += 1

        # Find majority changes
        common = {}
        for attr, value_counts in change_counts.items():
            # Find most common change
            if value_counts:
                most_common = max(value_counts.items(), key=lambda x: x[1])
                if most_common[1] >= len(deltas) * 0.5:  # Appears in >50% of deltas
                    common[attr] = {
                        'from': most_common[0][0],
                        'to': most_common[0][1],
                        'frequency': most_common[1] / len(deltas)
                    }

        return common

    def save(self, filepath: str) -> None:
        """Save inventory to JSON file."""
        schema = self.get_rule_friendly_schema()

        with open(filepath, 'w') as f:
            json.dump(schema, f, indent=2, default=str)

    def load(self, filepath: str) -> None:
        """Load inventory from JSON file."""
        with open(filepath, 'r') as f:
            schema = json.load(f)

        # Reconstruct entries from schema
        self.entries = {}
        for pid, obj_data in schema.get('objects', {}).items():
            entry = ObjectEntry(
                persistent_id=pid,
                first_seen_pair=obj_data['first_seen'],
                occurrences=obj_data['occurrences'],
                canonical_attributes=obj_data['attributes'],
                attribute_variations={k: set(v) for k, v in obj_data.get('variations', {}).items()},
                transformations=[],
                tags=set(obj_data.get('tags', []))
            )
            self.entries[pid] = entry

        # Update next_id
        if self.entries:
            max_id = max(int(pid.split('_')[1]) for pid in self.entries.keys())
            self._next_id = max_id + 1
