"""Relational Frame Theory utilities for ARC reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

from .object_reasoning import ObjectExtractor

# Directional vectors for spatial reasoning
DIRECTION_VECTORS = {
    "up": np.array([0, -1]),
    "down": np.array([0, 1]),
    "left": np.array([-1, 0]),
    "right": np.array([1, 0]),
    "up_left": np.array([-1, -1]),
    "up_right": np.array([1, -1]),
    "down_left": np.array([-1, 1]),
    "down_right": np.array([1, 1])
}

# Opposition relationships
OPPOSITIONS = {
    "up": "down",
    "down": "up", 
    "left": "right",
    "right": "left",
    "up_left": "down_right",
    "down_right": "up_left",
    "up_right": "down_left", 
    "down_left": "up_right"
}


@dataclass
class RelationalFact:
    relation: str
    subject: Tuple[int, int, int]  # (color, height, width)
    object: Tuple[int, int, int]   # (color, height, width)
    metadata: Dict[str, float]
    direction_vector: Optional[np.ndarray] = None  # For spatial relations
    confidence: float = 1.0

    def get_spatial_relation(self) -> Optional[str]:
        """Get the spatial relation name from direction vector."""
        if self.direction_vector is None:
            return None
        for name, vec in DIRECTION_VECTORS.items():
            if np.allclose(self.direction_vector, vec, atol=0.1):
                return name
        return None

    def get_opposition(self) -> Optional[str]:
        """Get the opposite spatial relation."""
        spatial = self.get_spatial_relation()
        return OPPOSITIONS.get(spatial) if spatial else None


class RelationalFrameAnalyzer:
    """Extract relational facts (spatial, contextual, logical) from grids."""

    def __init__(self) -> None:
        self._extractor = ObjectExtractor()
        self.fact_database: List[RelationalFact] = []

    def analyze(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, List[RelationalFact]]:
        facts: Dict[str, List[RelationalFact]] = {
            "alignment": [],
            "containment": [],
            "symmetry": [],
            "spatial": [],
            "transformation": []
        }

        for inp, out in train_pairs:
            self._record_spatial_relations(inp, out, facts)
            self._record_alignment(inp, out, facts)
            self._record_alignment(out, inp, facts)
            self._record_containment(inp, out, facts)
            self._record_symmetry(out, facts)
            self._record_transformations(inp, out, facts)

        # Derive composite and inverse relations to enrich reasoning graph
        composite_facts: List[RelationalFact] = []
        inverse_facts: List[RelationalFact] = []

        base_lists: List[RelationalFact] = []
        base_lists.extend(facts.get("spatial", []))
        base_lists.extend(facts.get("transformation", []))

        filtered_base = self._filter_facts_for_derivation(base_lists)

        for first in filtered_base:
            for second in filtered_base:
                derived = self.derive_composite_relations(first, second)
                if derived is not None:
                    composite_facts.append(derived)

            opposite = self._derive_inverse_relation(first)
            if opposite is not None:
                inverse_facts.append(opposite)

        if composite_facts:
            facts.setdefault("composite", []).extend(composite_facts)
        if inverse_facts:
            facts.setdefault("inverse", []).extend(inverse_facts)

        # Store facts for compositional reasoning
        self.fact_database = []
        for fact_list in facts.values():
            self.fact_database.extend(fact_list)

        return facts

    def derive_composite_relations(self, fact1: RelationalFact, fact2: RelationalFact) -> Optional[RelationalFact]:
        """Derive new relations through transitivity and composition."""
        # Transitivity: if A relates to B and B relates to C, derive A->C relation
        if (fact1.object == fact2.subject and 
            fact1.direction_vector is not None and 
            fact2.direction_vector is not None):
            
            # Compose direction vectors
            composite_vector = fact1.direction_vector + fact2.direction_vector
            
            return RelationalFact(
                relation="composite_spatial",
                subject=fact1.subject,
                object=fact2.object,
                metadata={
                    "derived_from": f"{fact1.relation}+{fact2.relation}",
                    "composite_distance": float(np.linalg.norm(composite_vector))
                },
                direction_vector=composite_vector,
                confidence=min(fact1.confidence, fact2.confidence) * 0.8  # Reduce confidence for derived facts
            )
        return None

    def find_relation_patterns(self) -> List[Dict[str, Any]]:
        """Find recurring patterns in relational facts."""
        patterns = []

        # Group facts by relation type
        relation_groups = {}
        for fact in self.fact_database:
            relation_groups.setdefault(fact.relation, []).append(fact)
        
        # Look for consistent spatial transformations
        for relation_type, fact_list in relation_groups.items():
            if len(fact_list) > 1 and relation_type == "composite_spatial":
                # Find consistent direction patterns
                vectors = [f.direction_vector for f in fact_list if f.direction_vector is not None]
                if vectors:
                    avg_vector = np.mean(vectors, axis=0)
                    if np.linalg.norm(avg_vector) == 0:
                        continue
                    consistency = np.mean([np.dot(v, avg_vector) / (np.linalg.norm(v) * np.linalg.norm(avg_vector)) 
                                         for v in vectors])
                    if consistency > 0.8:  # High consistency threshold
                        patterns.append({
                            "type": "consistent_spatial_transform",
                            "direction": avg_vector,
                            "consistency": consistency,
                            "count": len(vectors)
                        })
        
        return patterns

    def _record_spatial_relations(
        self,
        source: np.ndarray,
        target: np.ndarray,
        facts: Dict[str, List[RelationalFact]],
    ) -> None:
        """Record spatial relationships between objects across input/output."""
        source_objs = self._extractor.extract_objects(source)
        target_objs = self._extractor.extract_objects(target)
        
        for src_obj in source_objs:
            src_center = self._get_object_center(src_obj)
            src_signature = (int(src_obj.color), 
                           src_obj.bounding_box[2] - src_obj.bounding_box[0] + 1,
                           src_obj.bounding_box[3] - src_obj.bounding_box[1] + 1)
            
            for tgt_obj in target_objs:
                tgt_center = self._get_object_center(tgt_obj)
                tgt_signature = (int(tgt_obj.color),
                               tgt_obj.bounding_box[2] - tgt_obj.bounding_box[0] + 1,
                               tgt_obj.bounding_box[3] - tgt_obj.bounding_box[1] + 1)
                
                # Calculate direction vector
                direction = np.array([tgt_center[1] - src_center[1], tgt_center[0] - src_center[0]])
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)  # Normalize
                    
                    fact = RelationalFact(
                        relation="spatial_transform",
                        subject=src_signature,
                        object=tgt_signature,
                        metadata={
                            "distance": float(np.linalg.norm(np.array(tgt_center) - np.array(src_center))),
                            "src_center": src_center,
                            "tgt_center": tgt_center
                        },
                        direction_vector=direction
                    )
                    facts.setdefault("spatial", []).append(fact)

    def _record_transformations(
        self,
        source: np.ndarray,
        target: np.ndarray, 
        facts: Dict[str, List[RelationalFact]],
    ) -> None:
        """Record object transformations between input and output."""
        source_objs = self._extractor.extract_objects(source)
        target_objs = self._extractor.extract_objects(target)

        grid_norm = max(source.shape[0], source.shape[1], 1)

        for src_obj in source_objs:
            src_height = src_obj.bounding_box[2] - src_obj.bounding_box[0] + 1
            src_width = src_obj.bounding_box[3] - src_obj.bounding_box[1] + 1
            src_sig = (
                int(src_obj.color),
                src_height,
                src_width,
            )

            src_center = self._get_object_center(src_obj)

            best_match = None
            best_score = -1.0

            for tgt_obj in target_objs:
                tgt_height = tgt_obj.bounding_box[2] - tgt_obj.bounding_box[0] + 1
                tgt_width = tgt_obj.bounding_box[3] - tgt_obj.bounding_box[1] + 1
                tgt_sig = (
                    int(tgt_obj.color),
                    tgt_height,
                    tgt_width,
                )

                # Compute similarity scores for color, size, and spatial distance
                color_score = 1.0 if src_sig[0] == tgt_sig[0] else 0.0

                if src_height == 0 or src_width == 0:
                    size_similarity = 0.0
                else:
                    height_ratio = min(src_height, tgt_height) / max(src_height, tgt_height)
                    width_ratio = min(src_width, tgt_width) / max(src_width, tgt_width)
                    size_similarity = 0.5 * (height_ratio + width_ratio)

                tgt_center = self._get_object_center(tgt_obj)
                offset = np.array(tgt_center) - np.array(src_center)
                distance = np.linalg.norm(offset) / grid_norm
                distance_score = max(0.0, 1.0 - distance)

                match_score = (1.2 * color_score) + size_similarity + (0.8 * distance_score)

                # Penalize drastically different shapes/colors
                if color_score == 0.0 and size_similarity < 0.6:
                    match_score -= 0.5

                if match_score > best_score:
                    best_score = match_score
                    best_match = (tgt_obj, tgt_sig, offset, distance, size_similarity)

            if best_match and best_score > 0.5:
                tgt_obj, tgt_sig, offset, distance, size_similarity = best_match

                if np.linalg.norm(offset) > 0:
                    direction = offset / np.linalg.norm(offset)
                else:
                    direction = None

                fact = RelationalFact(
                    relation="object_transformation",
                    subject=src_sig,
                    object=tgt_sig,
                    metadata={
                        "translation": (float(offset[1]), float(offset[0])),
                        "match_score": float(best_score),
                        "distance": float(distance),
                        "size_similarity": float(size_similarity),
                    },
                    direction_vector=direction if direction is not None else None,
                )
                facts.setdefault("transformation", []).append(fact)

    def _get_object_center(self, obj) -> Tuple[int, int]:
        """Get the center coordinates of an object."""
        bbox = obj.bounding_box
        center_r = (bbox[0] + bbox[2]) // 2
        center_c = (bbox[1] + bbox[3]) // 2
        return (center_r, center_c)

    def _record_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        facts: Dict[str, List[RelationalFact]],
    ) -> None:
        objs = self._extractor.extract_objects(source)
        for obj in objs:
            bbox = obj.bounding_box
            height = bbox[2] - bbox[0] + 1
            width = bbox[3] - bbox[1] + 1
            subject = (int(obj.color), height, width)
            target_bbox = self._find_matching_region(target, self._object_mask(obj))
            if target_bbox is None:
                continue
            relation = RelationalFact(
                relation="aligned",
                subject=subject,
                object=(int(target_bbox[0]), int(target_bbox[1]), int(target_bbox[2])),
                metadata={
                    "offset_row": float(target_bbox[0] - bbox[0]),
                    "offset_col": float(target_bbox[1] - bbox[1]),
                },
            )
            facts.setdefault("alignment", []).append(relation)

    def _record_containment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        facts: Dict[str, List[RelationalFact]],
    ) -> None:
        source_bg = int(self._background_color(source))
        target_bg = int(self._background_color(target))
        if source_bg != target_bg:
            fact = RelationalFact(
                relation="background_shift",
                subject=(source_bg, 0, 0),
                object=(target_bg, 0, 0),
                metadata={},
            )
            facts.setdefault("containment", []).append(fact)

    def _record_symmetry(self, grid: np.ndarray, facts: Dict[str, List[RelationalFact]]) -> None:
        if np.array_equal(grid, np.fliplr(grid)):
            facts.setdefault("symmetry", []).append(
                RelationalFact(relation="mirror_vertical", subject=(1, 0, 0), object=(1, 0, 0), metadata={})
            )
        if np.array_equal(grid, np.flipud(grid)):
            facts.setdefault("symmetry", []).append(
                RelationalFact(relation="mirror_horizontal", subject=(1, 0, 0), object=(1, 0, 0), metadata={})
            )

    def _find_matching_region(self, grid: np.ndarray, pattern: np.ndarray) -> Tuple[int, int, int] | None:
        h, w = grid.shape
        ph, pw = pattern.shape
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                window = grid[r : r + ph, c : c + pw]
                if np.array_equal(window, pattern):
                    return (r, c, ph * pw)
        return None

    @staticmethod
    def _background_color(grid: np.ndarray) -> int:
        values, counts = np.unique(grid, return_counts=True)
        return int(values[counts.argmax()])

    @staticmethod
    def _object_mask(obj) -> np.ndarray:
        min_r, min_c, max_r, max_c = obj.bounding_box
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        mask = np.zeros((height, width), dtype=np.int16)
        for r, c in obj.positions:
            mask[r - min_r, c - min_c] = obj.color
        return mask

    def _derive_inverse_relation(self, fact: RelationalFact) -> Optional[RelationalFact]:
        if fact.direction_vector is None:
            return None

        inverse_vector = -fact.direction_vector
        translation = fact.metadata.get("translation")
        if translation and isinstance(translation, tuple) and len(translation) == 2:
            inverse_translation = (-translation[0], -translation[1])
        else:
            inverse_translation = None

        return RelationalFact(
            relation="opposite_spatial",
            subject=fact.object,
            object=fact.subject,
            metadata={
                "derived_from": fact.relation,
                "translation": inverse_translation,
                "match_score": fact.metadata.get("match_score", fact.confidence) * 0.7,
            },
            direction_vector=inverse_vector,
            confidence=fact.confidence * 0.7,
        )

    def _filter_facts_for_derivation(
        self,
        facts_list: List[RelationalFact],
        max_per_subject: int = 3,
        max_total: int = 200,
    ) -> List[RelationalFact]:
        if not facts_list:
            return []

        grouped: Dict[Tuple[int, int, int], List[RelationalFact]] = {}
        for fact in facts_list:
            grouped.setdefault(fact.subject, []).append(fact)

        filtered: List[RelationalFact] = []
        for fact_list in grouped.values():
            fact_list.sort(
                key=lambda f: f.metadata.get("match_score", f.confidence) if f.metadata else f.confidence,
                reverse=True,
            )
            filtered.extend(fact_list[:max_per_subject])

        if len(filtered) > max_total:
            filtered.sort(
                key=lambda f: f.metadata.get("match_score", f.confidence) if f.metadata else f.confidence,
                reverse=True,
            )
            filtered = filtered[:max_total]

        return filtered
