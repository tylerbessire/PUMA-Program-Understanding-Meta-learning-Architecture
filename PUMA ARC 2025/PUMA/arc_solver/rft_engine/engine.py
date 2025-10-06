"""Lightweight Relational Frame Theory inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from ..object_reasoning import ObjectExtractor, SpatialAnalyzer
from ..grid import Array


@dataclass
class RelationalFact:
    """Single relation between two abstract stimuli."""

    source: str
    target: str
    frame: str
    context: str
    confidence: float
    metadata: Dict[str, Union[float, int]] = field(default_factory=dict)

    def mirrored(self) -> "RelationalFact":
        """Return the relation with source/target swapped (mutual entailment)."""

        return RelationalFact(
            source=self.target,
            target=self.source,
            frame=self.frame,
            context=self.context,
            confidence=self.confidence,
            metadata=self.metadata.copy(),
        )

    def compose_with(self, other: "RelationalFact") -> Optional["RelationalFact"]:
        """Return a composed relation if frames are compatible (combinatorial entailment).

        Currently we only support composition of translation comparisons by summing
        their offsets.  Returns ``None`` if the composition is not applicable.
        """

        if (
            self.frame == other.frame == "comparison"
            and self.context.startswith("translation")
            and other.context.startswith("translation")
            and self.target == other.source
        ):
            dr1 = self.metadata.get("dr")
            dc1 = self.metadata.get("dc")
            dr2 = other.metadata.get("dr")
            dc2 = other.metadata.get("dc")
            if dr1 is None or dc1 is None or dr2 is None or dc2 is None:
                return None
            combined = (dr1 + dr2, dc1 + dc2)
            return RelationalFact(
                source=self.source,
                target=other.target,
                frame="comparison",
                context=f"translation:{combined[0]}:{combined[1]}",
                confidence=min(self.confidence, other.confidence),
                metadata={"dr": combined[0], "dc": combined[1]},
            )
        return None


@dataclass
class RFTInference:
    """Inference bundle returned by :class:`RFTEngine`."""

    relations: List[RelationalFact]
    function_hints: Dict[str, Set[str]]

    def estimate_behavioural_signal(self, program: Sequence[Tuple[str, Dict[str, int]]]) -> float:
        """Return bonus proportional to overlap between hints and program ops."""

        if not program or not self.function_hints:
            return 0.0
        program_ops = {op for op, _ in program}
        hinted_ops: Set[str] = set()
        for hints in self.function_hints.values():
            hinted_ops.update(hints)
        if not hinted_ops:
            return 0.0
        matched = len(program_ops & hinted_ops)
        return min(1.0, matched / max(1, len(hinted_ops)))


class RFTEngine:
    """Extracts relational frames and functional hints from training pairs."""

    def __init__(self) -> None:
        self._extractor = ObjectExtractor()
        self._spatial = SpatialAnalyzer()

    def analyse(self, train_pairs: Iterable[Tuple[Array, Array]]) -> RFTInference:
        relations: List[RelationalFact] = []
        function_hints: Dict[str, Set[str]] = {}

        for idx, (inp, out) in enumerate(train_pairs):
            objects_in = self._extractor.extract_objects(inp)
            objects_out = self._extractor.extract_objects(out)
            mapping = self._match_objects(objects_in, objects_out)

            for obj_in in objects_in:
                node_id = self._node_id(idx, "input", obj_in.id)
                function_hints.setdefault(node_id, set())
                if obj_in.id in mapping:
                    out_obj = mapping[obj_in.id]
                    target_id = self._node_id(idx, "output", out_obj.id)
                    relations.extend(self._derive_relations(node_id, target_id, obj_in, out_obj))
                    hints = self._derive_function_hints(obj_in, out_obj)
                    if hints:
                        function_hints[node_id].update(hints)
                        function_hints.setdefault(target_id, set()).update(hints)
                else:
                    function_hints[node_id].add("delete")

            for obj_out in objects_out:
                node_id = self._node_id(idx, "output", obj_out.id)
                function_hints.setdefault(node_id, set())
                if obj_out.id not in mapping.values():
                    function_hints[node_id].add("create")

            relations.extend(self._spatial_relations(idx, objects_in, objects_out))

        self._apply_entailment(relations)
        self._transform_functions(relations, function_hints)

        return RFTInference(relations=relations, function_hints=function_hints)

    # ------------------------------------------------------------------
    # Relational extraction helpers
    # ------------------------------------------------------------------
    def _match_objects(
        self,
        inputs: Sequence,
        outputs: Sequence,
    ) -> Dict[int, object]:
        matches: Dict[int, object] = {}
        used: Set[int] = set()
        for obj in inputs:
            best_idx: Optional[int] = None
            best_score = -1.0
            for cand in outputs:
                if cand.id in used:
                    continue
                score = self._shape_similarity(obj, cand)
                if score > best_score:
                    best_score = score
                    best_idx = cand.id
            if best_idx is not None and best_score > 0.3:
                used.add(best_idx)
                matches[obj.id] = next(c for c in outputs if c.id == best_idx)
        return matches

    def _shape_similarity(self, obj_a, obj_b) -> float:
        size_ratio = min(obj_a.size, obj_b.size) / max(obj_a.size, obj_b.size)
        if size_ratio == 0:
            return 0.0
        width_ratio = min(obj_a.width, obj_b.width) / max(obj_a.width, obj_b.width)
        height_ratio = min(obj_a.height, obj_b.height) / max(obj_a.height, obj_b.height)
        shape_bonus = 1.0 if obj_a.shape_type == obj_b.shape_type else 0.0
        return 0.5 * size_ratio + 0.2 * width_ratio + 0.2 * height_ratio + 0.1 * shape_bonus

    def _derive_relations(self, source_id: str, target_id: str, obj_in, obj_out) -> List[RelationalFact]:
        relations: List[RelationalFact] = []
        source_desc = obj_in.descriptors or {}
        target_desc = obj_out.descriptors or {}
        color_metadata = {"color_in": obj_in.color, "color_out": obj_out.color}
        color_metadata.update(
            {
                "shape_in": obj_in.shape_type,
                "shape_out": obj_out.shape_type,
                "border_palette_in": tuple(source_desc.get("border_colors", [])),
                "border_palette_out": tuple(target_desc.get("border_colors", [])),
                "touches_border_in": int(bool(source_desc.get("touches_border"))),
                "touches_border_out": int(bool(target_desc.get("touches_border"))),
            }
        )
        if obj_in.color == obj_out.color:
            relations.append(
                RelationalFact(
                    source=source_id,
                    target=target_id,
                    frame="coordination",
                    context="color",
                    confidence=0.9,
                    metadata=color_metadata,
                )
            )
        else:
            relations.append(
                RelationalFact(
                    source=source_id,
                    target=target_id,
                    frame="opposition",
                    context="color",
                    confidence=0.8,
                    metadata=color_metadata,
                )
            )
        translation = self._translation_vector(obj_in, obj_out)
        if translation is not None:
            relations.append(
                RelationalFact(
                    source_id,
                    target_id,
                    "comparison",
                    f"translation:{translation[0]}:{translation[1]}",
                    0.85,
                    metadata={
                        "dr": translation[0],
                        "dc": translation[1],
                        "width": obj_in.width,
                        "height": obj_in.height,
                        "shape_in": obj_in.shape_type,
                        "shape_out": obj_out.shape_type,
                    },
                )
            )
        return relations

    def _derive_function_hints(self, obj_in, obj_out) -> Set[str]:
        hints: Set[str] = set()
        if obj_in.color != obj_out.color:
            hints.add("recolor")
        translation = self._translation_vector(obj_in, obj_out)
        if translation and translation != (0, 0):
            hints.add("translate")
        if obj_in.size != obj_out.size:
            hints.add("resize")

        source_desc = obj_in.descriptors or {}
        target_desc = obj_out.descriptors or {}

        if source_desc.get("touches_border") and target_desc.get("touches_border"):
            hints.add("preserve_border")
        if source_desc.get("symmetry_horizontal") or target_desc.get("symmetry_horizontal"):
            hints.add("symmetry_horizontal")
        if source_desc.get("symmetry_vertical") or target_desc.get("symmetry_vertical"):
            hints.add("symmetry_vertical")

        interior_in = source_desc.get("interior_colors", [])
        interior_out = target_desc.get("interior_colors", [])
        if interior_in == [] and interior_out:
            hints.add("fill_placeholder")
        if len(source_desc.get("row_stripes", [])) > 0 and any(len(set(row)) > 1 for row in source_desc.get("row_stripes", [])):
            hints.add("stripe_sensitive")

        return hints

    def _translation_vector(self, obj_in, obj_out) -> Optional[Tuple[int, int]]:
        if obj_in.size != obj_out.size:
            return None
        min_r_in, min_c_in, _, _ = obj_in.bounding_box
        min_r_out, min_c_out, _, _ = obj_out.bounding_box
        return (min_r_out - min_r_in, min_c_out - min_c_in)

    def _spatial_relations(self, idx: int, inputs, outputs) -> List[RelationalFact]:
        relations: List[RelationalFact] = []
        combined = list(inputs) + list(outputs)
        if not combined:
            return relations
        for rel in self._spatial.analyze_relations(combined):
            node_a = self._node_id(idx, "spatial", rel.obj1_id)
            node_b = self._node_id(idx, "spatial", rel.obj2_id)
            relations.append(
                RelationalFact(
                    source=node_a,
                    target=node_b,
                    frame="spatial",
                    context=rel.relation,
                    confidence=rel.confidence,
                    metadata={
                        "distance": rel.distance,
                    },
                )
            )
        return relations

    def _apply_entailment(self, relations: List[RelationalFact]) -> None:
        seen: Set[Tuple[str, str, str, str]] = {
            (rel.source, rel.target, rel.frame, rel.context) for rel in relations
        }
        extra: List[RelationalFact] = []

        # Mutual entailment (bidirectional reasoning)
        for rel in list(relations):
            if rel.frame in {"coordination", "opposition", "comparison", "spatial"}:
                mirrored = rel.mirrored()
                key = (mirrored.source, mirrored.target, mirrored.frame, mirrored.context)
                if key not in seen:
                    extra.append(mirrored)
                    seen.add(key)

        # Combinatorial entailment (compose translations)
        base_relations = relations + extra  # include symmetrical relations when composing
        for rel_a in base_relations:
            for rel_b in base_relations:
                composed = rel_a.compose_with(rel_b)
                if composed is None:
                    continue
                key = (composed.source, composed.target, composed.frame, composed.context)
                if key not in seen:
                    extra.append(composed)
                    seen.add(key)

        relations.extend(extra)

    def _transform_functions(self, relations: List[RelationalFact], hints: Dict[str, Set[str]]) -> None:
        adjacency: Dict[str, Set[str]] = {}
        for rel in relations:
            adjacency.setdefault(rel.source, set()).add(rel.target)
        updated = True
        while updated:
            updated = False
            for source, neighbours in adjacency.items():
                source_hints = hints.get(source, set())
                for neighbour in neighbours:
                    if neighbour not in hints:
                        hints[neighbour] = set()
                    before = len(hints[neighbour])
                    hints[neighbour].update(source_hints)
                    if len(hints[neighbour]) > before:
                        updated = True

    def _node_id(self, pair_idx: int, domain: str, obj_id: int) -> str:
        return f"{pair_idx}:{domain}:{obj_id}"


# [S:ALG v1] component=rft_engine entailment=mutual+transitive pass
