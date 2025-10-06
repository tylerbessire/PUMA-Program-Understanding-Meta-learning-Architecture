"""Hypothesis generation and testing for fluid intelligence in ARC tasks.

This module implements a lightweight hypothesis generation framework.  It is
not meant to be an exhaustive reasoning system but rather a scaffold that can
be extended in later phases.  The engine analyses training pairs and proposes
plausible transformations together with simple confidence estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid import Array, bg_color
from .dsl import apply_program
from .continuous_learning import ContinuousSelfMemory
from .rft import RelationalFrameAnalyzer
from .object_reasoning import ObjectExtractor
from .glyph_extraction import GlyphExtractor, GlyphConfig


@dataclass
class Hypothesis:
    """Represents a hypothesis about task transformation."""

    description: str
    transformation_type: str  # "rotation", "color_swap", "pattern_fill", etc.
    confidence: float
    evidence: List[Dict[str, Any]]
    program_sketch: Optional[List[Tuple[str, Dict[str, Any]]]] = None


class HypothesisEngine:
    """Generates and tests hypotheses about ARC task transformations."""

    def __init__(self, continuous_memory: Optional[ContinuousSelfMemory] = None):
        self.continuous_memory = continuous_memory
        self.rft_analyzer = RelationalFrameAnalyzer()
        self.object_extractor = ObjectExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        """Generate multiple competing hypotheses about the task transformation."""
        hypotheses: List[Hypothesis] = []

        # 1. Geometric transformation hypotheses
        hypotheses.extend(self._generate_geometric_hypotheses(train_pairs))

        # 2. Color transformation hypotheses
        hypotheses.extend(self._generate_color_hypotheses(train_pairs))

        # 3. Pattern completion hypotheses
        hypotheses.extend(self._generate_pattern_hypotheses(train_pairs))

        # 3.5. Expansion / tiling hypotheses
        hypotheses.extend(self._generate_expansion_hypotheses(train_pairs))

        # 3.55. Hole-fill recolor hypotheses
        hypotheses.extend(self._generate_fill_hole_hypotheses(train_pairs))

        # 3.57. Glyph extraction hypotheses
        glyph_hyp = self._generate_glyph_extraction_hypothesis(train_pairs)
        if glyph_hyp:
            hypotheses.append(glyph_hyp)

        # 3.6 Relational reasoning hypotheses
        hypotheses.extend(self._generate_relational_hypotheses(train_pairs))

        # 4. Object manipulation hypotheses
        hypotheses.extend(self._generate_object_hypotheses(train_pairs))

        # 4.5 Area-based frame fill hypotheses
        hypotheses.extend(self._generate_area_fill_hypotheses(train_pairs))

        # 5. Memory-guided hypotheses
        if self.continuous_memory:
            hypotheses.extend(self._generate_memory_hypotheses(train_pairs))

        return sorted(hypotheses, key=lambda h: h.confidence, reverse=True)

    def test_hypothesis(self, hypothesis: Hypothesis, train_pairs: List[Tuple[Array, Array]]) -> float:
        """Test hypothesis validity against training data.

        Returns a confidence score between 0 and 1 based on how many training
        pairs are perfectly explained by the hypothesis.
        """
        if not train_pairs:
            return 0.0
        matches = 0
        for inp, out in train_pairs:
            pred = self.apply(hypothesis, inp)
            if pred is not None and pred.shape == out.shape and np.array_equal(pred, out):
                matches += 1
        return matches / len(train_pairs)

    def refine_hypothesis(self, hypothesis: Hypothesis, feedback: Dict[str, Any]) -> Hypothesis:
        """Refine hypothesis based on test results.

        A very small refinement mechanism is provided for now: the confidence is
        updated if feedback contains a ``confidence`` field and any additional
        evidence is appended to the hypothesis' evidence list.
        """
        new_conf = float(feedback.get("confidence", hypothesis.confidence))
        new_evidence = hypothesis.evidence + [feedback.get("evidence", {})]
        return Hypothesis(
            description=hypothesis.description,
            transformation_type=hypothesis.transformation_type,
            confidence=new_conf,
            evidence=new_evidence,
            program_sketch=hypothesis.program_sketch,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def apply(self, hypothesis: Hypothesis, grid: Array) -> Optional[Array]:
        """Apply the hypothesis to a grid, returning the transformed grid."""
        try:
            if hypothesis.transformation_type == "rotation" and hypothesis.program_sketch:
                k = int(hypothesis.program_sketch[0][1].get("k", 0))
                return np.rot90(grid, k)
            if hypothesis.transformation_type == "color_swap" and hypothesis.program_sketch:
                mapping = hypothesis.program_sketch[0][1].get("mapping", {})
                result = grid.copy()
                for src, dst in mapping.items():
                    result[grid == src] = dst
                return result
            if hypothesis.transformation_type == "pattern_fill" and hypothesis.program_sketch:
                color = int(hypothesis.program_sketch[0][1].get("color", 0))
                return np.full_like(grid, color)
            if hypothesis.transformation_type == "object_translation" and hypothesis.program_sketch:
                params = hypothesis.program_sketch[0][1]
                dy = int(params.get("dy", 0))
                dx = int(params.get("dx", 0))
                h, w = grid.shape
                result = np.zeros_like(grid)
                ys, xs = np.nonzero(grid)
                ys_new = ys + dy
                xs_new = xs + dx
                if (
                    (ys_new < 0).any()
                    or (ys_new >= h).any()
                    or (xs_new < 0).any()
                    or (xs_new >= w).any()
                ):
                    return None
                result[ys_new, xs_new] = grid[ys, xs]
                return result
            if hypothesis.transformation_type == "fill_holes" and hypothesis.program_sketch:
                params = hypothesis.program_sketch[0][1]
                fill_color = int(params.get("fill_color", 0))
                return self._fill_holes(grid, fill_color)
            if (
                hypothesis.transformation_type == "fill_regions_by_area"
                and hypothesis.program_sketch
            ):
                params = hypothesis.program_sketch[0][1]
                mapping = params.get("mapping", {})
                return self._fill_regions_by_area(grid, mapping)
            if (
                hypothesis.transformation_type == "glyph_extraction"
                and hypothesis.program_sketch
            ):
                params = hypothesis.program_sketch[0][1]
                configs_raw = params.get("configs", [])
                extractor = GlyphExtractor()
                extractor.configs = [
                    GlyphConfig(
                        top=cfg.get("cropping", [0, 0, 0, 0])[0],
                        bottom=cfg.get("cropping", [0, 0, 0, 0])[1],
                        left=cfg.get("cropping", [0, 0, 0, 0])[2],
                        right=cfg.get("cropping", [0, 0, 0, 0])[3],
                        ratio_h=cfg.get("ratio", [1, 1])[0],
                        ratio_w=cfg.get("ratio", [1, 1])[1],
                        output_shape=tuple(cfg.get("output_shape", [1, 1])),
                        mapping={str(k): int(v) for k, v in cfg.get("mapping", {}).items()},
                    )
                    for cfg in configs_raw
                ]
                return extractor.predict(grid)
            if hypothesis.transformation_type == "sort_rows":
                return np.sort(grid, axis=1)
            if hypothesis.transformation_type == "sort_columns":
                return np.sort(grid, axis=0)
            if hypothesis.transformation_type == "align_top_left":
                return self._align_grid_top_left(grid)
            if hypothesis.transformation_type == "block_row_flip" and hypothesis.program_sketch:
                params = hypothesis.program_sketch[0][1]
                factor_h = int(params.get("factor_h", 1))
                factor_w = int(params.get("factor_w", 1))
                row_pattern: List[str] = params.get("row_pattern", [])
                h, w = grid.shape
                if len(row_pattern) != factor_h:
                    return None
                result = np.zeros((h * factor_h, w * factor_w), dtype=grid.dtype)
                base = grid.copy()
                for br in range(factor_h):
                    orientation = row_pattern[br]
                    block = base.copy()
                    if orientation in ("flip_lr", "flip_both"):
                        block = np.fliplr(block)
                    if orientation in ("flip_ud", "flip_both"):
                        block = np.flipud(block)
                    for bc in range(factor_w):
                        result[
                            br * h : (br + 1) * h,
                            bc * w : (bc + 1) * w,
                        ] = block
                return result
            if hypothesis.transformation_type == "pattern_stamp" and hypothesis.program_sketch:
                params = hypothesis.program_sketch[0][1]
                factor_h = int(params.get("factor_h", 1))
                factor_w = int(params.get("factor_w", 1))
                background = int(params.get("background", 0))
                input_background = int(params.get("input_background", background))
                h, w = grid.shape
                if factor_h != h or factor_w != w:
                    return None
                result = np.full((h * factor_h, w * factor_w), background, dtype=grid.dtype)
                stamp = grid.copy()
                for i in range(h):
                    for j in range(w):
                        if grid[i, j] != input_background:
                            result[
                                i * h : (i + 1) * h,
                                j * w : (j + 1) * w,
                            ] = stamp
                return result
            if hypothesis.program_sketch:
                try:
                    return apply_program(grid, hypothesis.program_sketch)
                except Exception:
                    return None
        except Exception:
            return None
        return None

    # Hypothesis generation subroutines ---------------------------------
    def _generate_geometric_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        rotations = [1, 2, 3]  # 90, 180, 270 degrees
        for k in rotations:
            evidence: List[Dict[str, Any]] = []
            matches = 0
            for idx, (inp, out) in enumerate(train_pairs):
                rotated = np.rot90(inp, k)
                match = rotated.shape == out.shape and np.array_equal(rotated, out)
                evidence.append({"pair": idx, "rotation": k * 90, "match": match})
                if match:
                    matches += 1
            confidence = matches / len(train_pairs)
            if confidence > 0:
                hyps.append(
                    Hypothesis(
                        description=f"Rotate input by {k * 90} degrees",
                        transformation_type="rotation",
                        confidence=confidence,
                        evidence=evidence,
                        program_sketch=[("rotate", {"k": k})],
                    )
                )
        return hyps

    def _generate_color_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        global_mapping: Dict[int, int] = {}
        evidence: List[Dict[str, Any]] = []
        consistent = True
        for idx, (inp, out) in enumerate(train_pairs):
            mapping: Dict[int, int] = {}
            for ci, co in zip(inp.flat, out.flat):
                if ci in mapping and mapping[ci] != int(co):
                    consistent = False
                    break
                if ci != co:
                    mapping[ci] = int(co)
            evidence.append({"pair": idx, "mapping": mapping})
            for k, v in mapping.items():
                if k in global_mapping and global_mapping[k] != v:
                    consistent = False
                    break
                global_mapping[k] = v
            if not consistent:
                break
        if consistent and global_mapping:
            hyps.append(
                Hypothesis(
                    description=f"Recolor using mapping {global_mapping}",
                    transformation_type="color_swap",
                    confidence=1.0,
                    evidence=evidence,
                    program_sketch=[("recolor", {"mapping": global_mapping})],
                )
            )
        return hyps

    def _generate_pattern_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        colors = [np.unique(out) for _, out in train_pairs]
        if colors and all(len(c) == 1 for c in colors):
            color = int(colors[0][0])
            evidence = [{"pair": idx, "color": int(c[0])} for idx, c in enumerate(colors)]
            hyps.append(
                Hypothesis(
                    description=f"Fill grid with color {color}",
                    transformation_type="pattern_fill",
                    confidence=1.0,
                    evidence=evidence,
                    program_sketch=[("fill", {"color": color})],
                )
            )
        return hyps

    def _generate_expansion_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        if not train_pairs:
            return hyps

        # Verify that all pairs exhibit a consistent expansion ratio
        ratios = []
        for inp, out in train_pairs:
            h_in, w_in = inp.shape
            h_out, w_out = out.shape
            if h_in == 0 or w_in == 0:
                return hyps
            if h_out % h_in != 0 or w_out % w_in != 0:
                return hyps
            ratios.append((h_out // h_in, w_out // w_in))

        factor_h, factor_w = ratios[0]
        if any(r != (factor_h, factor_w) for r in ratios[1:]):
            return hyps
        if factor_h <= 1 and factor_w <= 1:
            return hyps

        row_flip = self._detect_block_row_flip(train_pairs, factor_h, factor_w)
        if row_flip:
            row_pattern = row_flip["row_pattern"]
            hyps.append(
                Hypothesis(
                    description="Expand grid with alternating horizontal flips",
                    transformation_type="block_row_flip",
                    confidence=1.0,
                    evidence=[{"row_pattern": row_pattern, "factors": (factor_h, factor_w)}],
                    program_sketch=[(
                        "block_row_flip",
                        {
                            "factor_h": factor_h,
                            "factor_w": factor_w,
                            "row_pattern": row_pattern,
                        },
                    )],
                )
            )

        stamp = self._detect_pattern_stamp(train_pairs, factor_h, factor_w)
        if stamp:
            hyps.append(
                Hypothesis(
                    description="Stamp input pattern at active cells",
                    transformation_type="pattern_stamp",
                    confidence=1.0,
                    evidence=[{"factors": (factor_h, factor_w), "background": stamp["background"]}],
                    program_sketch=[(
                        "pattern_stamp",
                        {
                            "factor_h": factor_h,
                            "factor_w": factor_w,
                            "background": stamp["background"],
                            "input_background": stamp["input_background"],
                        },
                    )],
                )
            )

        return hyps

    def _generate_fill_hole_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        if not train_pairs:
            return []

        fill_colors: List[int] = []
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return []
            diff_mask = out != inp
            if not diff_mask.any():
                return []
            diff_vals = np.unique(out[diff_mask])
            if len(diff_vals) != 1:
                return []
            fill_color = int(diff_vals[0])
            if fill_color in np.unique(inp):
                return []
            zero_mask = inp == 0
            if np.any(~zero_mask & diff_mask):
                return []

            visited = np.zeros_like(inp, dtype=bool)
            coords = np.argwhere(diff_mask)
            for r, c in coords:
                if visited[r, c]:
                    continue
                stack = [(int(r), int(c))]
                visited[r, c] = True
                touches_boundary = False
                fully_filled = True
                while stack:
                    rr, cc = stack.pop()
                    if rr in (0, inp.shape[0] - 1) or cc in (0, inp.shape[1] - 1):
                        touches_boundary = True
                    if not diff_mask[rr, cc]:
                        fully_filled = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]:
                            if zero_mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                if touches_boundary or not fully_filled:
                    return []

            fill_colors.append(fill_color)

        if not fill_colors or len(set(fill_colors)) != 1:
            return []

        fill_color = fill_colors[0]
        return [
            Hypothesis(
                description=f"Fill enclosed object holes with color {fill_color}",
                transformation_type="fill_holes",
                confidence=0.95,
                evidence=[{"fill_color": fill_color}],
                program_sketch=[("fill_holes", {"fill_color": fill_color})],
            )
        ]

    def _generate_area_fill_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        if not train_pairs:
            return []

        area_to_color: Dict[int, int] = {}
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return []
            zero_mask = inp == 0
            visited = np.zeros_like(inp, dtype=bool)
            h, w = inp.shape

            for r in range(h):
                for c in range(w):
                    if not zero_mask[r, c] or visited[r, c]:
                        continue
                    stack = [(r, c)]
                    visited[r, c] = True
                    component = []
                    touches_boundary = False
                    output_values: set[int] = set()

                    while stack:
                        rr, cc = stack.pop()
                        component.append((rr, cc))
                        if rr in (0, h - 1) or cc in (0, w - 1):
                            touches_boundary = True
                        output_values.add(int(out[rr, cc]))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if zero_mask[nr, nc] and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))

                    if touches_boundary:
                        continue

                    if output_values == {0}:
                        continue

                    if len(output_values) != 1 or 0 in output_values:
                        return []

                    fill_color = output_values.pop()
                    if fill_color in np.unique(inp):
                        return []

                    area = len(component)
                    existing = area_to_color.get(area)
                    if existing is not None and existing != fill_color:
                        return []
                    area_to_color[area] = fill_color

        if not area_to_color:
            return []

        return [
            Hypothesis(
                description="Fill enclosed regions based on area mapping",
                transformation_type="fill_regions_by_area",
                confidence=0.9,
                evidence=[{"area_to_color": dict(area_to_color)}],
                program_sketch=[("fill_regions_by_area", {"mapping": dict(area_to_color)})],
            )
        ]

    def _generate_glyph_extraction_hypothesis(self, train_pairs: List[Tuple[Array, Array]]) -> Optional[Hypothesis]:
        extractor = GlyphExtractor()
        if not extractor.train(train_pairs):
            return None

        configs_payload = []
        for cfg in extractor.configs:
            configs_payload.append(
                {
                    "cropping": [cfg.top, cfg.bottom, cfg.left, cfg.right],
                    "ratio": [cfg.ratio_h, cfg.ratio_w],
                    "output_shape": list(cfg.output_shape),
                    "mapping": cfg.mapping,
                }
            )

        return Hypothesis(
            description="Extract glyphs from large canvas via canonical block signatures",
            transformation_type="glyph_extraction",
            confidence=0.9,
            evidence=[{"config_count": len(configs_payload)}],
            program_sketch=[("glyph_extraction", {"configs": configs_payload})],
        )

    def _generate_relational_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        if not train_pairs:
            return hyps

        facts = self.rft_analyzer.analyze(train_pairs)
        rft_evidence = [
            {
                "relation": fact.relation,
                "subject": fact.subject,
                "object": fact.object,
                "metadata": fact.metadata,
            }
            for category in facts.values()
            for fact in category
        ]

        if self._rows_sorted(train_pairs):
            hyps.append(
                Hypothesis(
                    description="Sort rows left-to-right by color",
                    transformation_type="sort_rows",
                    confidence=1.0,
                    evidence=rft_evidence,
                    program_sketch=[("sort_rows", {})],
                )
            )

        if self._columns_sorted(train_pairs):
            hyps.append(
                Hypothesis(
                    description="Sort columns top-to-bottom by color",
                    transformation_type="sort_columns",
                    confidence=1.0,
                    evidence=rft_evidence,
                    program_sketch=[("sort_columns", {})],
                )
            )

        if self._aligns_top_left(train_pairs):
            hyps.append(
                Hypothesis(
                    description="Align objects toward the top-left anchor",
                    transformation_type="align_top_left",
                    confidence=1.0,
                    evidence=rft_evidence,
                    program_sketch=[("align_top_left", {})],
                )
            )

        return hyps

    def _rows_sorted(self, train_pairs: List[Tuple[Array, Array]]) -> bool:
        return all(np.array_equal(np.sort(inp, axis=1), out) for inp, out in train_pairs)

    def _columns_sorted(self, train_pairs: List[Tuple[Array, Array]]) -> bool:
        return all(np.array_equal(np.sort(inp, axis=0), out) for inp, out in train_pairs)

    def _aligns_top_left(self, train_pairs: List[Tuple[Array, Array]]) -> bool:
        return all(np.array_equal(self._align_grid_top_left(inp), out) for inp, out in train_pairs)

    def _detect_block_row_flip(
        self,
        train_pairs: List[Tuple[Array, Array]],
        factor_h: int,
        factor_w: int,
    ) -> Optional[Dict[str, Any]]:
        row_pattern: Optional[List[str]] = None
        for inp, out in train_pairs:
            h_in, w_in = inp.shape
            base = inp
            flip_lr = np.fliplr(base)
            flip_ud = np.flipud(base)
            flip_both = np.flipud(flip_lr)

            pair_pattern: List[str] = []
            for br in range(factor_h):
                orientations = set()
                for bc in range(factor_w):
                    block = out[br * h_in : (br + 1) * h_in, bc * w_in : (bc + 1) * w_in]
                    if np.array_equal(block, base):
                        orientations.add("identity")
                    elif np.array_equal(block, flip_lr):
                        orientations.add("flip_lr")
                    elif np.array_equal(block, flip_ud):
                        orientations.add("flip_ud")
                    elif np.array_equal(block, flip_both):
                        orientations.add("flip_both")
                    else:
                        return None
                if len(orientations) != 1:
                    return None
                orientation = orientations.pop()
                pair_pattern.append(orientation)
            if row_pattern is None:
                row_pattern = pair_pattern
            elif row_pattern != pair_pattern:
                return None

        if row_pattern is None:
            return None

        return {"row_pattern": row_pattern}

    def _detect_pattern_stamp(
        self,
        train_pairs: List[Tuple[Array, Array]],
        factor_h: int,
        factor_w: int,
    ) -> Optional[Dict[str, int]]:
        background_out: Optional[int] = None
        input_background_candidates: Optional[set[int]] = None

        for inp, out in train_pairs:
            h_in, w_in = inp.shape
            if h_in != factor_h or w_in != factor_w:
                return None

            pair_background_values: set[int] = set()
            pair_active_values: set[int] = set()
            pair_bg_color: Optional[int] = None

            for i in range(factor_h):
                for j in range(factor_w):
                    block = out[i * h_in : (i + 1) * h_in, j * w_in : (j + 1) * w_in]
                    if np.array_equal(block, inp):
                        pair_active_values.add(int(inp[i, j]))
                        continue
                    if np.all(block == block.flat[0]):
                        color = int(block.flat[0])
                        if pair_bg_color is None:
                            pair_bg_color = color
                        elif pair_bg_color != color:
                            return None
                        pair_background_values.add(int(inp[i, j]))
                        continue
                    return None

            if not pair_background_values:
                return None

            if background_out is None:
                background_out = pair_bg_color
            elif pair_bg_color != background_out:
                return None

            if input_background_candidates is None:
                input_background_candidates = set(pair_background_values)
            else:
                input_background_candidates &= pair_background_values

            if not input_background_candidates:
                return None

            if input_background_candidates & pair_active_values:
                return None

        if background_out is None or not input_background_candidates:
            return None

        background_in = min(input_background_candidates)
        return {"background": background_out, "input_background": background_in}

    def _align_grid_top_left(self, grid: Array) -> Array:
        bg = bg_color(grid)
        h, w = grid.shape
        result = np.full_like(grid, bg)
        rows = [grid[r] for r in range(h) if not np.all(grid[r] == bg)]
        if not rows:
            return grid.copy()
        for new_r, row in enumerate(rows):
            values = row[row != bg]
            if values.size:
                result[new_r, :values.size] = values
        return result

    def _fill_holes(self, grid: Array, fill_color: int) -> Array:
        result = grid.copy()
        h, w = result.shape
        reachable = np.zeros((h, w), dtype=bool)
        stack = []

        def enqueue(r: int, c: int) -> None:
            if 0 <= r < h and 0 <= c < w and not reachable[r, c] and result[r, c] == 0:
                reachable[r, c] = True
                stack.append((r, c))

        for r in range(h):
            enqueue(r, 0)
            enqueue(r, w - 1)
        for c in range(w):
            enqueue(0, c)
            enqueue(h - 1, c)

        while stack:
            cr, cc = stack.pop()
            enqueue(cr - 1, cc)
            enqueue(cr + 1, cc)
            enqueue(cr, cc - 1)
            enqueue(cr, cc + 1)

        holes = (result == 0) & (~reachable)
        if np.any(holes):
            result[holes] = fill_color
        return result

    def _fill_regions_by_area(self, grid: Array, mapping: Dict[int, int]) -> Array:
        result = grid.copy()
        h, w = result.shape
        visited = np.zeros_like(grid, dtype=bool)
        for r in range(h):
            for c in range(w):
                if result[r, c] != 0 or visited[r, c]:
                    continue
                stack = [(r, c)]
                visited[r, c] = True
                component = []
                touches_boundary = False
                while stack:
                    rr, cc = stack.pop()
                    component.append((rr, cc))
                    if rr in (0, h - 1) or cc in (0, w - 1):
                        touches_boundary = True
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if result[nr, nc] == 0 and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))

                if touches_boundary:
                    continue

                area = len(component)
                fill_color = mapping.get(area)
                if fill_color is None:
                    continue
                for rr, cc in component:
                    result[rr, cc] = fill_color
        return result

    def _generate_memory_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        suggestions = self.continuous_memory.suggest_transformations(train_pairs) if self.continuous_memory else []
        hyps: List[Hypothesis] = []
        for suggestion in suggestions:
            name = suggestion.get("transformation")
            sketch = suggestion.get("program_sketch")
            hypothesis = self._build_memory_hypothesis(name, sketch)
            if hypothesis is not None:
                hyps.append(hypothesis)
        return hyps

    def _build_memory_hypothesis(
        self, transformation: Optional[str], sketch: Any
    ) -> Optional[Hypothesis]:
        if not transformation:
            return None
        if transformation == "ground_truth_reference":
            return None
        description = f"Memory-guided transformation: {transformation}"
        program_sketch = sketch if isinstance(sketch, list) else None
        return Hypothesis(
            description=description,
            transformation_type=transformation,
            confidence=0.6,
            evidence=[{"source": "continuous_memory", "transformation": transformation}],
            program_sketch=program_sketch,
        )

    def _find_translation(self, inp: Array, out: Array) -> Optional[Tuple[int, int]]:
        if inp.shape != out.shape:
            return None
        coords_in = np.argwhere(inp != 0)
        coords_out = np.argwhere(out != 0)
        if len(coords_in) == 0 or len(coords_out) == 0 or len(coords_in) != len(coords_out):
            return None
        shift = coords_out[0] - coords_in[0]
        h, w = inp.shape
        translated = np.zeros_like(inp)
        ys = coords_in[:, 0] + shift[0]
        xs = coords_in[:, 1] + shift[1]
        if (ys < 0).any() or (ys >= h).any() or (xs < 0).any() or (xs >= w).any():
            return None
        translated[ys, xs] = inp[coords_in[:, 0], coords_in[:, 1]]
        if np.array_equal(translated, out):
            return int(shift[0]), int(shift[1])
        return None

    def _generate_object_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[Hypothesis]:
        hyps: List[Hypothesis] = []
        shifts: List[Tuple[int, int]] = []
        evidence: List[Dict[str, Any]] = []
        for idx, (inp, out) in enumerate(train_pairs):
            trans = self._find_translation(inp, out)
            evidence.append({"pair": idx, "shift": trans})
            if trans is not None:
                shifts.append(trans)
        if not shifts:
            return hyps
        common = shifts[0]
        if any(s != common for s in shifts):
            return hyps
        confidence = len(shifts) / len(train_pairs)
        hyps.append(
            Hypothesis(
                description=f"Translate object by {common}",
                transformation_type="object_translation",
                confidence=confidence,
                evidence=evidence,
                program_sketch=[("translate", {"dy": common[0], "dx": common[1]})],
            )
        )
        return hyps
