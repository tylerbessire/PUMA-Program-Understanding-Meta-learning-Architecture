"""Fast relational frame detectors for early pattern interrupts in ARC."""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
from collections import deque

import numpy as np

from .grid import Array
from .detector_core import BaseDetector, DetectionSignal
from .advanced_detectors import (
    GravityDetector,
    ColorCountDetector,
    MosaicDetector,
    BorderDetector,
    DiagonalDetector,
    DirectCropDetector,
    RotateCropDetector,
    RotateAdjacentToEightDetector,
    QuadrantFoldDetector,
    RecolorCropDetector,
    ComponentSubsetDetector,
)


class AttentionQueue:
    """Manage competing detector hypotheses with interrupt semantics."""

    def __init__(self, interrupt_threshold: float = 0.82):
        self.interrupt_threshold = interrupt_threshold
        self.signals: List[DetectionSignal] = []
        self.urgent_signal: Optional[DetectionSignal] = None

    def register(self, signal: DetectionSignal) -> None:
        """Register a detection signal, respecting interrupt behaviour."""

        if signal.confidence >= self.interrupt_threshold:
            if self.urgent_signal is None or signal.confidence > self.urgent_signal.confidence:
                self.urgent_signal = signal
        else:
            self.signals.append(signal)

    def get_winner(self) -> Optional[DetectionSignal]:
        """Return the highest-priority hypothesis, if any."""

        if self.urgent_signal is not None:
            return self.urgent_signal

        if not self.signals:
            return None

        self.signals.sort(
            key=lambda s: s.confidence * (1.0 + s.priority * 0.1),
            reverse=True,
        )
        return self.signals[0]


class RareColorExpansionDetector(BaseDetector):
    """Detects rare input colour expanding to dominate the output."""

    def __init__(self) -> None:
        super().__init__("rare_to_uniform", priority=10)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_counts = input_feats.get("color_counts", {})
        output_counts = output_feats.get("color_counts", {})

        if not input_counts or not output_counts:
            return None

        shared = set(input_counts) & set(output_counts)
        if len(shared) != 1:
            return None

        (color,) = shared
        total_in = sum(input_counts.values())
        total_out = sum(output_counts.values())
        if total_in == 0 or total_out == 0:
            return None

        input_ratio = input_counts[color] / total_in
        output_ratio = output_counts[color] / total_out

        if input_ratio < 0.12 and output_ratio > 0.98:
            background = max(input_counts, key=input_counts.get)
            theta = 0.15
            background_ratio = input_counts.get(background, 0) / total_in
            if background == color or background_ratio < theta:
                return None
            hypothesis = {
                "rule_type": "expand_rare_color",
                "source_color": int(color),
                "transformation": f"fill_entire_grid({int(color)})",
                "source_location": input_feats.get("spatial", {}).get(color, "unknown"),
            }
            return DetectionSignal(0.94, hypothesis, self.name, self.priority)

        return None


class ColorSwapDetector(BaseDetector):
    """Detect colour swap transformations."""

    def __init__(self) -> None:
        super().__init__("color_swap", priority=8)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_counts = input_feats.get("color_counts", {})
        output_counts = output_feats.get("color_counts", {})

        if set(input_counts) != set(output_counts) or len(input_counts) != 2:
            return None

        colours = list(input_counts)
        c1, c2 = colours[0], colours[1]
        total = sum(input_counts.values())
        if total == 0:
            return None

        swap_error = abs(input_counts[c1] - output_counts.get(c2, 0)) + abs(
            input_counts[c2] - output_counts.get(c1, 0)
        )

        if swap_error / total < 0.08:
            hypothesis = {
                "rule_type": "color_swap",
                "swap_pairs": [(int(c1), int(c2))],
                "transformation": f"swap({int(c1)}↔{int(c2)})",
            }
            return DetectionSignal(0.90, hypothesis, self.name, self.priority)

        return None


class GridSizeChangeDetector(BaseDetector):
    """Detect grid tiling or cropping relationships."""

    def __init__(self) -> None:
        super().__init__("size_transform", priority=9)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        inp_h, inp_w = input_feats.get("shape", (0, 0))
        out_h, out_w = output_feats.get("shape", (0, 0))

        if (inp_h, inp_w) == (out_h, out_w):
            return None

        if inp_h and inp_w and out_h and out_w:
            if out_h % inp_h == 0 and out_w % inp_w == 0:
                tile_h = out_h // inp_h
                tile_w = out_w // inp_w
                if tile_h == tile_w:
                    return DetectionSignal(
                        0.88,
                        {
                            "rule_type": "tile_grid",
                            "tile_factor": tile_h,
                            "transformation": f"tile_{tile_h}x{tile_w}",
                        },
                        self.name,
                        self.priority,
                    )

            if inp_h % out_h == 0 and inp_w % out_w == 0:
                crop_h = inp_h // out_h
                crop_w = inp_w // out_w
                return DetectionSignal(
                    0.85,
                    {
                        "rule_type": "crop_grid",
                        "crop_factor": (crop_h, crop_w),
                        "transformation": f"crop_by_{crop_h}x{crop_w}",
                    },
                    self.name,
                    self.priority,
                )

        return None


class SymmetryDetector(BaseDetector):
    """Detect symmetric outputs."""

    def __init__(self) -> None:
        super().__init__("symmetry", priority=7)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        shape = output_feats.get("shape", (0, 0))
        if min(shape) <= 1:
            return None
        if output_feats.get("has_vertical_symmetry"):
            return DetectionSignal(
                0.82,
                {
                    "rule_type": "mirror",
                    "axis": "vertical",
                    "transformation": "mirror_vertical",
                },
                self.name,
                self.priority,
            )

        if output_feats.get("has_horizontal_symmetry"):
            return DetectionSignal(
                0.82,
                {
                    "rule_type": "mirror",
                    "axis": "horizontal",
                    "transformation": "mirror_horizontal",
                },
                self.name,
                self.priority,
            )

        return None


class ObjectTranslationDetector(BaseDetector):
    """Detect consistent object translation vectors."""

    def __init__(self) -> None:
        super().__init__("object_translation", priority=8)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_components = input_feats.get("components", [])
        output_components = output_feats.get("components", [])
        if not input_components or len(input_components) != len(output_components):
            return None

        matches = []
        used = set()
        for comp in input_components:
            candidates = []
            for idx, out_comp in enumerate(output_components):
                if idx in used:
                    continue
                if (
                    out_comp["color"] == comp["color"]
                    and out_comp["area"] == comp["area"]
                    and (out_comp["bbox"][2] - out_comp["bbox"][0]) == (comp["bbox"][2] - comp["bbox"][0])
                    and (out_comp["bbox"][3] - out_comp["bbox"][1]) == (comp["bbox"][3] - comp["bbox"][1])
                ):
                    candidates.append((idx, out_comp))
            if len(candidates) != 1:
                return None
            idx, out_comp = candidates[0]
            used.add(idx)
            dy = out_comp["bbox"][0] - comp["bbox"][0]
            dx = out_comp["bbox"][1] - comp["bbox"][1]
            matches.append((dy, dx))

        if not matches:
            return None

        dy, dx = matches[0]
        if any((m[0] != dy or m[1] != dx) for m in matches[1:]):
            return None

        if dy == 0 and dx == 0:
            return None

        hypothesis = {
            "rule_type": "translate_objects",
            "vector": (int(dy), int(dx)),
            "background": input_feats.get("background_color", 0),
            "transformation": f"translate({int(dy)},{int(dx)})",
        }
        return DetectionSignal(0.88, hypothesis, self.name, self.priority)


class RepeatPatternDetector(BaseDetector):
    """Detect repeated sub-patterns in the output grid."""

    def __init__(self) -> None:
        super().__init__("repeat_pattern", priority=6)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        out_h, out_w = output_feats.get("shape", (0, 0))

        if out_h % 2 == 0 and out_w % 2 == 0 and out_h and out_w:
            hypothesis = {
                "rule_type": "repeat_pattern",
                "pattern_size": (out_h // 2, out_w // 2),
                "transformation": "tile_pattern",
            }
            return DetectionSignal(0.70, hypothesis, self.name, self.priority)

        return None


class BackgroundFillDetector(BaseDetector):
    """Detect background recolouring operations."""

    def __init__(self) -> None:
        super().__init__("background_fill", priority=5)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        inp_bg = input_feats.get("background_color")
        out_bg = output_feats.get("background_color")

        if inp_bg is None or out_bg is None or inp_bg == out_bg:
            return None

        output_counts = output_feats.get("color_counts", {})
        total = sum(output_counts.values())
        if total == 0:
            return None

        dominance = output_counts.get(out_bg, 0) / total
        if dominance > 0.5:
            hypothesis = {
                "rule_type": "change_background",
                "from_color": int(inp_bg),
                "to_color": int(out_bg),
                "transformation": f"recolor_bg({int(inp_bg)}→{int(out_bg)})",
            }
            return DetectionSignal(0.75, hypothesis, self.name, self.priority)

        return None


class DetectorOrchestrator:
    """Coordinate detectors and expose a simple scan API."""

    def __init__(self) -> None:
        self.detectors: List[BaseDetector] = [
            RareColorExpansionDetector(),
            ColorSwapDetector(),
            GridSizeChangeDetector(),
            ObjectTranslationDetector(),
            SymmetryDetector(),
            RepeatPatternDetector(),
            BackgroundFillDetector(),
            GravityDetector(),
            ColorCountDetector(),
            MosaicDetector(),
            BorderDetector(),
            DiagonalDetector(),
            DirectCropDetector(),
            RotateAdjacentToEightDetector(),
            RotateCropDetector(),
            QuadrantFoldDetector(),
            RecolorCropDetector(),
            ComponentSubsetDetector(),
        ]
        self.attention_queue = AttentionQueue()

    def scan_example(self, input_grid: Array, output_grid: Array) -> Optional[DetectionSignal]:
        input_feats = self._extract_features(input_grid)
        output_feats = self._extract_features(output_grid)

        self.attention_queue = AttentionQueue()
        for detector in self.detectors:
            signal = detector.scan(input_feats, output_feats)
            if signal is not None:
                self.attention_queue.register(signal)

        return self.attention_queue.get_winner()

    def _extract_features(self, grid: Array) -> Dict[str, Any]:
        features: Dict[str, Any] = {"shape": grid.shape, "grid": grid}

        unique, counts = np.unique(grid, return_counts=True)
        counts_dict = {int(col): int(cnt) for col, cnt in zip(unique, counts)}
        features["color_counts"] = counts_dict

        if counts.size > 0:
            dominant_idx = int(np.argmax(counts))
            features["dominant_color"] = int(unique[dominant_idx])
            features["background_color"] = features["dominant_color"]
        else:
            features["dominant_color"] = 0
            features["background_color"] = 0

        spatial: Dict[int, str] = {}
        centroids: Dict[int, Tuple[float, float]] = {}
        bounding_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        h, w = grid.shape
        for color in unique:
            coords = np.argwhere(grid == color)
            if coords.size == 0:
                continue
            centroid = coords.mean(axis=0)
            centroids[int(color)] = (float(centroid[0]), float(centroid[1]))
            if np.allclose(centroid, np.array([h / 2, w / 2]), atol=1.0):
                spatial[int(color)] = "center"
            elif coords.shape[0] == 4:
                corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
                if all(tuple(c) in corners for c in coords):
                    spatial[int(color)] = "corners"

            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            bounding_boxes[int(color)] = (int(min_r), int(min_c), int(max_r + 1), int(max_c + 1))

        components = self._compute_components(grid)

        features["spatial"] = spatial
        features["centroids"] = centroids
        features["bounding_boxes"] = bounding_boxes
        features["components"] = components
        features["has_vertical_symmetry"] = bool(np.array_equal(grid, np.fliplr(grid)))
        features["has_horizontal_symmetry"] = bool(np.array_equal(grid, np.flipud(grid)))

        return features

    def _compute_components(self, grid: Array) -> List[Dict[str, Any]]:
        h, w = grid.shape
        visited = np.zeros((h, w), dtype=bool)
        components: List[Dict[str, Any]] = []

        for r in range(h):
            for c in range(w):
                color = int(grid[r, c])
                if color == 0 or visited[r, c]:
                    continue

                queue = deque([(r, c)])
                visited[r, c] = True
                tiles: List[Tuple[int, int]] = []
                min_r = max_r = r
                min_c = max_c = c

                while queue:
                    cr, cc = queue.popleft()
                    tiles.append((cr, cc))
                    min_r = min(min_r, cr)
                    min_c = min(min_c, cc)
                    max_r = max(max_r, cr)
                    max_c = max(max_c, cc)

                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and not visited[nr, nc]
                            and int(grid[nr, nc]) == color
                        ):
                            visited[nr, nc] = True
                            queue.append((nr, nc))

                area = len(tiles)
                centroid_r = sum(t[0] for t in tiles) / area
                centroid_c = sum(t[1] for t in tiles) / area

                components.append(
                    {
                        "color": color,
                        "tiles": tiles,
                        "bbox": (min_r, min_c, max_r + 1, max_c + 1),
                        "area": area,
                        "centroid": (centroid_r, centroid_c),
                    }
                )

        return components
