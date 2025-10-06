"""Advanced detector scaffolding for future ARC patterns."""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple, Set

import numpy as np

from .grid import Array
from .detector_core import BaseDetector, DetectionSignal


class GravityDetector(BaseDetector):
    """Detect objects falling or rising under implied gravity."""

    def __init__(self) -> None:
        super().__init__("gravity", priority=7)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_counts = input_feats.get("color_counts", {})
        output_counts = output_feats.get("color_counts", {})
        background = input_feats.get("background_color")

        shared: List[int] = [
            color
            for color in input_counts
            if color in output_counts and color != background
        ]

        if not shared:
            return None

        input_centroids = input_feats.get("centroids", {})
        output_centroids = output_feats.get("centroids", {})
        if not input_centroids or not output_centroids:
            return None

        deltas = []
        for color in shared:
            cin = input_centroids.get(color)
            cout = output_centroids.get(color)
            if cin is None or cout is None:
                continue
            deltas.append((cout[0] - cin[0], cout[1] - cin[1]))

        if not deltas:
            return None

        avg_dy = sum(d[0] for d in deltas) / len(deltas)
        avg_dx = sum(d[1] for d in deltas) / len(deltas)

        if avg_dy <= 0.5 or abs(avg_dx) > 0.4:
            return None

        variance = max(abs(d[0] - avg_dy) for d in deltas)
        if variance > 1.0:
            return None

        dy = int(round(avg_dy))
        dx = int(round(avg_dx))
        if dy <= 0:
            return None

        hypothesis = {
            "rule_type": "gravity_drop",
            "vector": (dy, dx),
            "background": output_feats.get("background_color", background or 0),
            "transformation": f"gravity_drop({dy})",
        }
        confidence = min(0.9, 0.82 + 0.03 * len(shared))
        return DetectionSignal(confidence, hypothesis, self.name, self.priority)


class ColorCountDetector(BaseDetector):
    """Detect outputs driven by colour counting logic."""

    def __init__(self) -> None:
        super().__init__("color_count", priority=6)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_counts = input_feats.get("color_counts", {})
        output_grid: Array = output_feats.get("grid")
        if output_grid is None or not input_counts:
            return None

        h, w = output_grid.shape
        if min(h, w) != 1:
            return None

        sequence = list(output_grid.flatten())
        expected: List[int] = []
        for color in sorted(input_counts.keys()):
            expected.extend([int(color)] * int(input_counts[color]))

        if sequence != expected:
            return None

        orientation = "row" if h == 1 else "column"
        hypothesis = {
            "rule_type": "encode_color_counts",
            "ordering": sorted(input_counts.keys()),
            "orientation": orientation,
            "transformation": "encode_color_histogram",
        }
        return DetectionSignal(0.75, hypothesis, self.name, self.priority)


class MosaicDetector(BaseDetector):
    """Detect mosaics composed from repeated input fragments."""

    def __init__(self) -> None:
        super().__init__("mosaic", priority=7)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if input_grid is None or output_grid is None:
            return None

        inp_h, inp_w = input_grid.shape
        out_h, out_w = output_grid.shape
        if inp_h == 0 or inp_w == 0 or out_h % inp_h != 0 or out_w % inp_w != 0:
            return None

        tile_h = out_h // inp_h
        tile_w = out_w // inp_w
        if tile_h == 1 and tile_w == 1:
            return None

        tiled = np.tile(input_grid, (tile_h, tile_w))
        if not np.array_equal(tiled, output_grid):
            return None

        hypothesis = {
            "rule_type": "tile_grid",
            "tile_factor": (tile_h, tile_w),
            "transformation": f"tile_{tile_h}x{tile_w}",
        }
        confidence = min(0.88, 0.75 + 0.02 * (tile_h + tile_w))
        return DetectionSignal(confidence, hypothesis, self.name, self.priority)


class BorderDetector(BaseDetector):
    """Detect addition or removal of border structures."""

    def __init__(self) -> None:
        super().__init__("border", priority=6)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        inp_h, inp_w = input_feats.get("shape", (0, 0))
        out_h, out_w = output_feats.get("shape", (0, 0))
        output_grid: Array = output_feats.get("grid")
        input_grid: Array = input_feats.get("grid")
        if output_grid is None or input_grid is None:
            return None

        # Addition
        if out_h == inp_h + 2 and out_w == inp_w + 2:
            inner = output_grid[1:-1, 1:-1]
            if inner.shape == input_grid.shape and np.array_equal(inner, input_grid):
                border_color = int(output_grid[0, 0])
                hypothesis = {
                    "rule_type": "add_border",
                    "border_color": border_color,
                    "thickness": 1,
                    "transformation": f"add_border(color={border_color})",
                }
                return DetectionSignal(0.9, hypothesis, self.name, self.priority)

        # Removal
        if inp_h == out_h + 2 and inp_w == out_w + 2:
            inner = input_grid[1:-1, 1:-1]
            if inner.shape == output_grid.shape and np.array_equal(inner, output_grid):
                border_color = int(input_grid[0, 0])
                hypothesis = {
                    "rule_type": "remove_border",
                    "border_color": border_color,
                    "thickness": 1,
                    "transformation": f"remove_border(color={border_color})",
                }
                return DetectionSignal(0.9, hypothesis, self.name, self.priority)

        return None


class DiagonalDetector(BaseDetector):
    """Detect diagonal line manipulations."""

    def __init__(self) -> None:
        super().__init__("diagonal", priority=5)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        output_grid: Array = output_feats.get("grid")
        if output_grid is None:
            return None

        diag = np.diag(np.diag(output_grid))
        if not np.array_equal(output_grid, diag):
            return None

        colours = np.unique(output_grid)
        if len(colours) != 2:
            return None

        diag_color = int(np.diag(output_grid)[0])
        hypothesis = {
            "rule_type": "draw_diagonal",
            "color": diag_color,
            "transformation": f"draw_diagonal({diag_color})",
        }
        return DetectionSignal(0.7, hypothesis, self.name, self.priority)


class DirectCropDetector(BaseDetector):
    """Detect when output equals a subgrid crop of the input."""

    def __init__(self) -> None:
        super().__init__("direct_crop", priority=9)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if input_grid is None or output_grid is None:
            return None

        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        if oh > ih or ow > iw:
            return None

        matches = []
        for r in range(ih - oh + 1):
            for c in range(iw - ow + 1):
                if np.array_equal(input_grid[r : r + oh, c : c + ow], output_grid):
                    matches.append((r, c))

        if len(matches) != 1:
            return None

        top, left = matches[0]
        hypothesis = {
            "rule_type": "crop_subgrid",
            "top": int(top),
            "left": int(left),
            "height": int(oh),
            "width": int(ow),
            "transformation": f"crop_subgrid({top},{left},{oh},{ow})",
        }
        return DetectionSignal(0.93, hypothesis, self.name, self.priority)


class RecolorCropDetector(BaseDetector):
    """Detect removal of specific colours followed by a crop operation."""

    def __init__(self) -> None:
        super().__init__("recolor_crop", priority=12)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if input_grid is None or output_grid is None:
            return None

        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        if oh == 0 or ow == 0 or oh > ih or ow > iw:
            return None

        output_colors = set(np.unique(output_grid)) - {0}
        candidate_removals = set(np.unique(input_grid)) - output_colors - {0}
        if not candidate_removals:
            return None

        # Precompute bounding box of non-zero output tiles for trimming verification
        nz_rows, nz_cols = np.where(output_grid != 0)
        if nz_rows.size == 0:
            return None

        for top in range(ih - oh + 1):
            for left in range(iw - ow + 1):
                window = input_grid[top : top + oh, left : left + ow]
                removals: set[int] = set()
                valid = True
                for r in range(oh):
                    for c in range(ow):
                        val_in = int(window[r, c])
                        val_out = int(output_grid[r, c])
                        if val_in == val_out:
                            continue
                        if val_out == 0 and val_in in candidate_removals:
                            removals.add(val_in)
                        else:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid or not removals:
                    continue

                if removals & output_colors:
                    continue

                pruned = np.where(np.isin(input_grid, list(removals)), 0, input_grid)
                nz = np.argwhere(pruned != 0)
                if nz.size == 0:
                    continue
                min_r, min_c = nz.min(axis=0)
                max_r, max_c = nz.max(axis=0)
                crop = pruned[min_r : max_r + 1, min_c : max_c + 1]

                if np.array_equal(crop, output_grid):
                    hypothesis = {
                        "rule_type": "remove_colors_and_crop",
                        "removals": sorted(int(v) for v in removals),
                        "crop_top": int(min_r),
                        "crop_left": int(min_c),
                        "height": int(max_r - min_r + 1),
                        "width": int(max_c - min_c + 1),
                        "transformation": "remove_colors_and_crop",
                    }
                    return DetectionSignal(0.97, hypothesis, self.name, self.priority)

        return None


class RotateCropDetector(BaseDetector):
    """Collect rotation-specific template crops that match the output exactly."""

    def __init__(self) -> None:
        super().__init__("rotate_crop", priority=11)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if input_grid is None or output_grid is None:
            return None

        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        if oh == 0 or ow == 0 or oh > ih or ow > iw:
            return None

        templates: List[Dict[str, Any]] = []
        seen: Set[Tuple[int, bytes]] = set()

        for degrees, k in ((0, 0), (90, 1), (180, 2), (270, 3)):
            rotated = np.rot90(input_grid, k) if k else input_grid
            rh, rw = rotated.shape
            if oh > rh or ow > rw:
                continue

            for top in range(rh - oh + 1):
                window = rotated[top : top + oh]
                for left in range(rw - ow + 1):
                    candidate = window[:, left : left + ow]
                    if np.array_equal(candidate, output_grid):
                        key = (int(degrees), output_grid.tobytes())
                        if key in seen:
                            continue
                        seen.add(key)
                        templates.append(
                            {
                                "rotation": int(degrees),
                                "template": output_grid.tolist(),
                            }
                        )

        if not templates:
            return None

        hypothesis = {
            "rule_type": "rotate_template_match",
            "templates": templates,
        }

        return DetectionSignal(0.97, hypothesis, self.name, self.priority)


class RotateAdjacentToEightDetector(BaseDetector):
    """Detect patterns adjacent to colour-8 marker regions after rotation."""

    def __init__(self) -> None:
        super().__init__("rotate_adjacent_to_8", priority=12)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if input_grid is None or output_grid is None:
            return None

        oh, ow = output_grid.shape
        if oh == 0 or ow == 0:
            return None

        rotations = [
            (0, 0),
            (90, 1),
            (180, 2),
            (270, 3),
        ]

        for degrees, k in rotations:
            rotated = np.rot90(input_grid, k) if k else input_grid
            coords = np.argwhere(rotated == 8)
            if coords.size == 0:
                continue

            r_min = int(coords[:, 0].min())
            r_max = int(coords[:, 0].max())
            c_min = int(coords[:, 1].min())
            c_max = int(coords[:, 1].max())

            matches = self._match_output(rotated, output_grid)
            if not matches:
                continue

            for top, left in matches:
                bottom = top + oh - 1
                right = left + ow - 1

                direction = None
                row_offset = 0
                col_offset = 0

                # Right adjacency
                if left == c_max + 1 and top <= r_max and bottom >= r_min:
                    direction = "right"
                    row_offset = top - r_min
                    col_offset = left - (c_max + 1)
                # Left adjacency
                elif right == c_min - 1 and top <= r_max and bottom >= r_min:
                    direction = "left"
                    row_offset = top - r_min
                    col_offset = left - (c_min - ow)
                # Below adjacency
                elif top == r_max + 1 and left <= c_max and right >= c_min:
                    direction = "below"
                    row_offset = top - (r_max + 1)
                    col_offset = left - c_min
                # Above adjacency
                elif bottom == r_min - 1 and left <= c_max and right >= c_min:
                    direction = "above"
                    row_offset = top - (r_min - oh)
                    col_offset = left - c_min

                if direction is None:
                    continue

                hypothesis = {
                    "rule_type": "rotate_extract_adjacent_to_8",
                    "rotation": int(degrees),
                    "direction": direction,
                    "height": int(oh),
                    "width": int(ow),
                    "row_offset": int(row_offset),
                    "col_offset": int(col_offset),
                }
                return DetectionSignal(0.98, hypothesis, self.name, self.priority)

        return None

    def _match_output(self, grid: Array, output: Array) -> List[Tuple[int, int]]:
        gh, gw = grid.shape
        oh, ow = output.shape
        matches: List[Tuple[int, int]] = []

        if oh > gh or ow > gw:
            return matches

        for top in range(gh - oh + 1):
            window = grid[top : top + oh]
            for left in range(gw - ow + 1):
                if np.array_equal(window[:, left : left + ow], output):
                    matches.append((top, left))
        return matches


class QuadrantFoldDetector(BaseDetector):
    """Detect mismatches between mirrored quadrants and extract the anomaly."""

    def __init__(self) -> None:
        super().__init__("quadrant_fold", priority=12)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if input_grid is None or output_grid is None:
            return None

        ih, iw = input_grid.shape
        oh, ow = output_grid.shape
        if oh == 0 or ow == 0:
            return None

        quadrant_size = min(ih, iw) // 2
        if quadrant_size == 0:
            return None

        quadrants = self._extract_quadrants(input_grid)
        if not quadrants:
            return None

        anomaly = None
        for (name_a, qa), (name_b, qb) in self._mirrored_pairs(quadrants):
            if qa.shape != qb.shape:
                continue
            diff = qa != qb
            if diff.any():
                bbox = self._bounding_box(diff)
                if bbox is None:
                    continue
                anomaly = (name_a, qa[bbox], bbox)
                break

        if anomaly is None:
            return None

        _, candidate, _ = anomaly
        for degrees, k in ((0, 0), (90, 1), (180, 2), (270, 3)):
            rotated = np.rot90(candidate, k)
            if rotated.shape == output_grid.shape and np.array_equal(rotated, output_grid):
                hypothesis = {
                    "rule_type": "quadrant_fold",
                    "rotation": int(degrees),
                }
                return DetectionSignal(0.9, hypothesis, self.name, self.priority)

        return None

    def _extract_quadrants(self, grid: Array) -> Dict[str, Array]:
        h, w = grid.shape
        mid_r = h // 2
        mid_c = w // 2
        quadrants: Dict[str, Array] = {}
        quadrants["tl"] = grid[:mid_r, :mid_c]
        quadrants["tr"] = grid[:mid_r, mid_c:]
        quadrants["bl"] = grid[mid_r:, :mid_c]
        quadrants["br"] = grid[mid_r:, mid_c:]
        return quadrants

    def _mirrored_pairs(self, quadrants: Dict[str, Array]) -> List[Tuple[Tuple[str, Array], Tuple[str, Array]]]:
        pairs: List[Tuple[Tuple[str, Array], Tuple[str, Array]]] = []
        if "tl" in quadrants and "tr" in quadrants:
            pairs.append((("tl", quadrants["tl"]), ("tr", np.fliplr(quadrants["tr"])) ))
        if "bl" in quadrants and "br" in quadrants:
            pairs.append((("bl", quadrants["bl"]), ("br", np.fliplr(quadrants["br"])) ))
        if "tl" in quadrants and "bl" in quadrants:
            pairs.append((("tl", quadrants["tl"]), ("bl", np.flipud(quadrants["bl"])) ))
        if "tr" in quadrants and "br" in quadrants:
            pairs.append((("tr", quadrants["tr"]), ("br", np.flipud(quadrants["br"])) ))
        return pairs

    def _bounding_box(self, mask: Array) -> Optional[Tuple[slice, slice]]:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        rows = coords[:, 0]
        cols = coords[:, 1]
        return (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))


class ComponentSubsetDetector(BaseDetector):
    """Detect when the output consists of a subset of input components."""

    def __init__(self) -> None:
        super().__init__("component_subset", priority=10)

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> Optional[DetectionSignal]:
        input_components = input_feats.get("components", [])
        output_components = output_feats.get("components", [])
        input_grid: Array = input_feats.get("grid")
        output_grid: Array = output_feats.get("grid")
        if not input_components or not output_components or input_grid is None or output_grid is None:
            return None

        index: Dict[Tuple[Tuple[int, int], ...], Dict[Tuple[int, int], Dict[str, Any]]] = {}
        for comp in input_components:
            top, left, _, _ = comp["bbox"]
            sig = self._shape_signature(comp["tiles"])
            location_map = index.setdefault(sig, {})
            location_map[(int(top), int(left))] = comp

        sorted_outputs = sorted(
            enumerate(output_components), key=lambda item: item[1]["area"], reverse=True
        )

        for _, anchor in sorted_outputs:
            anchor_sig = self._shape_signature(anchor["tiles"])
            anchor_color = int(anchor["color"])
            anchor_top, anchor_left, _, _ = anchor["bbox"]

            candidates = [
                comp
                for comp in input_components
                if comp["color"] == anchor_color
                and self._shape_signature(comp["tiles"]) == anchor_sig
            ]

            for candidate in candidates:
                delta_r = int(candidate["bbox"][0]) - int(anchor_top)
                delta_c = int(candidate["bbox"][1]) - int(anchor_left)

                matched: List[Dict[str, Any]] = []
                seen_locations: set[Tuple[int, int]] = set()
                color_map: Dict[int, int] = {}
                ok = True

                for out_comp in output_components:
                    sig = self._shape_signature(out_comp["tiles"])
                    top, left, _, _ = out_comp["bbox"]
                    expected_top = int(top) + delta_r
                    expected_left = int(left) + delta_c
                    location_map = index.get(sig)
                    if location_map is None:
                        ok = False
                        break
                    comp_match = location_map.get((expected_top, expected_left))
                    if comp_match is None:
                        ok = False
                        break
                    if (expected_top, expected_left) in seen_locations:
                        ok = False
                        break
                    seen_locations.add((expected_top, expected_left))

                    input_color = int(comp_match["color"])
                    output_color = int(out_comp["color"])
                    mapped = color_map.get(input_color)
                    if mapped is not None and mapped != output_color:
                        ok = False
                        break
                    color_map[input_color] = output_color
                    matched.append(comp_match)

                if not ok or not matched:
                    continue

                min_r = min(int(comp["bbox"][0]) for comp in matched)
                min_c = min(int(comp["bbox"][1]) for comp in matched)
                max_r = max(int(comp["bbox"][2]) for comp in matched)
                max_c = max(int(comp["bbox"][3]) for comp in matched)

                crop = input_grid[min_r:max_r, min_c:max_c]
                recolored = crop.copy()
                for src, dst in color_map.items():
                    recolored[crop == src] = dst

                if np.array_equal(recolored, output_grid):
                    hypothesis = {
                        "rule_type": "recolor_crop_subgrid",
                        "top": int(min_r),
                        "left": int(min_c),
                        "height": int(max_r - min_r),
                        "width": int(max_c - min_c),
                        "color_map": {int(src): int(dst) for src, dst in color_map.items()},
                        "transformation": f"crop_subgrid({min_r},{min_c},{max_r - min_r},{max_c - min_c})",
                    }
                    return DetectionSignal(0.96, hypothesis, self.name, self.priority)

        return None

    def _same_shape(self, tiles_a: List[Tuple[int, int]], tiles_b: List[Tuple[int, int]]) -> bool:
        if len(tiles_a) != len(tiles_b):
            return False
        def normalized(tiles):
            min_r = min(t[0] for t in tiles)
            min_c = min(t[1] for t in tiles)
            return sorted((r - min_r, c - min_c) for r, c in tiles)
        return normalized(tiles_a) == normalized(tiles_b)

    def _shape_signature(self, tiles: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        min_r = min(t[0] for t in tiles)
        min_c = min(t[1] for t in tiles)
        return tuple(sorted((r - min_r, c - min_c) for r, c in tiles))
