"""Glyph extraction helper for large-grid to small-glyph ARC tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Array = np.ndarray


def _canonical_signature(block: Array) -> Tuple[Tuple[int, int], Tuple[int, ...]]:
    """Return a canonical signature for a block based on color rank ordering."""
    values, counts = np.unique(block, return_counts=True)
    # sort by descending count, then ascending color for stability
    order = sorted(zip(-counts, values))
    rank = {int(val): idx for idx, (_, val) in enumerate(order)}
    canonical_flat = tuple(rank[int(v)] for v in block.flatten())
    return block.shape, canonical_flat


def _signature_key(shape: Tuple[int, int], flat: Tuple[int, ...]) -> str:
    return f"{shape[0]}x{shape[1]}:" + ",".join(map(str, flat))


@dataclass
class GlyphConfig:
    top: int
    bottom: int
    left: int
    right: int
    ratio_h: int
    ratio_w: int
    output_shape: Tuple[int, int]
    mapping: Dict[str, int]


class GlyphExtractor:
    """Learns mappings from large canvases to small glyph outputs."""

    def __init__(self) -> None:
        self.configs: List[GlyphConfig] = []
        self._mapping: Dict[str, int] = {}
        self._hist_mapping: Dict[Tuple[int, ...], int] = {}

    @staticmethod
    def _find_minimal_cropping(inp_shape: Tuple[int, int], out_shape: Tuple[int, int], max_trim: int = 15) -> Optional[Tuple[int, int, int, int]]:
        H, W = inp_shape
        h_out, w_out = out_shape
        best: Optional[Tuple[int, int, int, int, int]] = None
        for top in range(max_trim + 1):
            for bottom in range(max_trim + 1):
                for left in range(max_trim + 1):
                    for right in range(max_trim + 1):
                        h = H - top - bottom
                        w = W - left - right
                        if h <= 0 or w <= 0:
                            continue
                        if h % h_out == 0 and w % w_out == 0:
                            removal = top + bottom + left + right
                            candidate = (removal, top, bottom, left, right)
                            if best is None or candidate < best:
                                best = candidate
        if best is None:
            return None
        _, top, bottom, left, right = best
        return top, bottom, left, right

    def train(self, train_pairs: Iterable[Tuple[Array, Array]]) -> bool:
        self.configs = []
        aggregated_mapping: Dict[str, int] = {}
        hist_counter: Dict[Tuple[int, ...], Dict[int, int]] = {}
        for inp, out in train_pairs:
            top_bottom_left_right = self._find_minimal_cropping(inp.shape, out.shape)
            if top_bottom_left_right is None:
                self.configs = []
                return False
            top, bottom, left, right = top_bottom_left_right
            cropped = inp[top:inp.shape[0] - bottom, left:inp.shape[1] - right]
            if cropped.size == 0:
                self.configs = []
                return False
            h_ratio = cropped.shape[0] // out.shape[0]
            w_ratio = cropped.shape[1] // out.shape[1]
            if h_ratio == 0 or w_ratio == 0:
                self.configs = []
                return False

            mapping: Dict[str, int] = {}
            for r in range(out.shape[0]):
                for c in range(out.shape[1]):
                    block = cropped[r * h_ratio:(r + 1) * h_ratio, c * w_ratio:(c + 1) * w_ratio]
                    shape, canonical = _canonical_signature(block)
                    key = _signature_key(shape, canonical)
                    color = int(out[r, c])
                    prev = mapping.get(key)
                    if prev is None:
                        mapping[key] = color
                    elif prev != color:
                        # inconsistent within same pair
                        return False
                    agg_prev = aggregated_mapping.get(key)
                    if agg_prev is None:
                        aggregated_mapping[key] = color
                    elif agg_prev != color:
                        # inconsistent across pairs
                        return False

                    hist = tuple(np.bincount(block.flatten(), minlength=10))
                    hist_counter.setdefault(hist, {}).setdefault(color, 0)
                    hist_counter[hist][color] += 1

            config = GlyphConfig(
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                ratio_h=h_ratio,
                ratio_w=w_ratio,
                output_shape=out.shape,
                mapping=mapping,
            )
            self.configs.append(config)
        self._mapping = aggregated_mapping
        # Build histogram fallback mapping
        self._hist_mapping = {
            hist: max(counts.items(), key=lambda kv: kv[1])[0]
            for hist, counts in hist_counter.items()
        }
        return bool(self.configs)

    def predict(self, grid: Array) -> Optional[Array]:
        G, W = grid.shape
        for config in self.configs:
            h_ratio = config.ratio_h
            w_ratio = config.ratio_w
            min_top, min_bottom, min_left, min_right = config.top, config.bottom, config.left, config.right
            for top_extra in range(0, max(1, (G - h_ratio) // h_ratio + 1) * h_ratio, h_ratio):
                top_crop = min_top + top_extra
                if top_crop >= G:
                    continue
                for bottom_extra in range(0, max(1, (G - top_crop - h_ratio) // h_ratio + 1) * h_ratio, h_ratio):
                    bottom_crop = min_bottom + bottom_extra
                    if top_crop + bottom_crop >= G:
                        continue
                    height = G - top_crop - bottom_crop
                    if height <= 0 or height % h_ratio != 0:
                        continue
                    out_h = height // h_ratio
                    for left_extra in range(0, max(1, (W - w_ratio) // w_ratio + 1) * w_ratio, w_ratio):
                        left_crop = min_left + left_extra
                        if left_crop >= W:
                            continue
                        for right_extra in range(0, max(1, (W - left_crop - w_ratio) // w_ratio + 1) * w_ratio, w_ratio):
                            right_crop = min_right + right_extra
                            if left_crop + right_crop >= W:
                                continue
                            width = W - left_crop - right_crop
                            if width <= 0 or width % w_ratio != 0:
                                continue
                            out_w = width // w_ratio
                            cropped = grid[top_crop:G - bottom_crop, left_crop:W - right_crop]
                            output = np.zeros((out_h, out_w), dtype=int)
                            valid = True
                            for r in range(out_h):
                                for c in range(out_w):
                                    block = cropped[r * h_ratio:(r + 1) * h_ratio, c * w_ratio:(c + 1) * w_ratio]
                                    shape, canonical = _canonical_signature(block)
                                    key = _signature_key(shape, canonical)
                                    color = config.mapping.get(key)
                                    if color is None:
                                        color = self._mapping.get(key)
                                    if color is None:
                                        hist = tuple(np.bincount(block.flatten(), minlength=10))
                                        color = self._hist_mapping.get(hist)
                                    if color is None:
                                        valid = False
                                        break
                                    output[r, c] = color
                                if not valid:
                                    break
                            if valid:
                                return output
        return None
