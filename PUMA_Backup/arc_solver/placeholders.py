"""Placeholder template detection and reconstruction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .grid import Array

BorderStripe = Optional[Tuple[int, ...]]


@dataclass(frozen=True)
class PlaceholderSignature:
    """Signature describing a placeholder region via borders and colour."""

    placeholder_color: int
    shape: Tuple[int, int]
    left: BorderStripe
    right: BorderStripe
    top: BorderStripe
    bottom: BorderStripe


@dataclass
class PlaceholderTemplate:
    """Stores the fill pattern associated with a placeholder signature."""

    signature: PlaceholderSignature
    fill_pattern: Array
    description: str = "placeholder_fill"


class PlaceholderTemplateEngine:
    """Detects consistent placeholder fills across training examples."""

    def detect_templates(
        self,
        train_pairs: Sequence[Tuple[Array, Array]],
    ) -> List[PlaceholderTemplate]:
        """Return templates that agree across all provided training pairs."""

        if not train_pairs:
            return []

        signature_to_patterns: Dict[PlaceholderSignature, List[Array]] = {}
        signature_counts: Dict[PlaceholderSignature, int] = {}
        total_pairs = len(train_pairs)

        for inp, out in train_pairs:
            regions = self._find_placeholder_regions(inp)
            if not regions:
                continue
            seen_in_pair: set[PlaceholderSignature] = set()
            for candidate in regions:
                signature = self._compute_signature(inp, candidate)
                if signature in seen_in_pair:
                    continue
                r1, c1, r2, c2, _ = candidate
                patch = out[r1:r2, c1:c2]
                signature_to_patterns.setdefault(signature, []).append(patch)
                seen_in_pair.add(signature)
            for signature in seen_in_pair:
                signature_counts[signature] = signature_counts.get(signature, 0) + 1

        templates: List[PlaceholderTemplate] = []

        for signature, patches in signature_to_patterns.items():
            if signature_counts.get(signature, 0) != total_pairs:
                continue
            if not patches:
                continue
            first = patches[0]
            if any(
                patch.shape != first.shape or not np.array_equal(patch, first)
                for patch in patches[1:]
            ):
                continue
            templates.append(
                PlaceholderTemplate(
                    signature=signature,
                    fill_pattern=first.copy(),
                )
            )

        return templates

    def apply_template(self, grid: Array, template: PlaceholderTemplate) -> Optional[Array]:
        """Return a grid with the placeholder region filled according to ``template``."""

        for candidate in self._find_placeholder_regions(grid):
            signature = self._compute_signature(grid, candidate)
            if signature != template.signature:
                continue
            r1, c1, r2, c2, _ = candidate
            if template.fill_pattern.shape != (r2 - r1, c2 - c1):
                continue
            result = grid.copy()
            result[r1:r2, c1:c2] = template.fill_pattern
            return result
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _find_placeholder_regions(self, grid: Array) -> List[Tuple[int, int, int, int, int]]:
        """Return all plausible placeholder regions in ``grid``."""

        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        regions: List[Tuple[int, int, int, int, int]] = []

        total_area = h * w if h and w else 0

        for r in range(h):
            for c in range(w):
                if visited[r, c]:
                    continue
                color = int(grid[r, c])

                # Determine width of contiguous strip for this colour starting here
                region_w = 0
                for cc in range(c, w):
                    if grid[r, cc] == color and not visited[r, cc]:
                        region_w += 1
                    else:
                        break
                if region_w == 0:
                    continue

                # Determine height ensuring perfect rectangle
                region_h = 0
                rectangle = True
                for rr in range(r, h):
                    row_ok = True
                    for cc in range(c, c + region_w):
                        if cc >= w or grid[rr, cc] != color or visited[rr, cc]:
                            row_ok = False
                            break
                    if row_ok:
                        region_h += 1
                    else:
                        break

                if region_h == 0:
                    continue

                for rr in range(r, r + region_h):
                    for cc in range(c, c + region_w):
                        if grid[rr, cc] != color:
                            rectangle = False
                            break
                    if not rectangle:
                        break

                # Mark visited cells regardless so we don't revisit the same block
                for rr in range(r, r + region_h):
                    for cc in range(c, c + region_w):
                        visited[rr, cc] = True

                if not rectangle:
                    continue

                area = region_h * region_w
                if area < 4:
                    continue
                if region_h < 2 or region_w < 2:
                    continue
                if total_area and area / total_area >= 0.6:
                    continue

                regions.append((r, c, r + region_h, c + region_w, color))

        return regions

    def _compute_signature(
        self,
        grid: Array,
        region: Tuple[int, int, int, int, int],
    ) -> PlaceholderSignature:
        r1, c1, r2, c2, color = region
        left = self._border_slice(grid, r1, r2, c1 - 1, axis="col") if c1 > 0 else None
        right = self._border_slice(grid, r1, r2, c2, axis="col") if c2 < grid.shape[1] else None
        top = self._border_slice(grid, c1, c2, r1 - 1, axis="row") if r1 > 0 else None
        bottom = self._border_slice(grid, c1, c2, r2, axis="row") if r2 < grid.shape[0] else None
        return PlaceholderSignature(
            placeholder_color=color,
            shape=(r2 - r1, c2 - c1),
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )

    def _border_slice(
        self,
        grid: Array,
        start: int,
        end: int,
        index: int,
        *,
        axis: str,
    ) -> Tuple[int, ...]:
        if axis == "col":
            return tuple(int(grid[r, index]) for r in range(start, end))
        if axis == "row":
            return tuple(int(grid[index, c]) for c in range(start, end))
        raise ValueError(f"Unknown axis: {axis}")


def _serialise_border(stripe: BorderStripe) -> Optional[List[int]]:
    return list(stripe) if stripe is not None else None


def _deserialise_border(stripe: Optional[Sequence[int]]) -> BorderStripe:
    return tuple(int(v) for v in stripe) if stripe is not None else None


def serialize_placeholder_template(template: PlaceholderTemplate) -> Dict[str, Any]:
    """Return a JSON-serialisable representation of ``template``."""

    signature = template.signature
    return {
        "placeholder_color": int(signature.placeholder_color),
        "shape": [int(dim) for dim in signature.shape],
        "left": _serialise_border(signature.left),
        "right": _serialise_border(signature.right),
        "top": _serialise_border(signature.top),
        "bottom": _serialise_border(signature.bottom),
        "fill_pattern": template.fill_pattern.tolist(),
        "description": template.description,
    }


def deserialize_placeholder_template(payload: Dict[str, Any]) -> PlaceholderTemplate:
    """Reconstruct a :class:`PlaceholderTemplate` from ``payload``."""

    signature = PlaceholderSignature(
        placeholder_color=int(payload["placeholder_color"]),
        shape=tuple(int(dim) for dim in payload["shape"]),
        left=_deserialise_border(payload.get("left")),
        right=_deserialise_border(payload.get("right")),
        top=_deserialise_border(payload.get("top")),
        bottom=_deserialise_border(payload.get("bottom")),
    )
    fill_pattern = np.array(payload["fill_pattern"], dtype=int)
    description = payload.get("description", "placeholder_fill")
    return PlaceholderTemplate(
        signature=signature,
        fill_pattern=fill_pattern,
        description=description,
    )
