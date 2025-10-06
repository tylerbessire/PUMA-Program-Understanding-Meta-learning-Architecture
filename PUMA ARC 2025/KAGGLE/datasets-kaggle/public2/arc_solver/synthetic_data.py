"""Synthetic data generation utilities for ARC neuro-symbolic training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
from typing import List, Sequence

import numpy as np

from .grid import Array, to_list
from .ilp_engine import ILPProgram


def _random_rectangles(rng: random.Random, grid: Array, colors: Sequence[int]) -> Array:
    h, w = grid.shape
    num_blocks = rng.randint(1, 4)
    for _ in range(num_blocks):
        color = rng.choice(colors)
        height = rng.randint(1, max(1, h // 2))
        width = rng.randint(1, max(1, w // 2))
        r = rng.randint(0, h - height)
        c = rng.randint(0, w - width)
        grid[r : r + height, c : c + width] = color
    return grid


def random_grid(
    rng: random.Random,
    height: int = 6,
    width: int = 6,
    colors: Sequence[int] = range(1, 7),
) -> Array:
    grid = np.zeros((height, width), dtype=np.int16)
    return _random_rectangles(rng, grid, colors)


@dataclass
class SyntheticExample:
    input_grid: Array
    program: ILPProgram
    output_grid: Array


class SyntheticDatasetGenerator:
    """Build synthetic (input, program, output) triples for training."""

    def __init__(self, seed: int = 0, colors: Sequence[int] = range(1, 7)) -> None:
        self.rng = random.Random(seed)
        self.colors = list(colors)

    def generate(self, base_program: ILPProgram, count: int = 100) -> List[SyntheticExample]:
        examples: List[SyntheticExample] = []
        for _ in range(count):
            grid = random_grid(self.rng, colors=self.colors)
            output = base_program.apply_to_grid(grid)
            examples.append(SyntheticExample(grid, base_program, output))
        return examples


def save_examples(examples: Sequence[SyntheticExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for example in examples:
        records.append(
            {
                "input": to_list(example.input_grid),
                "criteria": example.program.criteria,
                "ignore_color": example.program.ignore_color,
                "output": to_list(example.output_grid),
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
