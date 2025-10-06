"""Neuro-symbolic pipeline combining ILP induction and perception training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .grid import Array
from .ilp_engine import ILPExample, ILPProgram, SimpleAttributeILP
from .synthetic_data import SyntheticDatasetGenerator
from .neural.perception import RelationalPerceptionNet, train_perception_model, predict_mask


def _apply_program(program: ILPProgram, grid: Array) -> Array:
    return program.apply_to_grid(grid)


@dataclass
class NeuroSymbolicModel:
    program: ILPProgram
    perception: RelationalPerceptionNet

    def solve(self, grid: Array) -> Array:
        program_output = _apply_program(self.program, grid)
        perception_mask = predict_mask(self.perception, grid)
        result = np.full_like(grid, self.program.ignore_color)
        result[perception_mask > 0] = grid[perception_mask > 0]
        for r, c in zip(*np.where(program_output != self.program.ignore_color)):
            result[r, c] = program_output[r, c]
        return result


class NeuroSymbolicPipeline:
    def __init__(self, ilp: SimpleAttributeILP | None = None) -> None:
        self.ilp = ilp or SimpleAttributeILP()
        self.model: NeuroSymbolicModel | None = None

    def fit(
        self,
        examples: Sequence[ILPExample],
        synthetic_count: int = 200,
        synthetic_seed: int = 0,
        perception_epochs: int = 5,
    ) -> None:
        program = self.ilp.induce(examples)
        if program is None:
            raise ValueError("Unable to induce ILP program from examples")
        generator = SyntheticDatasetGenerator(seed=synthetic_seed)
        synthetic_examples = generator.generate(program, count=synthetic_count)
        perception = train_perception_model(synthetic_examples, epochs=perception_epochs)
        self.model = NeuroSymbolicModel(program, perception)

    def solve(self, grid: Array) -> Array:
        if not self.model:
            raise RuntimeError("Pipeline has not been fitted")
        return self.model.solve(grid)
