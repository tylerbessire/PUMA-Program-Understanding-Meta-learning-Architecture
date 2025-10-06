"""Perception module utilities for neuro-symbolic ARC solver."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..grid import Array
from ..synthetic_data import SyntheticExample


def _one_hot(batch: torch.Tensor, num_classes: int) -> torch.Tensor:
    return (
        torch.nn.functional.one_hot(batch.long(), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )


class SyntheticMaskDataset(Dataset):
    def __init__(self, examples: Sequence[SyntheticExample]):
        self.inputs = [torch.tensor(ex.input_grid, dtype=torch.float32) for ex in examples]
        self.targets = [
            torch.tensor((ex.output_grid != ex.program.ignore_color).astype("float32"))
            for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx].unsqueeze(0)


class RelationalPerceptionNet(nn.Module):
    def __init__(self, num_colors: int = 10, hidden: int = 16) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.conv = nn.Sequential(
            nn.Conv2d(num_colors, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_one_hot(x, self.num_colors))


def train_perception_model(
    examples: Sequence[SyntheticExample],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_colors: int = 10,
    device: str | torch.device = "cpu",
) -> RelationalPerceptionNet:
    dataset = SyntheticMaskDataset(examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RelationalPerceptionNet(num_colors=num_colors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(max(1, epochs)):
        for grids, masks in dataloader:
            grids = grids.to(device)
            masks = masks.to(device)
            logits = model(grids)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def predict_mask(model: RelationalPerceptionNet, grid: Array) -> Array:
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0)
        logits = model(tensor)
        mask = torch.sigmoid(logits).squeeze(0).squeeze(0)
        return (mask > 0.5).cpu().numpy().astype(grid.dtype)
