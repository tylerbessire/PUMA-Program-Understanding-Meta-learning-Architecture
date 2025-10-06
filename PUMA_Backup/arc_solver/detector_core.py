"""Shared detector primitives for the ARC solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DetectionSignal:
    """A pattern detector's output signal."""

    confidence: float
    hypothesis: Dict[str, Any]
    detector_name: str
    priority: int = 0


class BaseDetector:
    """Minimal base class for detectors."""

    def __init__(self, name: str, priority: int = 0) -> None:
        self.name = name
        self.priority = priority

    def scan(
        self, input_feats: Dict[str, Any], output_feats: Dict[str, Any]
    ) -> DetectionSignal | None:
        raise NotImplementedError

