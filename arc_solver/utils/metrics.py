"""Metrics utilities with deterministic moving averages."""

from __future__ import annotations

from collections import deque
from typing import Deque


class MovingAverage:
    """Fixed-window moving average with deterministic updates."""

    def __init__(self, window: int) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self._buffer: Deque[float] = deque(maxlen=window)
        self._running_sum: float = 0.0

    def add_sample(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError("value must be numeric")
        if len(self._buffer) == self.window:
            oldest = self._buffer[0]
            self._running_sum -= oldest
        self._buffer.append(float(value))
        self._running_sum += float(value)

    @property
    def value(self) -> float:
        if not self._buffer:
            return 0.0
        return self._running_sum / float(len(self._buffer))


# [S:UTIL v1] component=moving_average windowed pass
