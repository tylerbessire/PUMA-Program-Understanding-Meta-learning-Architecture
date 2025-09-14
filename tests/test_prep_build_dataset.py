"""Tests for dataset preparation script.

[S:TEST v1] unit=1 integration=1 pass
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prep_build_dataset import build_dataset
from arc_solver.canonical import canonicalize_pair


def test_build_dataset(tmp_path: Path) -> None:
    X, Y = build_dataset(output_dir=tmp_path)
    assert len(X) == len(Y) > 0
    x_path = tmp_path / "train_X.npy"
    y_path = tmp_path / "train_Y.npy"
    assert x_path.exists() and y_path.exists()
    cx, cy = canonicalize_pair(X[0], Y[0])
    assert np.array_equal(cx, X[0])
    assert np.array_equal(cy, Y[0])
