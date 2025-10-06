"""Build canonicalised ARC training dataset.

[S:DATA v1] builder pass
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np

from arc_solver.canonical import canonicalize_pair

logger = logging.getLogger(__name__)


def build_dataset(
    challenges_path: Path = Path("data/arc-agi_training_challenges.json"),
    output_dir: Path = Path("data"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ARC training challenges and save canonicalised grids.

    Parameters
    ----------
    challenges_path:
        Path to ``arc-agi_training_challenges.json``.
    output_dir:
        Directory in which to save ``train_X.npy`` and ``train_Y.npy``.
    """
    with challenges_path.open("r", encoding="utf-8") as f:
        challenges = json.load(f)

    train_X: list[np.ndarray] = []
    train_Y: list[np.ndarray] = []
    for task_id, task in challenges.items():
        for example in task.get("train", []):
            inp = np.array(example["input"], dtype=np.int16)
            out = np.array(example["output"], dtype=np.int16)
            inp_c, out_c = canonicalize_pair(inp, out)
            train_X.append(inp_c)
            train_Y.append(out_c)
    assert len(train_X) == len(train_Y)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_X.npy", np.array(train_X, dtype=object))
    np.save(output_dir / "train_Y.npy", np.array(train_Y, dtype=object))
    logger.info(
        "Processed %d examples across %d tasks", len(train_X), len(challenges)
    )
    logger.info(
        "Saved dataset to %s and %s", output_dir / "train_X.npy", output_dir / "train_Y.npy"
    )
    return np.array(train_X, dtype=object), np.array(train_Y, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonicalised ARC dataset")
    parser.add_argument(
        "--challenges",
        type=Path,
        default=Path("data/arc-agi_training_challenges.json"),
        help="Path to training challenges JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save train_X.npy and train_Y.npy",
    )
    args = parser.parse_args()

    build_dataset(args.challenges, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
