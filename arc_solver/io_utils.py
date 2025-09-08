"""
Input/output helpers for the ARC solver.

This module provides utilities for loading the Kaggle rerun JSON and writing the
submission JSON in the proper format expected by the competition. It looks for
the test challenge file in common locations and falls back to raising an
exception if none is found.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any


# Potential paths where the Kaggle evaluation file might reside. These
# correspond to typical Kaggle dataset mount points.
CANDIDATE_PATHS = [
    "arc-agi_test_challenges.json",
    "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    "/kaggle/input/arc-agi-2/arc-agi_test_challenges.json",
]


def load_rerun_json() -> Dict[str, Any]:
    """Load the JSON file containing all test tasks for the competition.

    Returns a dictionary keyed by task id. Raises FileNotFoundError if none
    of the candidate paths exist.
    """
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError(
        "Could not find arc-agi_test_challenges.json in expected locations."
    )


def save_submission(obj: Dict[str, Any], out_path: str = "submission.json") -> str:
    """Write the solutions object to a JSON file.

    Returns the path to the written file for convenience.
    """
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)
    return out_path