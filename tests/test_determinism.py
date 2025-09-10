"""Tests for deterministic execution of the submission script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_once(payload: dict, workdir: Path) -> bytes:
    """Run arc_submit.py once with the given payload."""
    challenge_path = workdir / "arc-agi_test_challenges.json"
    challenge_path.write_text(json.dumps(payload))
    process = subprocess.run(
        [sys.executable, str(Path(__file__).parent.parent / "arc_submit.py")],
        cwd=str(workdir),
        env={"ARC_TIMEOUT_SEC": "2", **os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return process.stdout


def test_two_runs_identical(tmp_path: Path) -> None:
    """Arc_submit should produce identical output across runs."""
    payload = {
        "t1": {
            "train": [{"input": [[1]], "output": [[1]]}],
            "test": [{"input": [[1]]}],
        }
    }
    output1 = _run_once(payload, tmp_path)
    output2 = _run_once(payload, tmp_path)
    assert output1 == output2

