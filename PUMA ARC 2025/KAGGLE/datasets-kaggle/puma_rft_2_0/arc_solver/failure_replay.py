"""Failure remediation pipeline using neuro-symbolic induction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import json

from .failure_learning import ProgramBank
from .grid import to_array
from .ilp_engine import ILPExample
from .neuro_symbolic import NeuroSymbolicPipeline


def _load_failure_ids(log_path: Path) -> Set[str]:
    task_ids: Set[str] = set()
    if not log_path.exists():
        return task_ids
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = record.get("task_id")
            if task_id:
                task_ids.add(task_id)
    return task_ids


def remediate_failures(
    *,
    failure_log_path: Path,
    challenges_path: Path,
    program_bank_path: Path,
    synthetic_count: int = 200,
    synthetic_seed: int = 0,
    perception_epochs: int = 3,
) -> Dict[str, int]:
    failure_ids = _load_failure_ids(failure_log_path)
    if not failure_ids:
        return {"processed": 0, "stored": 0, "missing": 0}

    challenges = json.loads(challenges_path.read_text(encoding="utf-8"))
    bank = ProgramBank(program_bank_path)

    processed = stored = missing = 0

    for task_id in sorted(failure_ids):
        if bank.load(task_id):
            continue
        challenge = challenges.get(task_id)
        if not challenge:
            missing += 1
            continue

        train_pairs = challenge.get("train", [])
        if not train_pairs:
            missing += 1
            continue

        examples = []
        for pair in train_pairs:
            try:
                inp = to_array(pair["input"])
                out = to_array(pair["output"])
            except Exception:
                continue
            examples.append(ILPExample(inp, out))

        if not examples:
            missing += 1
            continue

        pipeline = NeuroSymbolicPipeline()
        try:
            pipeline.fit(
                examples,
                synthetic_count=synthetic_count,
                synthetic_seed=synthetic_seed,
                perception_epochs=perception_epochs,
            )
        except Exception:
            continue

        if not pipeline.model:
            continue

        bank.save(task_id, pipeline.model.program)
        processed += 1
        stored += 1

    return {
        "processed": processed,
        "stored": stored,
        "missing": missing,
    }
