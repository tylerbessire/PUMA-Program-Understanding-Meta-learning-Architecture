"""Failure logging and program bank utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import json

from .grid import Array, to_list
from .ilp_engine import ILPProgram


def _ensure_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class FailureLog:
    """Append-only log storing failed task attempts."""

    def __init__(self, path: Path | str) -> None:
        self.path = _ensure_path(Path(path))

    def record(
        self,
        *,
        task_id: str,
        attempt: int,
        input_grid: Array,
        output_grid: Array,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "attempt": attempt,
            "reason": reason,
            "metadata": metadata or {},
            "input": to_list(input_grid),
            "output": to_list(output_grid),
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")


class ProgramBank:
    """Store induced programs for later reuse."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, task_id: str, program: ILPProgram) -> Path:
        record = {
            "task_id": task_id,
            "criteria": {k: int(v) for k, v in program.criteria.items()},
            "ignore_color": int(program.ignore_color),
        }
        path = self.directory / f"{task_id}.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path

    def load(self, task_id: str) -> Optional[ILPProgram]:
        path = self.directory / f"{task_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ILPProgram(data.get("criteria", {}), ignore_color=data.get("ignore_color", 0))

    def list_programs(self) -> Iterable[str]:
        for path in sorted(self.directory.glob("*.json")):
            yield path.stem
