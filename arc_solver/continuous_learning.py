"""Continuous learning and self-memory management for the ARC solver."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _canonical_shape(arr: np.ndarray) -> tuple[int, int]:
    if arr.ndim == 1:
        return (1, int(arr.shape[0]))
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D grid, got {arr.shape}")
    return tuple(int(x) for x in arr.shape)


def _analyze_signature(train_pairs: List[tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
    if not train_pairs:
        return {
            "input_size": None,
            "output_size": None,
            "size_change": None,
            "primary_pattern": "unknown",
            "color_change": False,
        }

    inp, out = train_pairs[0]
    input_size = _canonical_shape(inp)
    output_size = _canonical_shape(out)

    signature: Dict[str, Any] = {
        "input_size": input_size,
        "output_size": output_size,
        "size_change": None,
        "primary_pattern": "unknown",
        "color_change": False,
    }

    if input_size == output_size:
        signature["primary_pattern"] = "same_size"
        if np.array_equal(inp, out):
            signature["primary_pattern"] = "identity"
        elif np.array_equal(np.rot90(inp, 1), out) or np.array_equal(np.rot90(inp, 2), out) or np.array_equal(np.rot90(inp, 3), out):
            signature["primary_pattern"] = "rotation"
        elif np.array_equal(np.flipud(inp), out) or np.array_equal(np.fliplr(inp), out):
            signature["primary_pattern"] = "reflection"
        else:
            inp_colors = set(int(x) for x in inp.flatten())
            out_colors = set(int(x) for x in out.flatten())
            if inp_colors != out_colors:
                signature["primary_pattern"] = "recolor"
                signature["color_change"] = True
            else:
                signature["primary_pattern"] = "complex_same_size"
    else:
        signature["size_change"] = f"{input_size[0]}x{input_size[1]}->{output_size[0]}x{output_size[1]}"
        if output_size[0] * output_size[1] >= input_size[0] * input_size[1]:
            signature["primary_pattern"] = "expansion"
        else:
            signature["primary_pattern"] = "extraction"

    distinct_colors = sorted({int(c) for _, out_grid in train_pairs for c in out_grid.flatten()})
    signature["output_palette"] = distinct_colors[:6]
    signature["pair_count"] = len(train_pairs)
    return signature


def _signature_key(signature: Dict[str, Any]) -> str:
    parts = [
        str(signature.get("primary_pattern")),
        str(signature.get("input_size")),
        str(signature.get("output_size")),
        str(signature.get("size_change")),
        str(signature.get("color_change")),
    ]
    return "|".join(parts)


@dataclass
class TaskExperience:
    task_id: str
    signature: Dict[str, Any]
    transformation: Optional[str]
    solved: bool
    timestamp: str
    meta: Dict[str, Any]


class ContinuousSelfMemory:
    """Persistent tracker of solver experiences."""

    def __init__(
        self,
        memory_path: str = "continuous_memory.json",
        bootstrap: bool = True,
        challenges_path: Optional[str] = "data/arc-agi_training_challenges.json",
        solutions_path: Optional[str] = "data/arc-agi_training_solutions.json",
    ):
        self.path = Path(memory_path)
        self.state: Dict[str, Any] = {
            "persona": {
                "name": "PUMA-Continuum",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tasks_recorded": 0,
                "successful_tasks": 0,
            },
            "experiences": [],
            "signature_index": {},
            "signature_stats": {},
            "metadata": {},
        }
        self._load()
        self._rebuild_indices()
        if bootstrap and challenges_path and solutions_path:
            self.bootstrap_from_training_data(challenges_path, solutions_path)

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.state.update(data)

    def _save(self) -> None:
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        tmp_path.replace(self.path)

    def _add_to_index(self, experience: TaskExperience) -> None:
        key = _signature_key(experience.signature)
        idx = len(self.state["experiences"]) - 1
        self.state["signature_index"].setdefault(key, []).append(idx)

    def _update_signature_stats(self, signature: Dict[str, Any], solved: bool) -> None:
        key = _signature_key(signature)
        stats = self.state.setdefault("signature_stats", {}).setdefault(
            key, {"successes": 0, "failures": 0}
        )
        if solved:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

    def _store_experience(self, experience: TaskExperience) -> None:
        self._store_experience(experience)
        self._update_signature_stats(experience.signature, experience.solved)

    def record_experience(
        self,
        task_id: str,
        train_pairs: List[tuple[np.ndarray, np.ndarray]],
        transformation: Optional[str],
        solved: bool,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        signature = _analyze_signature(train_pairs)
        experience = TaskExperience(
            task_id=task_id,
            signature=signature,
            transformation=transformation,
            solved=solved,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            meta=meta or {},
        )
        self.state.setdefault("experiences", []).append(asdict(experience))
        self._add_to_index(experience)

        persona = self.state.setdefault("persona", {})
        persona["tasks_recorded"] = persona.get("tasks_recorded", 0) + 1
        if solved:
            persona["successful_tasks"] = persona.get("successful_tasks", 0) + 1

        self._save()

    def suggest_transformations(
        self, train_pairs: List[tuple[np.ndarray, np.ndarray]], top_k: int = 3
    ) -> List[Dict[str, Any]]:
        signature = _analyze_signature(train_pairs)
        key = _signature_key(signature)
        indices = self.state.get("signature_index", {}).get(key, [])
        if not indices:
            return []
        freq: Dict[str, Dict[str, Any]] = {}
        for idx in indices:
            record = self.state["experiences"][idx]
            transformation = record.get("transformation")
            if not transformation:
                continue
            entry = freq.setdefault(
                transformation,
                {
                    "score": 0,
                    "successes": 0,
                    "failures": 0,
                    "program_sketch": record.get("meta", {}).get("program_sketch"),
                },
            )
            if record.get("solved"):
                entry["score"] += 1
                entry["successes"] += 1
            else:
                entry["score"] -= 1
                entry["failures"] += 1
            if not entry.get("program_sketch") and record.get("meta"):
                entry["program_sketch"] = record["meta"].get("program_sketch")
        ranked = sorted(freq.items(), key=lambda kv: kv[1]["score"], reverse=True)
        filtered = [item for item in ranked if item[1]["score"] > 0] or ranked
        top = filtered[:top_k]
        return [
            {
                "transformation": name,
                "score": data["score"],
                "successes": data["successes"],
                "failures": data["failures"],
                "program_sketch": data.get("program_sketch"),
            }
            for name, data in top
        ]

    def bootstrap_from_training_data(
        self,
        challenges_path: str,
        solutions_path: str,
        limit: Optional[int] = None,
    ) -> None:
        metadata = self.state.setdefault("metadata", {})
        if metadata.get("training_bootstrap"):
            return

        challenges_file = Path(challenges_path)
        solutions_file = Path(solutions_path)
        if not challenges_file.exists() or not solutions_file.exists():
            return

        challenges = json.loads(challenges_file.read_text())
        solutions = json.loads(solutions_file.read_text())

        for idx, (task_id, task) in enumerate(challenges.items()):
            if limit and idx >= limit:
                break
            train_pairs = [
                (np.asarray(pair["input"], dtype=np.int16), np.asarray(pair["output"], dtype=np.int16))
                for pair in task.get("train", [])
            ]
            if not train_pairs:
                continue
            signature = _analyze_signature(train_pairs)
            experience = TaskExperience(
                task_id=task_id,
                signature=signature,
                transformation="ground_truth_reference",
                solved=True,
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                meta={"bootstrap": True, "solutions_present": task_id in solutions},
            )
            self._store_experience(experience)

        metadata["training_bootstrap"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._save()

    def persona_summary(self) -> Dict[str, Any]:
        persona = self.state.get("persona", {})
        total = persona.get("tasks_recorded", 0)
        solved = persona.get("successful_tasks", 0)
        return {
            "name": persona.get("name", "PUMA-Continuum"),
            "tasks_recorded": total,
            "successful_tasks": solved,
            "success_rate": solved / total if total else 0.0,
            "memory_entries": len(self.state.get("experiences", [])),
        }

    def signature_performance(
        self, train_pairs: List[tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        signature = _analyze_signature(train_pairs)
        key = _signature_key(signature)
        stats = self.state.setdefault("signature_stats", {}).get(
            key, {"successes": 0, "failures": 0}
        )
        return {
            **signature,
            "successes": stats.get("successes", 0),
            "failures": stats.get("failures", 0),
        }

    def _rebuild_indices(self) -> None:
        experiences = self.state.get("experiences", [])

        index: Dict[str, List[int]] = {}
        for idx, record in enumerate(experiences):
            signature = record.get("signature", {})
            key = _signature_key(signature)
            index.setdefault(key, []).append(idx)
        self.state["signature_index"] = index

        stats: Dict[str, Dict[str, int]] = {}
        for record in experiences:
            signature = record.get("signature", {})
            key = _signature_key(signature)
            solved = bool(record.get("solved"))
            entry = stats.setdefault(key, {"successes": 0, "failures": 0})
            if solved:
                entry["successes"] += 1
            else:
                entry["failures"] += 1
        self.state["signature_stats"] = stats
