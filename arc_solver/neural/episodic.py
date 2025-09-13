"""Episodic memory and retrieval for the ARC solver.

This module implements a lightweight yet fully functional episodic memory
system. Previously solved tasks (episodes) are stored together with the
programs that solved them and rich feature representations. A hierarchical
index organises episodes into coarse feature buckets while repeated solutions
are consolidated to avoid unbounded growth. At inference time the solver can
query this database for tasks with similar signatures or feature vectors and
reuse their solutions as candidates.

The implementation is intentionally deterministic and avoids any external
dependencies so that it remains compatible with the Kaggle competition
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import json
import os

import numpy as np

from ..grid import Array
from ..features import compute_task_signature, extract_task_features

# Type alias for a DSL program representation used across the project
Program = List[Tuple[str, Dict[str, Any]]]


@dataclass
class Episode:
    """Representation of a single solved ARC task.

    Attributes:
        task_signature: Canonical signature of the training pairs.
        programs: List of successful programs for this task.
        task_id: Optional identifier of the original task.
        train_pairs: Training input/output pairs that produced the solution.
        success_count: Number of times this episode led to a correct solution.
        metadata: Additional arbitrary information.
        features: Cached feature dictionary used for similarity comparisons.
    """

    task_signature: str
    programs: List[Program]
    task_id: str = ""
    train_pairs: List[Tuple[Array, Array]] = field(default_factory=list)
    success_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Compute feature representation for this episode."""
        try:
            self.features = (
                extract_task_features(self.train_pairs) if self.train_pairs else {}
            )
        except Exception as exc:  # pragma: no cover - defensive programming
            raise ValueError(f"invalid training pairs for episode: {exc}") from exc

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the episode to a plain Python dictionary."""
        return {
            "task_signature": self.task_signature,
            "programs": [
                [(op, params) for op, params in program]
                for program in self.programs
            ],
            "task_id": self.task_id,
            "train_pairs": [
                (inp.tolist(), out.tolist()) for inp, out in self.train_pairs
            ],
            "success_count": self.success_count,
            "metadata": self.metadata,
            "features": self.features,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Reconstruct an :class:`Episode` from a dictionary."""
        train_pairs = [
            (np.array(inp, dtype=int), np.array(out, dtype=int))
            for inp, out in data.get("train_pairs", [])
        ]
        programs: List[Program] = []
        for program in data.get("programs", []):
            prog_ops: Program = []
            for op, params in program:
                if op == "recolor":
                    mapping = params.get("mapping") or params.get("color_map") or {}
                    params = {"mapping": {int(k): int(v) for k, v in mapping.items()}}
                prog_ops.append((op, params))
            programs.append(prog_ops)

        episode = cls(
            task_signature=data["task_signature"],
            programs=programs,
            task_id=data.get("task_id", ""),
            train_pairs=train_pairs,
            success_count=data.get("success_count", 1),
            metadata=data.get("metadata", {}),
        )
        # If features were stored, reuse them to avoid recomputation
        if "features" in data:
            episode.features = data["features"]
        return episode


class EpisodeDatabase:
    """Persistent store for :class:`Episode` objects."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.episodes: Dict[int, Episode] = {}
        self.signature_index: Dict[str, List[int]] = defaultdict(list)
        self.program_index: Dict[str, List[int]] = defaultdict(list)
        # Hierarchical index groups episodes by coarse feature buckets.
        # This enables fast retrieval of structurally similar tasks while
        # keeping the system deterministic and lightweight.
        self.hierarchy_index: Dict[str, List[int]] = defaultdict(list)
        self.db_path = db_path
        self._next_id = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _program_key(program: Program) -> str:
        """Create a hashable key for a program.

        Parameters are sorted to guarantee determinism.
        """
        normalised = [
            (op, tuple(sorted(params.items()))) for op, params in program
        ]
        return json.dumps(normalised)

    def _hierarchy_key(self, features: Dict[str, Any]) -> str:
        """Return a coarse key used for hierarchical organisation.

        The key buckets episodes by basic properties such as number of
        training pairs, average input colours and whether recolouring is
        likely.  These buckets act as top-level memory regions that group
        broadly similar tasks.
        """

        num_pairs = int(features.get("num_train_pairs", 0))
        colours = int(features.get("input_colors_mean", 0))
        recolor = int(bool(features.get("likely_recolor", False)))
        return f"{num_pairs}:{colours}:{recolor}"

    def _compute_similarity(self, f1: Dict[str, Any], f2: Dict[str, Any]) -> float:
        """Compute cosine similarity between two feature dictionaries."""
        numerical_keys = [
            "num_train_pairs",
            "input_height_mean",
            "input_width_mean",
            "output_height_mean",
            "output_width_mean",
            "size_ratio_mean",
            "input_colors_mean",
            "output_colors_mean",
            "color_mapping_size",
            "input_objects_mean",
            "output_objects_mean",
            "object_count_preserved",
            "likely_rotation",
            "likely_reflection",
            "likely_translation",
            "likely_recolor",
            "likely_crop",
            "likely_pad",
        ]

        v1 = np.array([float(f1.get(k, 0.0)) for k in numerical_keys])
        v2 = np.array([float(f2.get(k, 0.0)) for k in numerical_keys])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = float(np.dot(v1, v2) / (norm1 * norm2))

        # Add a small bonus for matching boolean properties
        boolean_keys = [
            "shape_preserved",
            "background_color_consistent",
            "has_color_mapping",
        ]
        matches = sum(
            1
            for k in boolean_keys
            if bool(f1.get(k, False)) == bool(f2.get(k, False))
        )
        cosine_sim += matches / len(boolean_keys) * 0.2
        return min(cosine_sim, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def store_episode(
        self,
        task_signature: str,
        programs: List[Program],
        task_id: str,
        train_pairs: List[Tuple[Array, Array]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a solved episode and return its identifier."""
        episode = Episode(
            task_signature=task_signature,
            programs=programs,
            task_id=task_id,
            train_pairs=train_pairs,
            metadata=metadata or {},
        )

        episode_id = self._next_id
        self._next_id += 1

        self.episodes[episode_id] = episode
        self.signature_index[task_signature].append(episode_id)
        for program in programs:
            key = self._program_key(program)
            self.program_index[key].append(episode_id)
        hier_key = self._hierarchy_key(episode.features)
        self.hierarchy_index[hier_key].append(episode_id)

        return episode_id

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """Retrieve an episode by its identifier."""
        return self.episodes.get(episode_id)

    def query_by_signature(self, signature: str) -> List[Episode]:
        """Return all episodes with the given task signature."""
        ids = self.signature_index.get(signature, [])
        return [self.episodes[i] for i in ids]

    def query_by_similarity(
        self,
        train_pairs: List[Tuple[Array, Array]],
        similarity_threshold: float = 0.5,
        max_results: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """Return episodes whose feature similarity exceeds the threshold."""

        if not train_pairs:
            return []
        query_features = extract_task_features(train_pairs)
        results: List[Tuple[Episode, float]] = []
        for episode in self.episodes.values():
            similarity = self._compute_similarity(query_features, episode.features)
            if similarity >= similarity_threshold:
                results.append((episode, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def query_hierarchy(
        self,
        train_pairs: List[Tuple[Array, Array]],
        similarity_threshold: float = 0.5,
        max_results: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """Return episodes from the same hierarchical bucket.

        Episodes are grouped into coarse buckets based on simple features.
        This allows a two-level lookup: first by bucket, then by detailed
        similarity within that bucket.
        """

        if not train_pairs:
            return []
        query_features = extract_task_features(train_pairs)
        key = self._hierarchy_key(query_features)
        ids = self.hierarchy_index.get(key, [])
        results: List[Tuple[Episode, float]] = []
        for eid in ids:
            episode = self.episodes[eid]
            similarity = self._compute_similarity(query_features, episode.features)
            if similarity >= similarity_threshold:
                results.append((episode, similarity))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def get_candidate_programs(
        self, train_pairs: List[Tuple[Array, Array]], max_programs: int = 10
    ) -> List[Program]:
        """Return programs from similar episodes for reuse."""
        candidates: List[Program] = []
        results = self.query_hierarchy(train_pairs, 0.0, max_programs)
        if not results:
            results = self.query_by_similarity(train_pairs, 0.0, max_programs)
        for episode, _ in results:
            for program in episode.programs:
                candidates.append(program)
                if len(candidates) >= max_programs:
                    return candidates
        return candidates

    def remove_episode(self, episode_id: int) -> None:
        """Remove an episode from the database."""
        episode = self.episodes.pop(episode_id, None)
        if not episode:
            return
        self.signature_index[episode.task_signature] = [
            i for i in self.signature_index[episode.task_signature] if i != episode_id
        ]
        for program in episode.programs:
            key = self._program_key(program)
            self.program_index[key] = [
                i for i in self.program_index[key] if i != episode_id
            ]
        hier_key = self._hierarchy_key(episode.features)
        self.hierarchy_index[hier_key] = [
            i for i in self.hierarchy_index[hier_key] if i != episode_id
        ]

    def consolidate(self) -> None:
        """Merge episodes with identical signature and program set."""

        signature_map: Dict[Tuple[str, str], int] = {}
        to_remove: List[int] = []
        for eid, episode in self.episodes.items():
            program_key = json.dumps(
                sorted(self._program_key(p) for p in episode.programs)
            )
            key = (episode.task_signature, program_key)
            if key in signature_map:
                target_id = signature_map[key]
                self.episodes[target_id].success_count += episode.success_count
                to_remove.append(eid)
            else:
                signature_map[key] = eid

        for eid in to_remove:
            self.remove_episode(eid)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, filepath: Optional[str] = None) -> None:
        """Persist database to a JSON file."""
        save_path = filepath or self.db_path
        if not save_path:
            return
        data = {
            "next_id": self._next_id,
            "episodes": {str(eid): ep.to_dict() for eid, ep in self.episodes.items()},
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Optional[str] = None) -> None:
        """Load database from a JSON file."""
        load_path = filepath or self.db_path
        if not load_path or not os.path.exists(load_path):
            return
        try:
            with open(load_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return  # Treat malformed file as empty database

        self._next_id = int(data.get("next_id", 1))
        self.episodes = {
            int(eid): Episode.from_dict(ep_data)
            for eid, ep_data in data.get("episodes", {}).items()
        }

        # Rebuild indexes deterministically
        self.signature_index.clear()
        self.program_index.clear()
        self.hierarchy_index.clear()
        for eid, episode in self.episodes.items():
            self.signature_index[episode.task_signature].append(eid)
            for program in episode.programs:
                key = self._program_key(program)
                self.program_index[key].append(eid)
            hier_key = self._hierarchy_key(episode.features)
            self.hierarchy_index[hier_key].append(eid)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_statistics(self) -> Dict[str, Any]:
        """Return basic database statistics."""
        total_episodes = len(self.episodes)
        total_programs = sum(len(ep.programs) for ep in self.episodes.values())
        avg_programs = (
            float(total_programs) / total_episodes if total_episodes else 0.0
        )
        return {
            "total_episodes": total_episodes,
            "unique_signatures": len(self.signature_index),
            "total_programs": total_programs,
            "average_programs_per_episode": avg_programs,
        }


class EpisodicRetrieval:
    """High-level interface used by the solver to access episodic memory."""

    def __init__(self, db_path: str = "episodes.json") -> None:
        self.database = EpisodeDatabase(db_path)
        self.cache: Dict[str, List[Program]] = {}

    def query_for_programs(
        self, train_pairs: List[Tuple[Array, Array]], max_candidates: int = 5
    ) -> List[Program]:
        """Return candidate programs similar to the given training pairs."""

        if not train_pairs:
            return []
        cache_key = compute_task_signature(train_pairs)
        if cache_key in self.cache:
            return self.cache[cache_key]

        candidates = self.database.get_candidate_programs(train_pairs, max_candidates)
        self.cache[cache_key] = candidates
        if len(self.cache) > 100:
            self.cache.clear()
        return candidates

    def add_successful_solution(
        self,
        train_pairs: List[Tuple[Array, Array]],
        programs: List[Program],
        task_id: str = "",
    ) -> None:
        """Store a successful solution in the episodic database."""

        signature = compute_task_signature(train_pairs)
        self.database.store_episode(signature, programs, task_id, train_pairs)
        self.cache.clear()

    def save(self) -> None:
        """Persist the underlying database."""
        self.database.save()

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the episodic database."""
        return self.database.get_statistics()


class AnalogicalReasoner:
    """Advanced analogical reasoning for ARC tasks."""

    def find_structural_analogies(
        self,
        current_task: List[Tuple[Array, Array]],
        memory: EpisodicRetrieval,
        threshold: float = 0.6,
        max_results: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """Find tasks with similar abstract structure, not just surface features."""

        if not current_task:
            return []
        query_features = extract_task_features(current_task)
        results: List[Tuple[Episode, float]] = []
        for episode in memory.database.episodes.values():
            sim = memory.database._compute_similarity(query_features, episode.features)
            if sim >= threshold:
                results.append((episode, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def map_solution_structure(
        self,
        source_solution: Program,
        target_task: List[Tuple[Array, Array]],
    ) -> Program:
        """Map solution from analogous task to current task."""

        if not source_solution:
            return []
        recolor_mapping: Dict[int, int] = {}
        if target_task:
            inp, out = target_task[0]
            for ci, co in zip(inp.flat, out.flat):
                if ci != co:
                    recolor_mapping[int(ci)] = int(co)
        mapped: Program = []
        for op, params in source_solution:
            if op == "recolor" and recolor_mapping:
                mapped.append((op, {"mapping": recolor_mapping}))
            else:
                mapped.append((op, params))
        return mapped

    def abstract_common_patterns(
        self, similar_tasks: List[List[Tuple[Array, Array]]]
    ) -> Dict[str, float]:
        """Extract abstract transformation rules from multiple similar tasks."""

        if not similar_tasks:
            return {}
        feature_dicts = [extract_task_features(t) for t in similar_tasks if t]
        if not feature_dicts:
            return {}
        keys = set().union(*(fd.keys() for fd in feature_dicts))
        pattern: Dict[str, float] = {}
        for k in keys:
            vals = [float(fd.get(k, 0.0)) for fd in feature_dicts]
            pattern[k] = float(np.mean(vals))
        return pattern

