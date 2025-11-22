"""
RFT Reasoning Engine

Implements Relational Frame Theory for analogical reasoning and relation derivation.
Learns arbitrary relations and enables derivation of new relations without explicit training.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import defaultdict


class RelationType(Enum):
    """Types of relational frames"""
    COORDINATION = "coordination"  # Similarity (X is like Y)
    OPPOSITION = "opposition"  # Difference (X is opposite of Y)
    HIERARCHY = "hierarchy"  # Categorization (X is a type of Y)
    TEMPORAL = "temporal"  # Before/after (X happens before Y)
    CAUSAL = "causal"  # If-then (X causes Y)
    COMPARATIVE = "comparative"  # More/less (X is more than Y)
    SPATIAL = "spatial"  # Location (X is near Y)


@dataclass
class RelationalFrame:
    """
    A learned relational frame between stimuli.
    Enables derivation: if A relates to B, and B relates to C,
    then A relates to C (transitivity).
    """
    relation_type: RelationType
    source: str
    target: str
    strength: float
    context: Optional[List[str]] = None
    derived: bool = False  # Was this derived vs. directly learned?

    def to_dict(self) -> Dict:
        return {
            'relation_type': self.relation_type.value,
            'source': self.source,
            'target': self.target,
            'strength': self.strength,
            'context': self.context,
            'derived': self.derived
        }


class RFTEngine:
    """
    Relational Frame Theory reasoning engine.
    Learns relational frames and derives new relations through transformation.
    """

    def __init__(self, atomspace=None):
        self.atomspace = atomspace
        self.frames: List[RelationalFrame] = []
        self.frame_index: Dict[str, List[RelationalFrame]] = defaultdict(list)

    def derive_relations(self, episodes: List) -> List[RelationalFrame]:
        """
        Discover relational patterns in episodes.

        Args:
            episodes: List of Episode objects

        Returns:
            List of discovered relational frames
        """
        relations = []

        # Find coordination frames (similarity)
        similarities = self.find_coordination_frames(episodes)
        relations.extend(similarities)

        # Find opposition frames
        oppositions = self.find_opposition_frames(episodes)
        relations.extend(oppositions)

        # Find hierarchical relations
        hierarchies = self.find_hierarchy_frames(episodes)
        relations.extend(hierarchies)

        # Find temporal relations
        temporal = self.find_temporal_frames(episodes)
        relations.extend(temporal)

        # Find causal relations
        causal = self.find_causal_frames(episodes)
        relations.extend(causal)

        # Store frames
        for frame in relations:
            self.add_frame(frame)

        return relations

    def find_coordination_frames(self, episodes: List) -> List[RelationalFrame]:
        """
        Find similarity relations (X is like Y).
        Basis for analogical reasoning.
        """
        frames = []

        # Group episodes by similar outcomes
        outcome_groups = defaultdict(list)
        for episode in episodes:
            if episode.outcome:
                # Simple grouping by success/failure
                outcome_key = 'success' if episode.outcome.get('success') else 'failure'
                outcome_groups[outcome_key].append(episode)

        # Episodes in same group are similar
        for group_episodes in outcome_groups.values():
            if len(group_episodes) >= 2:
                # Create coordination frames between similar episodes
                for i in range(len(group_episodes) - 1):
                    frame = RelationalFrame(
                        relation_type=RelationType.COORDINATION,
                        source=group_episodes[i].id,
                        target=group_episodes[i + 1].id,
                        strength=0.7,
                        context=['similar_outcome']
                    )
                    frames.append(frame)

        return frames

    def find_opposition_frames(self, episodes: List) -> List[RelationalFrame]:
        """
        Find opposition relations (X is opposite of Y).
        """
        frames = []

        # Find episodes with opposite emotional valences
        positive_episodes = [e for e in episodes if e.emotional_valence > 0.5]
        negative_episodes = [e for e in episodes if e.emotional_valence < -0.5]

        # Create opposition frames
        for pos_ep in positive_episodes:
            for neg_ep in negative_episodes:
                frame = RelationalFrame(
                    relation_type=RelationType.OPPOSITION,
                    source=pos_ep.id,
                    target=neg_ep.id,
                    strength=0.6,
                    context=['opposite_valence']
                )
                frames.append(frame)
                break  # Only create one example per positive episode

        return frames

    def find_hierarchy_frames(self, episodes: List) -> List[RelationalFrame]:
        """
        Find hierarchical relations (X is a type of Y).
        """
        frames = []

        # Group episodes by type
        type_groups = defaultdict(list)
        for episode in episodes:
            type_groups[episode.type.value].append(episode)

        # All episodes of a type are instances of that type
        for episode_type, group_episodes in type_groups.items():
            for episode in group_episodes:
                frame = RelationalFrame(
                    relation_type=RelationType.HIERARCHY,
                    source=episode.id,
                    target=f"category:{episode_type}",
                    strength=1.0,
                    context=['type_hierarchy']
                )
                frames.append(frame)

        return frames

    def find_temporal_frames(self, episodes: List) -> List[RelationalFrame]:
        """
        Find temporal relations (X before Y).
        """
        frames = []

        # Sort by timestamp
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)

        # Create temporal frames for consecutive episodes
        for i in range(len(sorted_episodes) - 1):
            frame = RelationalFrame(
                relation_type=RelationType.TEMPORAL,
                source=sorted_episodes[i].id,
                target=sorted_episodes[i + 1].id,
                strength=1.0,  # Temporal order is definite
                context=['chronological']
            )
            frames.append(frame)

        return frames

    def find_causal_frames(self, episodes: List) -> List[RelationalFrame]:
        """
        Find causal relations (X causes Y).
        If action in episode N, outcome in episode N+1 -> potential causation.
        """
        frames = []

        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)

        for i in range(len(sorted_episodes) - 1):
            current = sorted_episodes[i]
            next_ep = sorted_episodes[i + 1]

            # If current has action and next has outcome, potential causation
            if current.action and next_ep.outcome:
                frame = RelationalFrame(
                    relation_type=RelationType.CAUSAL,
                    source=current.id,
                    target=next_ep.id,
                    strength=0.5,  # Uncertain - correlation not causation
                    context=['potential_causation']
                )
                frames.append(frame)

        return frames

    def add_frame(self, frame: RelationalFrame):
        """Add frame to knowledge base"""
        self.frames.append(frame)
        self.frame_index[frame.source].append(frame)
        self.frame_index[frame.target].append(frame)

    def derive_by_transitivity(
        self,
        relation_type: RelationType,
        start: str,
        end: str
    ) -> Optional[RelationalFrame]:
        """
        Derive new relation through transitivity.
        If A->B and B->C, then A->C (for transitive relations).
        """
        # Find path from start to end
        path = self._find_relation_path(start, end, relation_type)

        if path and len(path) >= 2:
            # Can derive relation
            # Strength decays with path length
            strength = 1.0 / len(path)

            derived_frame = RelationalFrame(
                relation_type=relation_type,
                source=start,
                target=end,
                strength=strength,
                derived=True,
                context=['derived_by_transitivity']
            )

            return derived_frame

        return None

    def _find_relation_path(
        self,
        start: str,
        end: str,
        relation_type: RelationType,
        max_depth: int = 3
    ) -> Optional[List[str]]:
        """
        Find path of relations from start to end.
        Simple BFS.
        """
        if start == end:
            return [start]

        visited = set()
        queue = [(start, [start])]

        while queue and len(queue[0][1]) <= max_depth:
            current, path = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            # Get outgoing frames
            for frame in self.frame_index.get(current, []):
                if frame.relation_type == relation_type and frame.source == current:
                    new_path = path + [frame.target]

                    if frame.target == end:
                        return new_path

                    queue.append((frame.target, new_path))

        return None

    def reason_by_analogy(self, novel_situation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use past experience to understand new situations.
        Find similar past episodes via relational frames.
        """
        # Find similar episodes using coordination frames
        similar_frames = [
            f for f in self.frames
            if f.relation_type == RelationType.COORDINATION
        ]

        if not similar_frames:
            return {'prediction': None, 'confidence': 0.0}

        # For now, simple heuristic - return most recent similar frame
        most_recent = max(similar_frames, key=lambda f: f.strength)

        return {
            'prediction': f"Similar to {most_recent.source}",
            'confidence': most_recent.strength,
            'analogy_source': most_recent.source
        }

    def get_related_items(
        self,
        item: str,
        relation_type: Optional[RelationType] = None
    ) -> List[RelationalFrame]:
        """
        Get all items related to given item.
        """
        frames = self.frame_index.get(item, [])

        if relation_type:
            frames = [f for f in frames if f.relation_type == relation_type]

        return frames

    def get_frame_statistics(self) -> Dict[str, int]:
        """Get statistics about learned frames"""
        stats = defaultdict(int)

        for frame in self.frames:
            stats[frame.relation_type.value] += 1
            if frame.derived:
                stats['derived'] += 1
            else:
                stats['learned'] += 1

        stats['total'] = len(self.frames)

        return dict(stats)
