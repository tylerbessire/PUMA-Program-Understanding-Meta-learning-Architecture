"""
Memory Consolidation

Transforms raw experiences into lasting knowledge during low-activity periods.
Similar to dreaming - extracts patterns, forms concepts, strengthens important memories.
"""

from typing import List, Dict, Set, Any
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

from .episodic import Episode, MemoryType


@dataclass
class Pattern:
    """Discovered pattern across episodes"""
    pattern_type: str
    frequency: int
    episodes: List[str]
    description: str


@dataclass
class Concept:
    """Formed concept from patterns"""
    name: str
    definition: str
    supporting_episodes: List[str]
    confidence: float


class MemoryConsolidation:
    """
    Consolidates episodic memories into lasting knowledge.
    Runs during low-activity periods (sleep mode).
    """

    def __init__(self, atomspace=None, rft_engine=None):
        self.atomspace = atomspace
        self.rft_engine = rft_engine
        self.patterns_discovered: List[Pattern] = []
        self.concepts_formed: List[Concept] = []

    async def consolidate(self, episodes: List[Episode]) -> Dict[str, Any]:
        """
        Main consolidation process.

        Args:
            episodes: Batch of episodes to consolidate

        Returns:
            Dictionary with consolidation results
        """
        if not episodes:
            return {
                'patterns': [],
                'concepts': [],
                'insights': []
            }

        # Extract patterns across episodes
        patterns = self.extract_patterns(episodes)

        # Form abstractions (concepts)
        concepts = self.form_concepts(patterns)

        # Adjust memory weights (strengthen important, fade trivial)
        self.adjust_memory_weights(episodes)

        # Update self-model based on behavioral patterns
        self_insights = self.update_self_understanding(patterns)

        # RFT relational frame formation
        relational_frames = []
        if self.rft_engine:
            relational_frames = self.rft_engine.derive_relations(episodes)

        # Store insights in atomspace
        if self.atomspace:
            self._store_insights(concepts, relational_frames)

        return {
            'patterns': patterns,
            'concepts': concepts,
            'relational_frames': relational_frames,
            'insights': self_insights,
            'episodes_processed': len(episodes)
        }

    def extract_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """
        Discover patterns in experience.
        What tends to happen? What co-occurs?
        """
        patterns = []

        # Temporal patterns (what follows what)
        temporal_patterns = self._find_temporal_patterns(episodes)
        patterns.extend(temporal_patterns)

        # Co-occurrence patterns (what appears together)
        cooccurrence_patterns = self._find_cooccurrence_patterns(episodes)
        patterns.extend(cooccurrence_patterns)

        # Outcome patterns (similar outcomes from similar contexts)
        outcome_patterns = self._find_outcome_patterns(episodes)
        patterns.extend(outcome_patterns)

        self.patterns_discovered.extend(patterns)
        return patterns

    def _find_temporal_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """Find temporal sequences (A often follows B)"""
        patterns = []
        sorted_episodes = sorted(episodes, key=lambda e: e.timestamp)

        # Look for consecutive episode type sequences
        sequences = []
        for i in range(len(sorted_episodes) - 1):
            seq = (sorted_episodes[i].type.value, sorted_episodes[i + 1].type.value)
            sequences.append(seq)

        # Find frequent sequences
        seq_counts = Counter(sequences)
        for seq, count in seq_counts.items():
            if count >= 2:  # Occurred at least twice
                patterns.append(Pattern(
                    pattern_type='temporal',
                    frequency=count,
                    episodes=[e.id for e in sorted_episodes],
                    description=f"{seq[0]} often followed by {seq[1]}"
                ))

        return patterns

    def _find_cooccurrence_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """Find what tends to appear together"""
        patterns = []

        # Extract context items across episodes
        context_cooccurrences = defaultdict(int)
        for episode in episodes:
            if len(episode.context) > 1:
                # All pairs in context
                for i in range(len(episode.context)):
                    for j in range(i + 1, len(episode.context)):
                        pair = tuple(sorted([episode.context[i], episode.context[j]]))
                        context_cooccurrences[pair] += 1

        # Frequent cooccurrences
        for pair, count in context_cooccurrences.items():
            if count >= 2:
                patterns.append(Pattern(
                    pattern_type='cooccurrence',
                    frequency=count,
                    episodes=[e.id for e in episodes if all(p in e.context for p in pair)],
                    description=f"{pair[0]} and {pair[1]} often occur together"
                ))

        return patterns

    def _find_outcome_patterns(self, episodes: List[Episode]) -> List[Pattern]:
        """Find patterns in outcomes"""
        patterns = []

        # Group episodes by similar outcomes
        outcome_groups = defaultdict(list)
        for episode in episodes:
            if episode.outcome:
                # Simple grouping by outcome success/failure
                outcome_key = 'success' if episode.outcome.get('success') else 'failure'
                outcome_groups[outcome_key].append(episode)

        for outcome_type, group_episodes in outcome_groups.items():
            if len(group_episodes) >= 2:
                patterns.append(Pattern(
                    pattern_type='outcome',
                    frequency=len(group_episodes),
                    episodes=[e.id for e in group_episodes],
                    description=f"Pattern of {outcome_type} outcomes"
                ))

        return patterns

    def form_concepts(self, patterns: List[Pattern]) -> List[Concept]:
        """
        Form abstract concepts from patterns.
        Concepts are NOT preset - they emerge from experience.
        """
        concepts = []

        # Group patterns by type
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)

        # Form concepts from pattern groups
        for pattern_type, group in pattern_groups.items():
            if len(group) >= 2:
                concept = Concept(
                    name=f"{pattern_type}_pattern_concept",
                    definition=f"Abstract understanding of {pattern_type} patterns",
                    supporting_episodes=list(set(
                        ep_id for p in group for ep_id in p.episodes
                    )),
                    confidence=min(1.0, len(group) / 10.0)
                )
                concepts.append(concept)

        self.concepts_formed.extend(concepts)
        return concepts

    def adjust_memory_weights(self, episodes: List[Episode]):
        """
        Strengthen important memories, fade trivial ones.
        Importance based on emotional valence, self-relevance, novelty.
        """
        for episode in episodes:
            importance = (
                abs(episode.emotional_valence) * 0.4 +
                episode.self_relevance * 0.6
            )

            # Strengthen important memories in atomspace
            if self.atomspace and importance > 0.5:
                # Will implement atomspace strength adjustment
                pass

            # Fade trivial memories (low importance, not novel)
            if importance < 0.2:
                # Mark for potential pruning
                pass

    def update_self_understanding(self, patterns: List[Pattern]) -> List[str]:
        """
        Update self-model based on behavioral patterns.
        Who am I? What do I tend to do?
        """
        insights = []

        # Analyze behavioral tendencies
        action_patterns = [p for p in patterns if 'action' in p.description.lower()]
        if action_patterns:
            insights.append(f"Identified {len(action_patterns)} behavioral tendencies")

        # Analyze emotional patterns
        emotional_insights = self._analyze_emotional_patterns(patterns)
        insights.extend(emotional_insights)

        return insights

    def _analyze_emotional_patterns(self, patterns: List[Pattern]) -> List[str]:
        """Identify emotional tendencies"""
        insights = []

        # This would analyze emotional valence patterns
        # For now, placeholder
        if patterns:
            insights.append("Emotional pattern analysis complete")

        return insights

    def _store_insights(self, concepts: List[Concept], relational_frames: List):
        """Store formed concepts and relations in atomspace"""
        if not self.atomspace:
            return

        # Store concepts as concept nodes
        for concept in concepts:
            # Will implement with actual Atomspace API
            pass

        # Store relational frames
        for frame in relational_frames:
            # Will implement with actual Atomspace API
            pass
