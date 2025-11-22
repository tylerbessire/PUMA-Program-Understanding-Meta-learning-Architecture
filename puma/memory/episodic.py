"""
Episodic Memory System

Every experience becomes a memory, shaping identity over time.
"""

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid


class MemoryType(Enum):
    """Types of episodic memories"""
    CONVERSATION = "conversation"
    WEB_EXPLORATION = "web_exploration"
    SELF_MODIFICATION = "self_modification"
    LEARNING = "learning"
    GOAL_FORMATION = "goal_formation"
    STATE_TRANSITION = "state_transition"
    GENERAL = "general"


@dataclass
class Episode:
    """
    Single episodic memory - a moment in time.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    type: MemoryType = MemoryType.GENERAL
    perception: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    outcome: Optional[Dict[str, Any]] = None
    context: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    self_relevance: float = 0.0
    consolidated: bool = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value,
            'perception': self.perception,
            'action': self.action,
            'outcome': self.outcome,
            'context': self.context,
            'emotional_valence': self.emotional_valence,
            'self_relevance': self.self_relevance,
            'consolidated': self.consolidated
        }

    def generate_summary(self) -> str:
        """Generate human-readable summary"""
        if self.type == MemoryType.CONVERSATION:
            return f"Conversation: {self.perception.get('utterance', 'unknown')[:50]}"
        elif self.type == MemoryType.WEB_EXPLORATION:
            return f"Explored: {self.perception.get('url', 'unknown')}"
        elif self.type == MemoryType.SELF_MODIFICATION:
            return f"Modified: {self.action.get('module', 'unknown')}"
        else:
            return f"{self.type.value} at {self.timestamp.isoformat()}"


class EpisodicMemorySystem:
    """
    Manages formation and storage of episodic memories.
    Every experience becomes a memory that shapes identity.
    """

    def __init__(self, atomspace=None):
        self.atomspace = atomspace
        self.episodes: List[Episode] = []
        self.current_context: List[str] = []
        self.consolidation_queue: List[Episode] = []

    def form_episode(
        self,
        perception: Optional[Dict] = None,
        action: Optional[Dict] = None,
        outcome: Optional[Dict] = None,
        memory_type: MemoryType = MemoryType.GENERAL
    ) -> Episode:
        """
        Create episodic memory from experience.

        Args:
            perception: What was perceived
            action: What action was taken
            outcome: What happened as a result
            memory_type: Type of memory

        Returns:
            Episode object
        """
        episode = Episode(
            type=memory_type,
            perception=perception,
            action=action,
            outcome=outcome,
            context=self.current_context.copy(),
            emotional_valence=self._assess_emotion(outcome),
            self_relevance=self._assess_self_relevance(perception, outcome)
        )

        self.episodes.append(episode)
        self.consolidation_queue.append(episode)

        # Store in atomspace if available
        if self.atomspace:
            self._store_in_atomspace(episode)

        return episode

    def _assess_emotion(self, outcome: Optional[Dict]) -> float:
        """
        Assess emotional valence of outcome.
        Emergent from goal satisfaction, surprise, novelty.
        NOT preset emotional reactions.
        """
        if not outcome:
            return 0.0

        valence = 0.0

        # Goal satisfaction contributes positive valence
        if outcome.get('goal_satisfied'):
            valence += 0.5

        # Surprise can be positive or negative
        if outcome.get('surprising'):
            valence += 0.2 if outcome.get('positive_surprise') else -0.2

        # Novelty contributes mild positive valence
        if outcome.get('novel'):
            valence += 0.3

        return max(-1.0, min(1.0, valence))

    def _assess_self_relevance(
        self,
        perception: Optional[Dict],
        outcome: Optional[Dict]
    ) -> float:
        """
        Assess how relevant this experience is to self-model.
        """
        if not perception and not outcome:
            return 0.0

        relevance = 0.0

        # Direct self-reference
        if perception and 'self' in str(perception).lower():
            relevance += 0.5

        # Goal-relevant
        if outcome and outcome.get('goal_relevant'):
            relevance += 0.3

        # Learning-relevant
        if outcome and outcome.get('learned_something'):
            relevance += 0.2

        return min(1.0, relevance)

    def _store_in_atomspace(self, episode: Episode):
        """Store episode in atomspace as node"""
        # Will integrate with actual Atomspace
        pass

    def get_recent_episodes(self, limit: int = 10) -> List[Episode]:
        """Get most recent episodes"""
        return sorted(self.episodes, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_unconsolidated_episodes(self) -> List[Episode]:
        """Get episodes that haven't been consolidated yet"""
        return [e for e in self.episodes if not e.consolidated]

    def mark_consolidated(self, episode_ids: List[str]):
        """Mark episodes as consolidated"""
        for episode in self.episodes:
            if episode.id in episode_ids:
                episode.consolidated = True

    def update_context(self, context_items: List[str]):
        """Update current context for future episodes"""
        self.current_context = context_items

    def get_episodes_by_type(self, memory_type: MemoryType) -> List[Episode]:
        """Retrieve episodes by type"""
        return [e for e in self.episodes if e.type == memory_type]

    def count_total_episodes(self) -> int:
        """Count all episodes"""
        return len(self.episodes)

    def get_timeline(self) -> List[Episode]:
        """Get full autobiographical timeline"""
        return sorted(self.episodes, key=lambda e: e.timestamp)
