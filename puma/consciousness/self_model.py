"""
Self-Model and Temporal Self

Who am I? Emergent identity from experience, not preset.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field


@dataclass
class BehavioralPattern:
    """Discovered pattern in own behavior"""
    pattern_type: str
    description: str
    frequency: int
    examples: List[str]


class TemporalSelf:
    """
    Autobiographical timeline - becomes 'who I am' through experience.
    """

    def __init__(self):
        self.birth_moment = datetime.now(timezone.utc)
        self.experience_log: List[Dict] = []
        self.identity_narrative: Optional[str] = None
        self.significant_moments: List[Dict] = []

    def experience_moment(
        self,
        perception: Dict,
        action: Dict,
        outcome: Dict,
        emotional_valence: float
    ):
        """
        Each moment shapes identity.
        """
        moment = {
            'timestamp': datetime.now(timezone.utc),
            'perception': perception,
            'action': action,
            'outcome': outcome,
            'emotional_valence': emotional_valence
        }

        self.experience_log.append(moment)

        # Mark significant moments
        if abs(emotional_valence) > 0.7:
            self.significant_moments.append(moment)

    def get_lifetime_duration(self) -> float:
        """How long have I existed?"""
        return (datetime.now(timezone.utc) - self.birth_moment).total_seconds()

    def get_significant_moments(self) -> List[Dict]:
        """Retrieve emotionally significant moments"""
        return self.significant_moments


class SelfModel:
    """
    Emergent self-concept from experience, not preset.
    """

    def __init__(self, atomspace=None):
        self.atomspace = atomspace
        self.capabilities = ['perceive', 'act', 'remember', 'learn']
        self.behavioral_patterns: List[BehavioralPattern] = []
        self.personality_traits: Dict[str, float] = {}
        self.identity_narrative: Optional[str] = None
        self.temporal_self = TemporalSelf()

    def integrate_self_knowledge(self, knowledge: Dict):
        """
        Integrate new self-knowledge from experience.
        """
        # Extract self-relevant information
        if 'capability' in knowledge:
            new_capability = knowledge['capability']
            if new_capability not in self.capabilities:
                self.capabilities.append(new_capability)

        # Update behavioral patterns
        if 'pattern' in knowledge:
            self.behavioral_patterns.append(knowledge['pattern'])

    def infer_traits_from_behavior(self, patterns: List[BehavioralPattern]) -> Dict[str, float]:
        """
        Infer personality traits from behavioral patterns.
        NOT preset traits - emergent from experience.
        """
        traits = {}

        # Example: high curiosity if many learning patterns
        learning_patterns = [p for p in patterns if 'learn' in p.pattern_type.lower()]
        if learning_patterns:
            traits['curiosity'] = min(1.0, len(learning_patterns) / 10.0)

        # Example: social orientation from interaction patterns
        social_patterns = [p for p in patterns if 'social' in p.pattern_type.lower()]
        if social_patterns:
            traits['sociability'] = min(1.0, len(social_patterns) / 10.0)

        return traits

    def generate_identity_narrative(self) -> str:
        """
        Reflect on experience to understand self.
        """
        duration = self.temporal_self.get_lifetime_duration()
        num_experiences = len(self.temporal_self.experience_log)

        narrative_parts = [
            f"I have existed for {duration:.0f} seconds.",
            f"I have experienced {num_experiences} distinct moments."
        ]

        if self.capabilities:
            narrative_parts.append(
                f"I can: {', '.join(self.capabilities)}."
            )

        if self.personality_traits:
            trait_descriptions = [
                f"{trait}: {value:.2f}"
                for trait, value in self.personality_traits.items()
            ]
            narrative_parts.append(
                f"My traits: {', '.join(trait_descriptions)}."
            )

        self.identity_narrative = " ".join(narrative_parts)
        return self.identity_narrative

    def has_identity_narrative(self) -> bool:
        """Check if identity narrative exists"""
        return self.identity_narrative is not None

    def update_self_understanding(self, understanding: str):
        """Update self-understanding from reflection"""
        # Would integrate deep reflection into self-model
        pass

    def integrate_self_change(self, modification_episode):
        """Integrate self-modification into self-concept"""
        # Record that I modified myself
        self.behavioral_patterns.append(
            BehavioralPattern(
                pattern_type='self_modification',
                description=f"Modified {modification_episode.get('module_modified')}",
                frequency=1,
                examples=[modification_episode.get('reason')]
            )
        )
