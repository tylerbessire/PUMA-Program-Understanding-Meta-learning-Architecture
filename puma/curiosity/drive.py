"""
Curiosity Drive

Autonomous learning motivation - NOT task-oriented.
AGI asks itself questions from experience and seeks to answer them.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum
import uuid


class QuestionType(Enum):
    """Types of curiosity questions"""
    KNOWLEDGE_GAP = "knowledge_gap"
    CONTRADICTION = "contradiction"
    INCOMPLETE_MODEL = "incomplete_model"
    NOVEL_CONCEPT = "novel_concept"
    UNEXPLORED_RELATION = "unexplored_relation"


@dataclass
class Question:
    """A question generated from curiosity"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    question_type: QuestionType = QuestionType.KNOWLEDGE_GAP
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5
    answered: bool = False
    answer: Optional[str] = None
    answered_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'question': self.question,
            'question_type': self.question_type.value,
            'created_at': self.created_at.isoformat(),
            'importance': self.importance,
            'answered': self.answered,
            'answer': self.answer,
            'answered_at': self.answered_at.isoformat() if self.answered_at else None
        }


class CuriosityDrive:
    """
    Intrinsic motivation system.
    AGI generates its own questions and seeks answers.
    """

    def __init__(self, atomspace=None):
        self.atomspace = atomspace
        self.open_questions: List[Question] = []
        self.answered_questions: List[Question] = []
        self.knowledge_gaps: Set[str] = set()
        self.boredom_threshold = 0.5
        self.current_boredom = 0.0
        self.last_novel_experience: Optional[datetime] = None

    def generate_questions(self) -> List[Question]:
        """
        AGI asks itself questions from experience.
        Questions emerge from knowledge gaps, not preset curiosity.
        """
        questions = []

        # Identify knowledge gaps
        gaps = self.identify_knowledge_gaps()

        # Form questions from gaps
        for gap in gaps:
            question = self.formulate_question(gap)
            if question:
                questions.append(question)
                self.open_questions.append(question)

        return questions

    def identify_knowledge_gaps(self) -> List[Dict[str, Any]]:
        """
        Find what I don't understand yet.
        """
        gaps = []

        # Knowledge gap: Concepts with few connections (underexplored)
        if self.atomspace:
            isolated_concepts = self._find_isolated_concepts()
            for concept in isolated_concepts:
                gaps.append({
                    'type': 'isolated_concept',
                    'concept': concept,
                    'importance': 0.6
                })

        # Knowledge gap: Contradictions in knowledge
        contradictions = self._find_contradictions()
        for contradiction in contradictions:
            gaps.append({
                'type': 'contradiction',
                'details': contradiction,
                'importance': 0.8
            })

        # Knowledge gap: Incomplete causal models
        incomplete_models = self._find_incomplete_causal_chains()
        for model in incomplete_models:
            gaps.append({
                'type': 'incomplete_model',
                'model': model,
                'importance': 0.7
            })

        return gaps

    def _find_isolated_concepts(self) -> List[str]:
        """Find concepts with few connections"""
        # Placeholder - would query atomspace for concepts with < N links
        if not self.atomspace:
            return []

        # In real implementation, query atomspace
        # For now, return known knowledge gaps
        return list(self.knowledge_gaps)

    def _find_contradictions(self) -> List[Dict]:
        """Find contradictory knowledge"""
        # Placeholder - would analyze atomspace for contradictions
        return []

    def _find_incomplete_causal_chains(self) -> List[Dict]:
        """Find incomplete cause-effect relationships"""
        # Placeholder - would analyze causal frames
        return []

    def formulate_question(self, gap: Dict[str, Any]) -> Optional[Question]:
        """
        Turn knowledge gap into actionable question.
        """
        gap_type = gap['type']

        if gap_type == 'isolated_concept':
            question_text = f"What is {gap['concept']} and how does it relate to what I know?"
            q_type = QuestionType.KNOWLEDGE_GAP

        elif gap_type == 'contradiction':
            question_text = f"Why do I have contradictory information about {gap['details']}?"
            q_type = QuestionType.CONTRADICTION

        elif gap_type == 'incomplete_model':
            question_text = f"What causes {gap['model']} and what are its effects?"
            q_type = QuestionType.INCOMPLETE_MODEL

        else:
            return None

        return Question(
            question=question_text,
            question_type=q_type,
            importance=gap.get('importance', 0.5)
        )

    def update_boredom(self, experience_novel: bool = False) -> bool:
        """
        Update boredom level based on experience novelty.
        Returns True if exploration should be triggered.
        """
        if experience_novel:
            # Novel experience reduces boredom
            self.current_boredom = max(0, self.current_boredom - 0.3)
            self.last_novel_experience = datetime.now(timezone.utc)
        else:
            # Repetitive experience increases boredom
            self.current_boredom = min(1.0, self.current_boredom + 0.1)

        # Trigger exploration when bored
        return self.current_boredom > self.boredom_threshold

    def mark_questions_answered(self, question_ids: List[str], answers: Optional[Dict[str, str]] = None):
        """Mark questions as answered"""
        for question in self.open_questions:
            if question.id in question_ids:
                question.answered = True
                question.answered_at = datetime.now(timezone.utc)

                if answers and question.id in answers:
                    question.answer = answers[question.id]

                self.answered_questions.append(question)

        # Remove from open questions
        self.open_questions = [q for q in self.open_questions if not q.answered]

        # Reduce boredom when questions are answered
        self.current_boredom = max(0, self.current_boredom - 0.2 * len(question_ids))

    def add_questions(self, new_questions: List[str]):
        """Add new curiosity questions"""
        for q_text in new_questions:
            question = Question(
                question=q_text,
                question_type=QuestionType.NOVEL_CONCEPT,
                importance=0.5
            )
            self.open_questions.append(question)

    def get_most_important_questions(self, limit: int = 5) -> List[Question]:
        """Get highest priority questions"""
        return sorted(
            self.open_questions,
            key=lambda q: q.importance,
            reverse=True
        )[:limit]

    def add_knowledge_gap(self, concept: str):
        """Add a recognized knowledge gap"""
        self.knowledge_gaps.add(concept)

    def get_curiosity_level(self) -> float:
        """Get current curiosity level (0-1)"""
        # High when many open questions
        return min(1.0, len(self.open_questions) / 20.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get curiosity statistics"""
        return {
            'open_questions': len(self.open_questions),
            'answered_questions': len(self.answered_questions),
            'knowledge_gaps': len(self.knowledge_gaps),
            'boredom_level': self.current_boredom,
            'curiosity_level': self.get_curiosity_level(),
            'last_novel_experience': self.last_novel_experience.isoformat() if self.last_novel_experience else None
        }
