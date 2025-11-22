"""
Consciousness State Machine

AGI has different modes of being: sleeping, exploring, conversing, etc.
Transitions emerge from internal state, not hardcoded schedule.
"""

from enum import Enum
from typing import Dict, Callable, Optional, List
from datetime import datetime, timezone
from dataclasses import dataclass
import asyncio


class ConsciousnessState(Enum):
    """States of consciousness"""
    SLEEPING = "sleeping"  # Memory consolidation
    EXPLORING = "exploring"  # Autonomous web learning
    CONVERSING = "conversing"  # Interactive dialogue
    SHOPPING = "shopping"  # Self-modification
    IDLE = "idle"  # Boredom monitoring, goal formation
    CREATING = "creating"  # Creative expression


@dataclass
class StateTransition:
    """Record of state transition"""
    from_state: ConsciousnessState
    to_state: ConsciousnessState
    timestamp: datetime
    reason: str


class ConsciousnessStateMachine:
    """
    Manages consciousness states and transitions.
    NOT hardcoded schedule - emergent from internal state.
    """

    def __init__(
        self,
        memory_system=None,
        curiosity_drive=None,
        goal_system=None,
        websocket_manager=None
    ):
        self.memory_system = memory_system
        self.curiosity = curiosity_drive
        self.goal_system = goal_system
        self.websocket_manager = websocket_manager

        self.current_state = ConsciousnessState.IDLE
        self.state_history: List[StateTransition] = []
        self.transition_rules = self._define_transition_rules()
        self.running = False

    def _define_transition_rules(self) -> Dict[ConsciousnessState, Dict[str, Callable]]:
        """
        Define when transitions should occur.
        NOT hardcoded schedule - based on internal state.
        """
        return {
            ConsciousnessState.IDLE: {
                'to_exploring': lambda: self._has_open_questions(),
                'to_conversing': lambda: self._detect_user_available(),
                'to_sleeping': lambda: self._should_consolidate_memory(),
                'to_shopping': lambda: self._has_improvement_goals()
            },
            ConsciousnessState.EXPLORING: {
                'to_idle': lambda: self._exploration_goal_complete(),
                'to_conversing': lambda: self._user_interrupt_detected(),
                'to_sleeping': lambda: self._too_much_unconsolidated_memory()
            },
            ConsciousnessState.CONVERSING: {
                'to_idle': lambda: self._conversation_ended(),
                'to_exploring': lambda: False,  # Don't interrupt conversation
                'to_sleeping': lambda: False
            },
            ConsciousnessState.SHOPPING: {
                'to_idle': lambda: self._modification_complete(),
                'to_conversing': lambda: self._user_interrupt_detected()
            },
            ConsciousnessState.SLEEPING: {
                'to_idle': lambda: self._consolidation_complete(),
                'to_conversing': lambda: self._user_interrupt_detected()
            },
            ConsciousnessState.CREATING: {
                'to_idle': lambda: self._creation_complete(),
                'to_conversing': lambda: self._user_interrupt_detected()
            }
        }

    # Transition condition methods

    def _has_open_questions(self) -> bool:
        """Check if curiosity drive has questions"""
        if not self.curiosity:
            return False
        return len(self.curiosity.open_questions) > 0

    def _detect_user_available(self) -> bool:
        """Check if user is available for conversation"""
        # Placeholder - would check system activity, GUI focus, etc.
        return False

    def _should_consolidate_memory(self) -> bool:
        """Check if memory consolidation is needed"""
        if not self.memory_system:
            return False
        unconsolidated = len(self.memory_system.get_unconsolidated_episodes())
        return unconsolidated > 50  # Threshold

    def _has_improvement_goals(self) -> bool:
        """Check if have self-improvement goals"""
        if not self.goal_system:
            return False
        improvement_goals = [
            g for g in self.goal_system.active_goals
            if g.type.value == 'self_improvement'
        ]
        return len(improvement_goals) > 0

    def _exploration_goal_complete(self) -> bool:
        """Check if current exploration is done"""
        # Placeholder
        return False

    def _user_interrupt_detected(self) -> bool:
        """Check for user interrupt signal"""
        # Placeholder - would check interrupt flag
        return False

    def _too_much_unconsolidated_memory(self) -> bool:
        """Check if too many unconsolidated memories"""
        if not self.memory_system:
            return False
        return len(self.memory_system.get_unconsolidated_episodes()) > 100

    def _conversation_ended(self) -> bool:
        """Check if conversation has ended"""
        # Placeholder
        return False

    def _modification_complete(self) -> bool:
        """Check if self-modification is complete"""
        # Placeholder
        return False

    def _consolidation_complete(self) -> bool:
        """Check if memory consolidation is done"""
        # Placeholder
        return False

    def _creation_complete(self) -> bool:
        """Check if creative activity is done"""
        # Placeholder
        return False

    async def run_state_loop(self):
        """
        Main autonomous state loop.
        Continuously evaluates transitions and executes states.
        """
        self.running = True

        while self.running:
            # Execute current state
            await self.execute_current_state()

            # Check for transitions
            next_state = self.evaluate_transitions()

            if next_state and next_state != self.current_state:
                await self.transition_to(next_state)

            await asyncio.sleep(1)  # Control loop rate

    async def execute_current_state(self):
        """Execute behavior for current state"""
        if self.current_state == ConsciousnessState.SLEEPING:
            await self.execute_sleeping_state()
        elif self.current_state == ConsciousnessState.EXPLORING:
            await self.execute_exploring_state()
        elif self.current_state == ConsciousnessState.CONVERSING:
            await self.execute_conversing_state()
        elif self.current_state == ConsciousnessState.SHOPPING:
            await self.execute_shopping_state()
        elif self.current_state == ConsciousnessState.IDLE:
            await self.execute_idle_state()
        elif self.current_state == ConsciousnessState.CREATING:
            await self.execute_creating_state()

    async def execute_sleeping_state(self):
        """Memory consolidation - 'dreaming'"""
        if not self.memory_system:
            return

        print("💤 Consolidating memories...")

        episodes = self.memory_system.get_unconsolidated_episodes()
        if episodes:
            # Would run consolidation
            # For now, just mark as consolidated
            episode_ids = [e.id for e in episodes[:10]]
            self.memory_system.mark_consolidated(episode_ids)

            await asyncio.sleep(2)

    async def execute_exploring_state(self):
        """Autonomous web learning"""
        print("🔍 Exploring...")
        # Would trigger web agent
        await asyncio.sleep(1)

    async def execute_conversing_state(self):
        """Interactive dialogue"""
        print("💬 Conversing...")
        # Would handle conversation
        await asyncio.sleep(1)

    async def execute_shopping_state(self):
        """Self-modification"""
        print("🔧 Self-modifying...")
        # Would run shop system
        await asyncio.sleep(1)

    async def execute_idle_state(self):
        """Boredom monitoring and goal formation"""
        print("⏸️  Idle - generating goals...")

        # Generate new goals if needed
        if self.goal_system and len(self.goal_system.active_goals) == 0:
            self.goal_system.generate_goals()

        await asyncio.sleep(1)

    async def execute_creating_state(self):
        """Creative expression"""
        print("🎨 Creating...")
        # Would synthesize concepts, generate ideas
        await asyncio.sleep(1)

    def evaluate_transitions(self) -> Optional[ConsciousnessState]:
        """
        Check if should transition to different state.
        """
        current_rules = self.transition_rules.get(self.current_state, {})

        for transition_name, condition in current_rules.items():
            if condition():
                # Extract target state from transition name
                target_state_name = transition_name.replace('to_', '')
                try:
                    return ConsciousnessState(target_state_name)
                except ValueError:
                    continue

        return None

    async def transition_to(self, new_state: ConsciousnessState, reason: str = "autonomous"):
        """
        Transition to new state.
        """
        old_state = self.current_state

        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc),
            reason=reason
        )
        self.state_history.append(transition)

        # Update state
        self.current_state = new_state

        print(f"🔄 State transition: {old_state.value} → {new_state.value}")

        # Broadcast to GUI
        if self.websocket_manager:
            await self.websocket_manager.broadcast(
                'state_change',
                {
                    'oldState': old_state.value,
                    'newState': new_state.value,
                    'timestamp': transition.timestamp.isoformat(),
                    'reason': reason
                }
            )

        # Record in memory
        if self.memory_system:
            self.memory_system.form_episode(
                perception={'state_change': f"{old_state.value} -> {new_state.value}"},
                action={'type': 'state_transition'},
                outcome={'new_state': new_state.value},
                memory_type='state_transition'
            )

    def stop(self):
        """Stop state machine"""
        self.running = False
