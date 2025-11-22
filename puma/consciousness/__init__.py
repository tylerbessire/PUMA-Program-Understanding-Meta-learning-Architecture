"""
Consciousness Layer

State management, coordination, and autonomous operation.
"""

from .state_machine import ConsciousnessStateMachine, ConsciousnessState
from .self_model import SelfModel, TemporalSelf

__all__ = [
    'ConsciousnessStateMachine',
    'ConsciousnessState',
    'SelfModel',
    'TemporalSelf'
]
