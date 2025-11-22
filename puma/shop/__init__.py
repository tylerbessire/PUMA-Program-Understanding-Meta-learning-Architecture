"""
The Shop - Self-Modification System

AGI can inspect and modify its own cognitive code.
"""

from .introspection import CodeIntrospection, CognitiveModule
from .modification import ModificationSystem, ModificationPlan
from .sandbox import ModificationSandbox, TestReport

__all__ = [
    'CodeIntrospection',
    'CognitiveModule',
    'ModificationSystem',
    'ModificationPlan',
    'ModificationSandbox',
    'TestReport'
]
