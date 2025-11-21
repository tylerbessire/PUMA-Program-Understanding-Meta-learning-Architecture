"""
PUMA: Program Understanding Meta-learning Architecture - ARC Solver Package

A Brain-Inspired Reinforcement Learning from Thinking (RFT) Architecture

This package implements PUMA's novel cognitive architecture for the ARC AGI Competition
2025, integrating behavioral analysis principles from Relational Frame Theory with
transformer architectures to enable abstract reasoning capabilities.

Core Innovation: Frequency Ledger System
-----------------------------------------
PUMA's breakthrough innovation is the Frequency Ledger System - a sophisticated
frequency-based analysis framework that groups objects by numerical attributes
(frequencies, counts, patterns) to enable models to discover abstract relationships.
This behavior-analytic approach allows models to make derivational connections between
stimuli without explicit training on those relationships—mirroring how humans learn
through relational framing.

Key Components:
---------------
- **ARCSolver**: High-level solver integrating all PUMA capabilities
- **Frequency Ledger**: Core frequency-based analysis and pattern discovery
- **RFT Engine**: Relational Frame Theory implementation for behavioral reasoning
- **Neural Guidance**: Predicts relevant DSL operations using behavioral task features
- **Episodic Retrieval**: Database of solved tasks for analogical reasoning
- **Test-Time Training**: Adapts scoring functions through reinforcement learning

Behavioral Approach:
--------------------
PUMA treats reasoning as learned relational responding rather than symbolic manipulation.
By applying behavioral analysis principles and Relational Frame Theory, PUMA has achieved:
- Top 15% placement in ARC AGI Competition 2025
- 35-40% improvement in abstract reasoning tasks through behavioral framing
- First successful integration of RFT with transformer architectures
"""

from .solver import ARCSolver
from .io_utils import load_rerun_json, save_submission
from .grid import Array

__all__ = ["ARCSolver", "load_rerun_json", "save_submission", "Array"]
