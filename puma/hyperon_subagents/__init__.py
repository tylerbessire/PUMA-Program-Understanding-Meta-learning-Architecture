"""
Hyperon Subagents Module

Integration of OpenCog Hyperon's MeTTa reasoning engine with PUMA's
cognitive architecture for symbolic reasoning and program execution.
Includes parallel subagent coordination and execution management.

Key Components:
--------------
- HyperonSubAgent: Individual MeTTa interpreter instance with specialized capabilities
- SubAgentManager: Coordinates multiple subagents for parallel distributed reasoning
- MeTTaExecutionEngine: Core MeTTa execution and integration
- SubAgentCoordinator: High-level coordination and communication patterns
- RFTHyperonBridge: Bridge between RFT (Relational Frame Theory) and Hyperon

States:
-------
- SubAgentState: IDLE, RUNNING, WAITING, COMPLETED, FAILED, SUSPENDED

Capabilities:
------------
- AgentCapability: REASONING, PATTERN_MATCHING, MEMORY_RETRIEVAL, GOAL_PLANNING,
                   RELATIONAL_FRAMING, ABSTRACTION, ANALOGY_MAKING, CONCEPT_SYNTHESIS
"""

from .metta_engine import (
    MeTTaExecutionEngine,
    ExecutionMode,
    ExecutionResult,
    MeTTaEngineError,
)

from .coordinator import (
    SubAgentCoordinator,
    CoordinationStrategy,
    CommunicationPattern,
    TaskResult,
    SubAgentTask,
    SubAgentStatus,
)

from .rft_bridge import (
    RFTHyperonBridge,
    MeTTaRelation,
)

from .manager import (
    HyperonSubAgent,
    SubAgentManager,
    SubAgentTask as ManagerSubAgentTask,
    SubAgentResult,
    SubAgentState,
    AgentCapability,
    HYPERON_AVAILABLE,
)

__all__ = [
    # MeTTa Engine
    "MeTTaExecutionEngine",
    "ExecutionMode",
    "ExecutionResult",
    "MeTTaEngineError",
    # Coordinator
    "SubAgentCoordinator",
    "CoordinationStrategy",
    "CommunicationPattern",
    "TaskResult",
    "SubAgentTask",
    "SubAgentStatus",
    # RFT Bridge
    "RFTHyperonBridge",
    "MeTTaRelation",
    # Manager
    "HyperonSubAgent",
    "SubAgentManager",
    "ManagerSubAgentTask",
    "SubAgentResult",
    "SubAgentState",
    "AgentCapability",
    "HYPERON_AVAILABLE",
]
