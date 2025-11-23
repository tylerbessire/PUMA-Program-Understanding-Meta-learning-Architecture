"""
Hyperon-PUMA Integration Module

This module provides high-level integration between OpenCog Hyperon's MeTTa
reasoning engine and PUMA's cognitive architecture. It coordinates all Hyperon
components including subagent systems, coordinators, bridges, and consciousness
integration.

Architecture Overview:
---------------------
1. HyperonPUMAIntegration - Main integration class
   - Initializes all Hyperon components
   - Provides convenience methods for common workflows
   - Integrates with consciousness states
   - Manages subagent lifecycle

2. Integration Points:
   - SubAgentManager: Parallel distributed reasoning
   - SubAgentCoordinator: Task coordination and communication
   - RFTHyperonBridge: RFT <-> MeTTa conversion
   - MeTTaExecutionEngine: Core MeTTa execution
   - ConsciousnessState: State-aware coordination

3. Workflows:
   - ARC task solving with distributed reasoning
   - RFT reasoning across subagent pool
   - Frequency analysis with MeTTa inference
   - Consciousness-aware task routing

Usage:
------
    # Initialize integration
    integration = HyperonPUMAIntegration(
        atomspace=atomspace,
        rft_engine=rft_engine,
        consciousness_state_machine=state_machine
    )

    # Initialize components
    await integration.initialize()

    # Solve ARC task
    result = await integration.solve_arc_task(task_data)

    # Perform RFT reasoning
    frames = await integration.reason_with_rft(
        source="A",
        target="B",
        relation_type=RelationType.COORDINATION
    )

    # Frequency analysis
    signature = await integration.analyze_frequencies(pattern_data)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# PUMA core imports
try:
    from puma.rft.reasoning import RFTEngine, RelationalFrame, RelationType
except ImportError:
    RFTEngine = RelationalFrame = RelationType = None

try:
    from puma.consciousness.state_machine import (
        ConsciousnessStateMachine,
        ConsciousnessState,
    )
except ImportError:
    ConsciousnessStateMachine = ConsciousnessState = None

try:
    from puma.memory import EpisodicMemorySystem
except ImportError:
    EpisodicMemorySystem = None

# Hyperon subagent imports
from puma.hyperon_subagents import (
    SubAgentManager,
    SubAgentCoordinator,
    RFTHyperonBridge,
    MeTTaExecutionEngine,
    SubAgentTask,
    SubAgentResult,
    SubAgentState,
    AgentCapability,
    CoordinationStrategy,
    CommunicationPattern,
    HYPERON_AVAILABLE,
)

# ARC solver imports
try:
    from arc_solver.frequency_ledger import FrequencyLedger, FrequencySignature
    from arc_solver.rft import RelationalFrameAnalyzer
except ImportError:
    FrequencyLedger = FrequencySignature = None
    RelationalFrameAnalyzer = None

# Atomspace imports
try:
    from atomspace_db.core import Atomspace
except ImportError:
    try:
        from core import Atomspace
    except ImportError:
        Atomspace = None


# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Integration Configuration
# ============================================================================


@dataclass
class HyperonConfig:
    """Configuration for Hyperon integration"""

    # Subagent pool configuration
    max_agents: int = 10
    create_specialized_pool: bool = True
    default_timeout: float = 30.0

    # Coordination configuration
    default_coordination_strategy: CoordinationStrategy = (
        CoordinationStrategy.PARALLEL
    )
    default_communication_pattern: CommunicationPattern = (
        CommunicationPattern.SHARED_MEMORY
    )

    # Performance configuration
    enable_metrics: bool = True
    enable_caching: bool = True
    cache_size: int = 1000

    # Integration configuration
    integrate_with_consciousness: bool = True
    integrate_with_memory: bool = True
    enable_frequency_ledger: bool = True


# ============================================================================
# Main Integration Class
# ============================================================================


class HyperonPUMAIntegration:
    """
    Main integration class coordinating all Hyperon components with PUMA.

    This class provides a high-level interface for:
    - Initializing all Hyperon subagent systems
    - Coordinating distributed reasoning tasks
    - Bridging RFT with MeTTa inference
    - Integrating with consciousness states
    - Managing subagent lifecycle and resources

    Attributes
    ----------
    atomspace : Atomspace
        PUMA's main atomspace for knowledge storage
    rft_engine : RFTEngine
        PUMA's relational frame theory reasoning engine
    consciousness_state_machine : ConsciousnessStateMachine
        PUMA's consciousness state manager
    memory_system : EpisodicMemorySystem
        PUMA's episodic memory system
    config : HyperonConfig
        Configuration for Hyperon integration
    """

    def __init__(
        self,
        atomspace: Optional[Atomspace] = None,
        rft_engine: Optional[RFTEngine] = None,
        consciousness_state_machine: Optional[ConsciousnessStateMachine] = None,
        memory_system: Optional[EpisodicMemorySystem] = None,
        config: Optional[HyperonConfig] = None,
    ):
        """
        Initialize Hyperon-PUMA integration.

        Parameters
        ----------
        atomspace : Atomspace, optional
            PUMA's main atomspace
        rft_engine : RFTEngine, optional
            PUMA's RFT reasoning engine
        consciousness_state_machine : ConsciousnessStateMachine, optional
            PUMA's consciousness state manager
        memory_system : EpisodicMemorySystem, optional
            PUMA's episodic memory system
        config : HyperonConfig, optional
            Configuration for Hyperon integration
        """
        self.atomspace = atomspace
        self.rft_engine = rft_engine
        self.consciousness_state_machine = consciousness_state_machine
        self.memory_system = memory_system
        self.config = config or HyperonConfig()

        # Hyperon components (initialized in initialize())
        self.subagent_manager: Optional[SubAgentManager] = None
        self.coordinator: Optional[SubAgentCoordinator] = None
        self.rft_bridge: Optional[RFTHyperonBridge] = None
        self.metta_engine: Optional[MeTTaExecutionEngine] = None
        self.frequency_ledger: Optional[FrequencyLedger] = None

        # State tracking
        self.initialized = False
        self.active_tasks: Dict[str, SubAgentTask] = {}
        self.task_results: Dict[str, SubAgentResult] = {}

        logger.info("HyperonPUMAIntegration created")

    async def initialize(self) -> None:
        """
        Initialize all Hyperon components.

        This method:
        1. Creates subagent manager and pool
        2. Initializes coordinator with communication patterns
        3. Sets up RFT-Hyperon bridge
        4. Configures MeTTa execution engine
        5. Integrates with consciousness states
        """
        if self.initialized:
            logger.warning("Already initialized")
            return

        logger.info("Initializing Hyperon-PUMA integration...")

        # 1. Initialize SubAgent Manager
        logger.info(f"Creating subagent pool (max {self.config.max_agents} agents)")
        self.subagent_manager = SubAgentManager(max_agents=self.config.max_agents)

        if self.config.create_specialized_pool:
            self.subagent_manager.create_specialized_agents()
            logger.info(
                f"Created specialized agent pool: {len(self.subagent_manager.agents)} agents"
            )

        # 2. Initialize SubAgent Coordinator
        logger.info("Initializing subagent coordinator")
        self.coordinator = SubAgentCoordinator(
            atomspace=self.atomspace,
            consciousness_state=self._get_current_consciousness_state(),
        )

        # Set communication pattern
        if self.atomspace:
            self.coordinator.set_communication_pattern(
                CommunicationPattern.SHARED_MEMORY
            )
        else:
            self.coordinator.set_communication_pattern(CommunicationPattern.BROADCAST)

        # 3. Initialize RFT-Hyperon Bridge
        if self.rft_engine or self.atomspace:
            logger.info("Initializing RFT-Hyperon bridge")
            self.rft_bridge = RFTHyperonBridge(atomspace=self.atomspace)
            logger.info("RFT-Hyperon bridge initialized")

        # 4. Initialize MeTTa Execution Engine
        logger.info("Initializing MeTTa execution engine")
        self.metta_engine = MeTTaExecutionEngine()
        logger.info("MeTTa execution engine initialized")

        # 5. Initialize Frequency Ledger (if enabled)
        if self.config.enable_frequency_ledger and FrequencyLedger:
            logger.info("Initializing frequency ledger")
            self.frequency_ledger = FrequencyLedger()
            logger.info("Frequency ledger initialized")

        self.initialized = True
        logger.info("Hyperon-PUMA integration initialized successfully")

        # Log integration status
        self._log_integration_status()

    def _get_current_consciousness_state(self) -> Optional[ConsciousnessState]:
        """Get current consciousness state if available"""
        if self.consciousness_state_machine:
            return self.consciousness_state_machine.current_state
        return None

    def _log_integration_status(self) -> None:
        """Log current integration status"""
        status = self.get_status()
        logger.info("Integration Status:")
        logger.info(f"  Hyperon Available: {status['hyperon_available']}")
        logger.info(f"  Subagents: {status['num_subagents']}")
        logger.info(f"  RFT Bridge: {'enabled' if status['rft_bridge_enabled'] else 'disabled'}")
        logger.info(
            f"  Consciousness Integration: {'enabled' if status['consciousness_integrated'] else 'disabled'}"
        )
        logger.info(
            f"  Memory Integration: {'enabled' if status['memory_integrated'] else 'disabled'}"
        )

    # ========================================================================
    # High-Level Workflow Methods
    # ========================================================================

    async def solve_arc_task(
        self,
        task_data: Dict[str, Any],
        max_reasoning_depth: int = 3,
        use_frequency_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Solve an ARC task using distributed Hyperon subagents.

        This workflow:
        1. Analyzes task patterns using frequency ledger
        2. Distributes reasoning across subagent pool
        3. Uses RFT for relational reasoning
        4. Synthesizes results into solution

        Parameters
        ----------
        task_data : dict
            ARC task data (train/test examples)
        max_reasoning_depth : int
            Maximum depth for recursive reasoning
        use_frequency_analysis : bool
            Whether to use frequency ledger analysis

        Returns
        -------
        dict
            Solution with reasoning trace
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"Solving ARC task with {len(task_data.get('train', []))} training examples")

        result = {
            "success": False,
            "solution": None,
            "reasoning_trace": [],
            "subagent_results": [],
            "execution_time": 0.0,
        }

        start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Frequency analysis (if enabled)
            if use_frequency_analysis and self.frequency_ledger:
                logger.info("Performing frequency analysis")
                freq_signature = await self._analyze_arc_patterns(task_data)
                result["reasoning_trace"].append(
                    {"step": "frequency_analysis", "signature": str(freq_signature)}
                )

            # Step 2: Create reasoning tasks for each training example
            reasoning_tasks = self._create_arc_reasoning_tasks(task_data)
            logger.info(f"Created {len(reasoning_tasks)} reasoning tasks")

            # Step 3: Execute tasks in parallel using subagent pool
            if self.subagent_manager:
                logger.info("Executing tasks with subagent pool")
                task_results = await self.subagent_manager.execute_parallel(
                    reasoning_tasks
                )
                result["subagent_results"] = [
                    {
                        "agent_id": r.agent_id,
                        "success": r.success,
                        "execution_time": r.execution_time,
                    }
                    for r in task_results
                ]

            # Step 4: Synthesize results
            solution = await self._synthesize_arc_solution(task_results, task_data)
            result["solution"] = solution
            result["success"] = solution is not None

        except Exception as e:
            logger.error(f"Error solving ARC task: {e}")
            result["error"] = str(e)

        result["execution_time"] = asyncio.get_event_loop().time() - start_time
        logger.info(
            f"ARC task solving completed in {result['execution_time']:.2f}s"
        )

        return result

    async def reason_with_rft(
        self,
        source: str,
        target: str,
        relation_type: Optional[RelationType] = None,
        context: Optional[List[str]] = None,
        use_subagents: bool = True,
    ) -> List[RelationalFrame]:
        """
        Perform RFT reasoning using Hyperon subagents.

        This workflow:
        1. Converts RFT frames to MeTTa expressions
        2. Distributes reasoning across subagents
        3. Performs derived relation inference
        4. Returns inferred relational frames

        Parameters
        ----------
        source : str
            Source concept
        target : str
            Target concept
        relation_type : RelationType, optional
            Type of relation to infer
        context : list of str, optional
            Context for reasoning
        use_subagents : bool
            Whether to use subagent pool for reasoning

        Returns
        -------
        list of RelationalFrame
            Inferred relational frames
        """
        if not self.initialized:
            await self.initialize()

        if not self.rft_bridge:
            logger.error("RFT bridge not available")
            return []

        logger.info(f"RFT reasoning: {source} -> {target}")

        # Create base frame
        if self.rft_engine and RelationalFrame and relation_type:
            base_frame = RelationalFrame(
                source=source,
                target=target,
                relation_type=relation_type,
                context=context or [],
            )

            # Convert to MeTTa
            metta_relation = self.rft_bridge.frame_to_metta(base_frame)

            if use_subagents and self.subagent_manager:
                # Create reasoning task
                task = SubAgentTask(
                    task_type="relational_reasoning",
                    metta_program=metta_relation.metta_expr,
                    context={"source": source, "target": target},
                    priority=0.8,
                )

                # Execute with relational framing capability
                result = await self.subagent_manager.execute_task(
                    task, required_capability=AgentCapability.RELATIONAL_FRAMING
                )

                if result.success:
                    # Parse results back to RFT frames
                    frames = self.rft_bridge.parse_metta_results(
                        result.output_atoms
                    )
                    logger.info(f"Inferred {len(frames)} relational frames")
                    return frames

        return []

    async def analyze_frequencies(
        self,
        pattern_data: Dict[str, Any],
        use_metta_inference: bool = True,
    ) -> Optional[FrequencySignature]:
        """
        Perform frequency analysis using MeTTa inference.

        This workflow:
        1. Extracts patterns from data
        2. Uses MeTTa for symbolic pattern matching
        3. Builds frequency signature
        4. Returns analysis results

        Parameters
        ----------
        pattern_data : dict
            Data containing patterns to analyze
        use_metta_inference : bool
            Whether to use MeTTa for pattern inference

        Returns
        -------
        FrequencySignature or None
            Frequency signature of patterns
        """
        if not self.initialized:
            await self.initialize()

        if not self.frequency_ledger:
            logger.error("Frequency ledger not available")
            return None

        logger.info("Performing frequency analysis")

        if use_metta_inference and self.metta_engine:
            # Create MeTTa program for pattern matching
            metta_program = self._create_frequency_analysis_program(pattern_data)

            # Execute with MeTTa engine
            result = self.metta_engine.run(metta_program)

            if result.success:
                # Build frequency signature from results
                signature = self._build_frequency_signature(result.output)
                return signature

        return None

    async def coordinate_consciousness_aware_task(
        self,
        task: SubAgentTask,
        required_state: Optional[ConsciousnessState] = None,
    ) -> SubAgentResult:
        """
        Execute a task with consciousness state awareness.

        This method routes tasks based on current consciousness state,
        ensuring appropriate resource allocation and priority.

        Parameters
        ----------
        task : SubAgentTask
            Task to execute
        required_state : ConsciousnessState, optional
            Required consciousness state for execution

        Returns
        -------
        SubAgentResult
            Task execution result
        """
        if not self.initialized:
            await self.initialize()

        current_state = self._get_current_consciousness_state()

        # Check state compatibility
        if required_state and current_state != required_state:
            logger.warning(
                f"Task requires {required_state} but current state is {current_state}"
            )
            # Optionally request state transition
            if self.consciousness_state_machine:
                await self.consciousness_state_machine.transition_to(
                    required_state, reason="task_requirement"
                )

        # Execute task based on consciousness state
        if current_state == ConsciousnessState.EXPLORING:
            # Use higher parallelism for exploration
            task.priority = min(task.priority + 0.1, 1.0)
        elif current_state == ConsciousnessState.SLEEPING:
            # Lower priority during consolidation
            task.priority = max(task.priority - 0.2, 0.0)

        # Execute with coordinator
        if self.coordinator:
            return await self.coordinator.execute_task(task)
        elif self.subagent_manager:
            return await self.subagent_manager.execute_task(task)
        else:
            raise RuntimeError("No execution system available")

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _analyze_arc_patterns(
        self, task_data: Dict[str, Any]
    ) -> Optional[FrequencySignature]:
        """Analyze patterns in ARC task data"""
        if not self.frequency_ledger:
            return None

        # Extract patterns from training examples
        for example in task_data.get("train", []):
            input_grid = example.get("input", [])
            output_grid = example.get("output", [])

            # Record pattern frequencies
            # (This would integrate with actual pattern extraction)
            pass

        return None

    def _create_arc_reasoning_tasks(
        self, task_data: Dict[str, Any]
    ) -> List[SubAgentTask]:
        """Create reasoning tasks for ARC problem"""
        tasks = []

        for i, example in enumerate(task_data.get("train", [])):
            # Create pattern matching task
            task = SubAgentTask(
                task_type="pattern_matching",
                metta_program=f"""
                ; Analyze training example {i}
                (match &self (pattern $x) $x)
                """,
                context={"example_id": i, "example": example},
                priority=0.7,
            )
            tasks.append(task)

        return tasks

    async def _synthesize_arc_solution(
        self, task_results: List[SubAgentResult], task_data: Dict[str, Any]
    ) -> Optional[Any]:
        """Synthesize solution from subagent results"""
        if not task_results:
            return None

        # Aggregate successful results
        successful_results = [r for r in task_results if r.success]

        if not successful_results:
            return None

        # Combine reasoning (simplified for now)
        return {
            "method": "hyperon_distributed_reasoning",
            "num_agents": len(successful_results),
            "confidence": len(successful_results) / len(task_results),
        }

    def _create_frequency_analysis_program(
        self, pattern_data: Dict[str, Any]
    ) -> str:
        """Create MeTTa program for frequency analysis"""
        return """
        ; Frequency analysis program
        (= (analyze-frequencies $patterns)
           (map extract-frequency $patterns))
        """

    def _build_frequency_signature(self, metta_output: List[Any]) -> FrequencySignature:
        """Build frequency signature from MeTTa output"""
        # Placeholder - would parse MeTTa output into signature
        if FrequencySignature:
            return FrequencySignature(
                color_frequencies={},
                shape_frequencies={},
                position_frequencies={},
                size_frequencies={},
            )
        return None

    # ========================================================================
    # Status and Monitoring
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of Hyperon integration.

        Returns
        -------
        dict
            Status information including:
            - initialized: Whether system is initialized
            - hyperon_available: Whether Hyperon is installed
            - num_subagents: Number of active subagents
            - active_tasks: Number of currently executing tasks
            - rft_bridge_enabled: Whether RFT bridge is available
            - consciousness_integrated: Whether consciousness integration is active
            - memory_integrated: Whether memory integration is active
        """
        status = {
            "initialized": self.initialized,
            "hyperon_available": HYPERON_AVAILABLE,
            "num_subagents": 0,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.task_results),
            "rft_bridge_enabled": self.rft_bridge is not None,
            "consciousness_integrated": self.consciousness_state_machine is not None,
            "memory_integrated": self.memory_system is not None,
        }

        if self.subagent_manager:
            pool_status = self.subagent_manager.get_pool_status()
            status["num_subagents"] = pool_status["total_agents"]
            status["agent_state_distribution"] = pool_status["state_distribution"]
            status["agent_capability_distribution"] = pool_status[
                "capability_distribution"
            ]

        if self.coordinator:
            status["coordination_strategy"] = str(
                self.config.default_coordination_strategy.value
            )
            status["communication_pattern"] = str(
                self.config.default_communication_pattern.value
            )

        return status

    async def shutdown(self) -> None:
        """
        Shutdown all Hyperon components gracefully.

        This method:
        1. Cancels all active tasks
        2. Shuts down subagent manager
        3. Cleans up resources
        """
        logger.info("Shutting down Hyperon-PUMA integration")

        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            logger.info(f"Cancelling task {task_id}")
            del self.active_tasks[task_id]

        # Shutdown subagent manager
        if self.subagent_manager:
            self.subagent_manager.shutdown()
            logger.info("Subagent manager shutdown complete")

        self.initialized = False
        logger.info("Hyperon-PUMA integration shutdown complete")

    def __repr__(self) -> str:
        return (
            f"HyperonPUMAIntegration("
            f"initialized={self.initialized}, "
            f"subagents={len(self.subagent_manager.agents) if self.subagent_manager else 0}, "
            f"rft_bridge={self.rft_bridge is not None})"
        )


# ============================================================================
# Convenience Functions
# ============================================================================


async def create_integration(
    atomspace: Optional[Atomspace] = None,
    rft_engine: Optional[RFTEngine] = None,
    consciousness_state_machine: Optional[ConsciousnessStateMachine] = None,
    memory_system: Optional[EpisodicMemorySystem] = None,
    config: Optional[HyperonConfig] = None,
) -> HyperonPUMAIntegration:
    """
    Create and initialize Hyperon-PUMA integration.

    Convenience function that creates the integration and initializes
    all components in one call.

    Parameters
    ----------
    atomspace : Atomspace, optional
        PUMA's main atomspace
    rft_engine : RFTEngine, optional
        PUMA's RFT reasoning engine
    consciousness_state_machine : ConsciousnessStateMachine, optional
        PUMA's consciousness state manager
    memory_system : EpisodicMemorySystem, optional
        PUMA's episodic memory system
    config : HyperonConfig, optional
        Configuration for Hyperon integration

    Returns
    -------
    HyperonPUMAIntegration
        Initialized integration instance
    """
    integration = HyperonPUMAIntegration(
        atomspace=atomspace,
        rft_engine=rft_engine,
        consciousness_state_machine=consciousness_state_machine,
        memory_system=memory_system,
        config=config,
    )

    await integration.initialize()
    return integration
