"""
Hyperon Subagent Management System

Manages multiple Hyperon MeTTa subagents for parallel reasoning, pattern matching,
memory retrieval, and goal planning within PUMA's cognitive architecture.

Integrates with PUMA's consciousness system, using Atomspace for inter-agent
communication and supporting emergent collective intelligence through distributed
reasoning capabilities.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from threading import Lock
import logging

try:
    from hyperon import MeTTa, GroundingSpace, Atom, SymbolAtom, ExpressionAtom
    HYPERON_AVAILABLE = True
except ImportError:
    # Graceful degradation when Hyperon is not installed
    HYPERON_AVAILABLE = False
    MeTTa = None
    GroundingSpace = None
    Atom = None
    SymbolAtom = None
    ExpressionAtom = None


logger = logging.getLogger("puma.hyperon_subagents.manager")
logger.addHandler(logging.NullHandler())


class SubAgentState(Enum):
    """States for subagent lifecycle management"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


class AgentCapability(Enum):
    """Capabilities that subagents can possess"""
    REASONING = "reasoning"
    PATTERN_MATCHING = "pattern_matching"
    MEMORY_RETRIEVAL = "memory_retrieval"
    GOAL_PLANNING = "goal_planning"
    RELATIONAL_FRAMING = "relational_framing"
    ABSTRACTION = "abstraction"
    ANALOGY_MAKING = "analogy_making"
    CONCEPT_SYNTHESIS = "concept_synthesis"


@dataclass
class SubAgentTask:
    """Task specification for a subagent"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    metta_program: str = ""
    input_atoms: List[Any] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    timeout: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dependencies: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            'id': self.id,
            'task_type': self.task_type,
            'metta_program': self.metta_program,
            'input_atoms': [str(atom) for atom in self.input_atoms],
            'context': self.context,
            'priority': self.priority,
            'timeout': self.timeout,
            'created_at': self.created_at.isoformat(),
            'dependencies': self.dependencies
        }


@dataclass
class SubAgentResult:
    """Result from subagent execution"""
    task_id: str
    agent_id: str
    success: bool
    output_atoms: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation"""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'success': self.success,
            'output_atoms': [str(atom) for atom in self.output_atoms],
            'error': self.error,
            'execution_time': self.execution_time,
            'metadata': self.metadata,
            'completed_at': self.completed_at.isoformat()
        }


class HyperonSubAgent:
    """
    Individual Hyperon MeTTa subagent with its own interpreter instance.

    Each subagent maintains:
    - Its own MeTTa interpreter for isolated reasoning
    - A set of specialized capabilities
    - State management for lifecycle tracking
    - Task execution history for learning and adaptation

    Subagents communicate via shared Atomspace and can perform parallel
    reasoning, pattern matching, memory operations, and goal planning.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        capabilities: Optional[Set[AgentCapability]] = None,
        atomspace: Optional[Any] = None,
        name: Optional[str] = None
    ):
        """
        Initialize a Hyperon subagent.

        Args:
            agent_id: Unique identifier for this agent
            capabilities: Set of capabilities this agent possesses
            atomspace: Shared atomspace for inter-agent communication
            name: Human-readable name for this agent
        """
        self.id = agent_id or str(uuid.uuid4())
        self.name = name or f"SubAgent-{self.id[:8]}"
        self.capabilities = capabilities or {AgentCapability.REASONING}
        self.atomspace = atomspace
        self.state = SubAgentState.IDLE

        # Initialize MeTTa interpreter if Hyperon is available
        if HYPERON_AVAILABLE:
            self.metta = MeTTa()
            if atomspace:
                # Link to shared grounding space for communication
                self._setup_shared_space()
        else:
            self.metta = None
            logger.warning(f"Hyperon not available for {self.name}, using simulation mode")

        # Task execution tracking
        self.current_task: Optional[SubAgentTask] = None
        self.task_history: List[Tuple[SubAgentTask, SubAgentResult]] = []
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

        # Thread safety
        self._lock = Lock()

        # Performance metrics
        self.total_execution_time = 0.0
        self.average_execution_time = 0.0

        # Capability-specific initialization
        self._initialize_capabilities()

        logger.info(f"Initialized {self.name} with capabilities: {[c.value for c in self.capabilities]}")

    def _setup_shared_space(self):
        """Set up connection to shared grounding space for inter-agent communication"""
        if not HYPERON_AVAILABLE or not self.metta:
            return

        # Add standard library imports
        self.metta.run("!(import! &self std)")

        # Define communication primitives
        communication_metta = """
        ; Inter-agent communication primitives
        (= (send-to-agent $agent $message)
           (add-atom &shared (MessageAtom $agent $message)))

        (= (receive-from-agent $agent)
           (match &shared (MessageAtom $agent $msg) $msg))

        (= (broadcast-to-all $message)
           (add-atom &shared (BroadcastAtom $message)))
        """

        try:
            self.metta.run(communication_metta)
        except Exception as e:
            logger.warning(f"Could not set up communication primitives: {e}")

    def _initialize_capabilities(self):
        """Initialize capability-specific MeTTa programs"""
        if not HYPERON_AVAILABLE or not self.metta:
            return

        # Reasoning capability
        if AgentCapability.REASONING in self.capabilities:
            reasoning_metta = """
            ; Forward chaining reasoning
            (= (infer $premise $rule)
               (match &self ($rule $premise $conclusion) $conclusion))

            ; Backward chaining
            (= (prove $goal $premises)
               (chain $goal $premises))

            ; Relational Frame Theory integration
            (= (derive-relation $a $b $frame)
               (match &self ($frame $a $b) True))
            """
            try:
                self.metta.run(reasoning_metta)
            except Exception as e:
                logger.warning(f"Could not initialize reasoning: {e}")

        # Pattern matching capability
        if AgentCapability.PATTERN_MATCHING in self.capabilities:
            pattern_metta = """
            ; Pattern matching primitives
            (= (find-pattern $pattern)
               (match &self $pattern $result))

            (= (match-all $pattern $space)
               (collapse (match $space $pattern $result)))

            ; Frequency-based pattern analysis (PUMA's Frequency Ledger)
            (= (frequency-analysis $objects)
               (group-by-frequency $objects))
            """
            try:
                self.metta.run(pattern_metta)
            except Exception as e:
                logger.warning(f"Could not initialize pattern matching: {e}")

        # Memory retrieval capability
        if AgentCapability.MEMORY_RETRIEVAL in self.capabilities:
            memory_metta = """
            ; Memory retrieval operations
            (= (retrieve-episode $query)
               (match &memory (Episode $props) (filter $props $query)))

            (= (temporal-query $start $end)
               (match &memory (Episode $props)
                  (and (>= (timestamp $props) $start)
                       (<= (timestamp $props) $end))))

            ; Autobiographical memory access
            (= (recall-similar $episode)
               (match &memory (Episode $props)
                  (similar $episode $props)))
            """
            try:
                self.metta.run(memory_metta)
            except Exception as e:
                logger.warning(f"Could not initialize memory retrieval: {e}")

        # Goal planning capability
        if AgentCapability.GOAL_PLANNING in self.capabilities:
            planning_metta = """
            ; Goal planning operations
            (= (plan-goal $goal $state)
               (hierarchical-task-network $goal $state))

            (= (decompose-goal $goal)
               (match &self (GoalDecomposition $goal $subgoals) $subgoals))

            ; Intention formation from drives
            (= (form-intention $drive $context)
               (synthesize-goal $drive $context))
            """
            try:
                self.metta.run(planning_metta)
            except Exception as e:
                logger.warning(f"Could not initialize goal planning: {e}")

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability"""
        return capability in self.capabilities

    def add_capability(self, capability: AgentCapability):
        """Add a new capability to this agent"""
        with self._lock:
            self.capabilities.add(capability)
            self._initialize_capabilities()

    async def execute_task(self, task: SubAgentTask) -> SubAgentResult:
        """
        Execute a task using this subagent's MeTTa interpreter.

        Args:
            task: Task specification to execute

        Returns:
            SubAgentResult containing execution results
        """
        start_time = datetime.now(timezone.utc)

        with self._lock:
            if self.state != SubAgentState.IDLE:
                return SubAgentResult(
                    task_id=task.id,
                    agent_id=self.id,
                    success=False,
                    error=f"Agent {self.name} is not idle (state: {self.state.value})"
                )

            self.state = SubAgentState.RUNNING
            self.current_task = task

        try:
            # Execute MeTTa program
            result = await self._execute_metta_program(task)

            # Update metrics
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.total_execution_time += execution_time
            self.execution_count += 1
            self.average_execution_time = self.total_execution_time / self.execution_count

            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1

            # Store in history
            self.task_history.append((task, result))

            # Update state
            with self._lock:
                self.state = SubAgentState.COMPLETED
                self.current_task = None

            return result

        except Exception as e:
            logger.exception(f"Error executing task {task.id} on {self.name}")

            with self._lock:
                self.state = SubAgentState.FAILED
                self.current_task = None
                self.failure_count += 1

            return SubAgentResult(
                task_id=task.id,
                agent_id=self.id,
                success=False,
                error=str(e)
            )

    async def _execute_metta_program(self, task: SubAgentTask) -> SubAgentResult:
        """Execute MeTTa program with timeout and error handling"""
        if not HYPERON_AVAILABLE or not self.metta:
            # Simulation mode for testing without Hyperon
            await asyncio.sleep(0.1)  # Simulate processing
            return SubAgentResult(
                task_id=task.id,
                agent_id=self.id,
                success=True,
                output_atoms=[f"Simulated result for {task.task_type}"],
                metadata={'simulation_mode': True}
            )

        try:
            # Add input atoms to space
            for atom in task.input_atoms:
                if isinstance(atom, str):
                    self.metta.run(f"(add-atom &self {atom})")
                else:
                    self.metta.space().add_atom(atom)

            # Execute program
            result = self.metta.run(task.metta_program)

            # Extract output atoms
            output_atoms = []
            if result:
                output_atoms = list(result)

            return SubAgentResult(
                task_id=task.id,
                agent_id=self.id,
                success=True,
                output_atoms=output_atoms,
                metadata={
                    'program_length': len(task.metta_program),
                    'output_count': len(output_atoms)
                }
            )

        except Exception as e:
            logger.error(f"MeTTa execution error in {self.name}: {e}")
            return SubAgentResult(
                task_id=task.id,
                agent_id=self.id,
                success=False,
                error=str(e)
            )

    def reset(self):
        """Reset agent to idle state"""
        with self._lock:
            self.state = SubAgentState.IDLE
            self.current_task = None

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent"""
        return {
            'agent_id': self.id,
            'name': self.name,
            'state': self.state.value,
            'capabilities': [c.value for c in self.capabilities],
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / self.execution_count if self.execution_count > 0 else 0.0,
            'average_execution_time': self.average_execution_time,
            'total_execution_time': self.total_execution_time
        }

    def __repr__(self) -> str:
        return f"HyperonSubAgent(id={self.id[:8]}, name={self.name}, state={self.state.value})"


class SubAgentManager:
    """
    Manages a pool of Hyperon subagents for parallel distributed reasoning.

    The SubAgentManager coordinates multiple specialized subagents, enabling:
    - Parallel execution of MeTTa programs across agent pool
    - Task routing based on agent capabilities
    - Load balancing and resource management
    - Inter-agent communication via shared Atomspace
    - Collective intelligence through distributed reasoning

    Integrates with PUMA's consciousness system to support autonomous
    cognitive processes including exploration, learning, and goal formation.
    """

    def __init__(
        self,
        atomspace: Optional[Any] = None,
        consciousness_state_machine: Optional[Any] = None,
        memory_system: Optional[Any] = None,
        goal_system: Optional[Any] = None,
        max_agents: int = 10
    ):
        """
        Initialize the subagent manager.

        Args:
            atomspace: Shared atomspace for inter-agent communication
            consciousness_state_machine: PUMA consciousness state machine
            memory_system: PUMA episodic memory system
            goal_system: PUMA goal formation system
            max_agents: Maximum number of subagents in the pool
        """
        self.atomspace = atomspace
        self.consciousness = consciousness_state_machine
        self.memory_system = memory_system
        self.goal_system = goal_system
        self.max_agents = max_agents

        # Agent pool management
        self.agents: Dict[str, HyperonSubAgent] = {}
        self.agent_pool: List[HyperonSubAgent] = []
        self._pool_lock = Lock()

        # Task queue and execution
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.pending_tasks: Dict[str, SubAgentTask] = {}
        self.completed_tasks: Dict[str, SubAgentResult] = {}
        self.running = False

        # Thread pool for parallel execution
        self.thread_pool = ThreadPoolExecutor(max_workers=max_agents)

        # Communication channels
        self.message_bus: Dict[str, List[Any]] = {}
        self._message_lock = Lock()

        logger.info(f"Initialized SubAgentManager with max {max_agents} agents")

    def create_agent(
        self,
        capabilities: Optional[Set[AgentCapability]] = None,
        name: Optional[str] = None
    ) -> HyperonSubAgent:
        """
        Create and register a new subagent.

        Args:
            capabilities: Capabilities for the new agent
            name: Optional name for the agent

        Returns:
            Created HyperonSubAgent instance
        """
        if len(self.agents) >= self.max_agents:
            raise RuntimeError(f"Maximum agent limit ({self.max_agents}) reached")

        agent = HyperonSubAgent(
            capabilities=capabilities,
            atomspace=self.atomspace,
            name=name
        )

        with self._pool_lock:
            self.agents[agent.id] = agent
            self.agent_pool.append(agent)

        logger.info(f"Created {agent.name} with ID {agent.id}")
        return agent

    def create_specialized_agents(self):
        """
        Create a set of specialized agents for different cognitive tasks.

        This creates a default agent pool with:
        - Reasoning specialists
        - Pattern matching specialists
        - Memory retrieval specialists
        - Goal planning specialists
        """
        # Create reasoning agents
        for i in range(2):
            self.create_agent(
                capabilities={AgentCapability.REASONING, AgentCapability.RELATIONAL_FRAMING},
                name=f"Reasoner-{i+1}"
            )

        # Create pattern matching agents
        for i in range(2):
            self.create_agent(
                capabilities={AgentCapability.PATTERN_MATCHING, AgentCapability.ABSTRACTION},
                name=f"PatternMatcher-{i+1}"
            )

        # Create memory retrieval agents
        for i in range(2):
            self.create_agent(
                capabilities={AgentCapability.MEMORY_RETRIEVAL},
                name=f"MemoryRetriever-{i+1}"
            )

        # Create goal planning agents
        for i in range(2):
            self.create_agent(
                capabilities={AgentCapability.GOAL_PLANNING, AgentCapability.CONCEPT_SYNTHESIS},
                name=f"GoalPlanner-{i+1}"
            )

        # Create multi-capability agent
        self.create_agent(
            capabilities={
                AgentCapability.REASONING,
                AgentCapability.PATTERN_MATCHING,
                AgentCapability.ANALOGY_MAKING
            },
            name="GeneralAgent"
        )

        logger.info(f"Created specialized agent pool: {len(self.agents)} agents")

    def find_capable_agent(
        self,
        required_capability: AgentCapability,
        prefer_idle: bool = True
    ) -> Optional[HyperonSubAgent]:
        """
        Find an agent with the required capability.

        Args:
            required_capability: Capability needed for the task
            prefer_idle: Prefer agents in IDLE state

        Returns:
            HyperonSubAgent with required capability, or None if not found
        """
        with self._pool_lock:
            candidates = [
                agent for agent in self.agent_pool
                if agent.has_capability(required_capability)
            ]

            if not candidates:
                return None

            if prefer_idle:
                idle_candidates = [
                    agent for agent in candidates
                    if agent.state == SubAgentState.IDLE
                ]
                if idle_candidates:
                    # Return least utilized idle agent
                    return min(idle_candidates, key=lambda a: a.execution_count)

            # Return least utilized agent overall
            return min(candidates, key=lambda a: a.execution_count)

    def find_agents_with_capability(
        self,
        required_capability: AgentCapability
    ) -> List[HyperonSubAgent]:
        """Find all agents with a specific capability"""
        with self._pool_lock:
            return [
                agent for agent in self.agent_pool
                if agent.has_capability(required_capability)
            ]

    async def submit_task(self, task: SubAgentTask) -> str:
        """
        Submit a task to the execution queue.

        Args:
            task: Task to execute

        Returns:
            Task ID for tracking
        """
        self.pending_tasks[task.id] = task
        await self.task_queue.put(task)
        logger.debug(f"Submitted task {task.id} ({task.task_type})")
        return task.id

    async def execute_task(
        self,
        task: SubAgentTask,
        required_capability: Optional[AgentCapability] = None
    ) -> SubAgentResult:
        """
        Execute a task immediately on an available agent.

        Args:
            task: Task to execute
            required_capability: Required agent capability

        Returns:
            SubAgentResult from execution
        """
        # Find suitable agent
        if required_capability:
            agent = self.find_capable_agent(required_capability)
        else:
            # Find any idle agent
            agent = self.find_capable_agent(
                AgentCapability.REASONING,  # Default capability
                prefer_idle=True
            )

        if not agent:
            return SubAgentResult(
                task_id=task.id,
                agent_id="none",
                success=False,
                error=f"No agent available with capability: {required_capability}"
            )

        # Execute task
        result = await agent.execute_task(task)
        self.completed_tasks[task.id] = result

        # Record in memory system if available
        if self.memory_system:
            self._record_task_in_memory(task, result)

        return result

    async def execute_parallel(
        self,
        tasks: List[SubAgentTask]
    ) -> List[SubAgentResult]:
        """
        Execute multiple tasks in parallel across available agents.

        Args:
            tasks: List of tasks to execute in parallel

        Returns:
            List of results from all tasks
        """
        # Create async tasks for each subagent task
        task_futures = [
            self.execute_task(task)
            for task in tasks
        ]

        # Wait for all to complete
        results = await asyncio.gather(*task_futures, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(SubAgentResult(
                    task_id=tasks[i].id,
                    agent_id="error",
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def map_reduce_reasoning(
        self,
        metta_programs: List[str],
        reduce_program: str,
        context: Optional[Dict[str, Any]] = None
    ) -> SubAgentResult:
        """
        Perform map-reduce style distributed reasoning.

        Args:
            metta_programs: List of MeTTa programs to execute in parallel (map phase)
            reduce_program: MeTTa program to combine results (reduce phase)
            context: Shared context for all tasks

        Returns:
            Combined result from reduce phase
        """
        # Map phase: execute programs in parallel
        map_tasks = [
            SubAgentTask(
                task_type="map_reasoning",
                metta_program=program,
                context=context or {},
                priority=0.8
            )
            for program in metta_programs
        ]

        map_results = await self.execute_parallel(map_tasks)

        # Reduce phase: combine results
        all_outputs = []
        for result in map_results:
            if result.success:
                all_outputs.extend(result.output_atoms)

        reduce_task = SubAgentTask(
            task_type="reduce_reasoning",
            metta_program=reduce_program,
            input_atoms=all_outputs,
            context=context or {},
            priority=0.9
        )

        reduce_result = await self.execute_task(reduce_task)
        return reduce_result

    def broadcast_message(self, message: Any, sender_id: Optional[str] = None):
        """
        Broadcast a message to all agents via the message bus.

        Args:
            message: Message to broadcast
            sender_id: ID of sending agent (optional)
        """
        with self._message_lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            broadcast = {
                'sender': sender_id or 'manager',
                'message': message,
                'timestamp': timestamp,
                'type': 'broadcast'
            }

            for agent_id in self.agents:
                if agent_id not in self.message_bus:
                    self.message_bus[agent_id] = []
                self.message_bus[agent_id].append(broadcast)

        logger.debug(f"Broadcast message to {len(self.agents)} agents")

    def send_message(self, recipient_id: str, message: Any, sender_id: Optional[str] = None):
        """
        Send a message to a specific agent.

        Args:
            recipient_id: ID of recipient agent
            message: Message to send
            sender_id: ID of sending agent (optional)
        """
        with self._message_lock:
            if recipient_id not in self.agents:
                logger.warning(f"Recipient {recipient_id} not found")
                return

            timestamp = datetime.now(timezone.utc).isoformat()
            msg = {
                'sender': sender_id or 'manager',
                'message': message,
                'timestamp': timestamp,
                'type': 'direct'
            }

            if recipient_id not in self.message_bus:
                self.message_bus[recipient_id] = []
            self.message_bus[recipient_id].append(msg)

    def get_messages(self, agent_id: str, clear: bool = True) -> List[Dict[str, Any]]:
        """
        Get messages for a specific agent.

        Args:
            agent_id: ID of agent to get messages for
            clear: Whether to clear messages after retrieval

        Returns:
            List of messages for the agent
        """
        with self._message_lock:
            messages = self.message_bus.get(agent_id, [])
            if clear:
                self.message_bus[agent_id] = []
            return messages

    def _record_task_in_memory(self, task: SubAgentTask, result: SubAgentResult):
        """Record task execution in PUMA's memory system"""
        if not self.memory_system:
            return

        try:
            from puma.memory.episodic import MemoryType

            self.memory_system.form_episode(
                perception={
                    'task_type': task.task_type,
                    'agent_id': result.agent_id,
                    'task_id': task.id
                },
                action={
                    'type': 'subagent_execution',
                    'metta_program': task.metta_program[:100]  # Truncate for storage
                },
                outcome={
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'output_count': len(result.output_atoms)
                },
                memory_type=MemoryType.LEARNING
            )
        except Exception as e:
            logger.warning(f"Could not record task in memory: {e}")

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get status of entire agent pool.

        Returns:
            Dictionary with pool statistics
        """
        with self._pool_lock:
            state_counts = {}
            for state in SubAgentState:
                state_counts[state.value] = sum(
                    1 for agent in self.agent_pool
                    if agent.state == state
                )

            capability_counts = {}
            for capability in AgentCapability:
                capability_counts[capability.value] = sum(
                    1 for agent in self.agent_pool
                    if agent.has_capability(capability)
                )

            return {
                'total_agents': len(self.agents),
                'max_agents': self.max_agents,
                'state_distribution': state_counts,
                'capability_distribution': capability_counts,
                'pending_tasks': len(self.pending_tasks),
                'completed_tasks': len(self.completed_tasks),
                'average_success_rate': sum(
                    agent.success_count / agent.execution_count
                    if agent.execution_count > 0 else 0.0
                    for agent in self.agent_pool
                ) / len(self.agent_pool) if self.agent_pool else 0.0
            }

    def get_agent_metrics(self) -> List[Dict[str, Any]]:
        """Get performance metrics for all agents"""
        with self._pool_lock:
            return [agent.get_metrics() for agent in self.agent_pool]

    def shutdown(self):
        """Shutdown the manager and all subagents"""
        logger.info("Shutting down SubAgentManager")

        # Reset all agents
        with self._pool_lock:
            for agent in self.agent_pool:
                agent.reset()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("SubAgentManager shutdown complete")

    def __repr__(self) -> str:
        return f"SubAgentManager(agents={len(self.agents)}, max={self.max_agents})"
