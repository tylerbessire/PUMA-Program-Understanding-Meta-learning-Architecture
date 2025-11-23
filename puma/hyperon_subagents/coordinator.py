"""
SubAgent Coordination System

Manages parallel subagent execution with sophisticated coordination strategies,
communication patterns, and fault tolerance. Integrates with PUMA consciousness
states and Hyperon Atomspace for distributed cognitive processing.

Architecture:
- Task distribution and load balancing
- Multiple communication patterns (broadcast, P2P, pub-sub)
- Coordination strategies (parallel, sequential, competitive)
- Result aggregation and consensus mechanisms
- Fault tolerance with retry logic
- Integration with PUMA consciousness states
- Monitoring and debugging capabilities
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict
import traceback
import uuid

# PUMA imports
try:
    from puma.consciousness.state_machine import ConsciousnessState
except ImportError:
    ConsciousnessState = None

# Atomspace imports
try:
    from atomspace_db.core import Atomspace, Atom, AtomType, Link
except ImportError:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from atomspace_db.core import Atomspace, Atom, AtomType, Link
    except ImportError:
        Atomspace = None
        Atom = None
        AtomType = None
        Link = None

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# Enums and Configuration
# ============================================================================


class CoordinationStrategy(Enum):
    """Strategy for coordinating subagent execution"""
    PARALLEL = "parallel"  # Execute all tasks concurrently
    SEQUENTIAL = "sequential"  # Execute tasks one after another with dependencies
    COMPETITIVE = "competitive"  # Multiple agents solve same task, best wins
    PIPELINE = "pipeline"  # Sequential with output passing
    HIERARCHICAL = "hierarchical"  # Tree-based delegation
    CONSENSUS = "consensus"  # Require agreement from multiple agents


class CommunicationPattern(Enum):
    """Pattern for inter-agent communication"""
    BROADCAST = "broadcast"  # One-to-all communication
    POINT_TO_POINT = "point_to_point"  # Direct agent-to-agent
    PUBLISH_SUBSCRIBE = "publish_subscribe"  # Topic-based messaging via Atomspace
    REQUEST_REPLY = "request_reply"  # Synchronous request-response
    SHARED_MEMORY = "shared_memory"  # Communication via shared Atomspace


class SubAgentStatus(Enum):
    """Status of a subagent"""
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for task execution"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class SubAgentTask:
    """Task to be executed by a subagent"""
    task_id: str
    name: str
    function: Callable
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class TaskResult:
    """Result of a task execution"""
    task_id: str
    agent_id: str
    status: SubAgentStatus
    result: Any = None
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == SubAgentStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'status': self.status.value,
            'result': self.result,
            'error': str(self.error) if self.error else None,
            'error_traceback': self.error_traceback,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time': self.execution_time,
            'retry_count': self.retry_count,
            'metadata': self.metadata,
        }


@dataclass
class SubAgent:
    """Represents a subagent in the system"""
    agent_id: str
    name: str
    status: SubAgentStatus = SubAgentStatus.IDLE
    current_task: Optional[str] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.tasks_completed / total

    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time"""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_execution_time / self.tasks_completed


@dataclass
class Message:
    """Inter-agent message"""
    message_id: str
    sender_id: str
    receiver_ids: List[str]
    topic: str
    content: Any
    pattern: CommunicationPattern
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reply_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationMetrics:
    """Metrics for coordination performance"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    average_execution_time: float = 0.0
    total_execution_time: float = 0.0
    active_agents: int = 0
    idle_agents: int = 0
    messages_sent: int = 0
    consensus_achieved: int = 0
    consensus_failed: int = 0


# ============================================================================
# Main Coordinator Class
# ============================================================================


class SubAgentCoordinator:
    """
    Coordinates parallel subagent execution with sophisticated strategies.

    Features:
    - Task distribution and load balancing
    - Multiple communication patterns
    - Coordination strategies (parallel, sequential, competitive, etc.)
    - Result aggregation and consensus
    - Fault tolerance and retry logic
    - Integration with PUMA consciousness states
    - Monitoring and debugging
    """

    def __init__(
        self,
        atomspace: Optional[Atomspace] = None,
        max_agents: int = 10,
        default_strategy: CoordinationStrategy = CoordinationStrategy.PARALLEL,
        enable_atomspace_pubsub: bool = True,
        consciousness_integration: bool = True,
    ):
        """
        Initialize the coordinator.

        Args:
            atomspace: Atomspace instance for shared memory and pub-sub
            max_agents: Maximum number of concurrent agents
            default_strategy: Default coordination strategy
            enable_atomspace_pubsub: Enable Atomspace-based pub-sub
            consciousness_integration: Enable PUMA consciousness state integration
        """
        self.atomspace = atomspace
        self.max_agents = max_agents
        self.default_strategy = default_strategy
        self.enable_atomspace_pubsub = enable_atomspace_pubsub
        self.consciousness_integration = consciousness_integration

        # Agent management
        self.agents: Dict[str, SubAgent] = {}
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: Dict[str, SubAgentTask] = {}
        self.task_results: Dict[str, TaskResult] = {}

        # Communication
        self.message_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self.topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self.pending_requests: Dict[str, asyncio.Future] = {}

        # Coordination state
        self.running = False
        self.coordination_lock = asyncio.Lock()
        self.task_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.completed_tasks: Set[str] = set()

        # Metrics and monitoring
        self.metrics = CoordinationMetrics()
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Consciousness state (if integrated)
        self.current_consciousness_state: Optional[ConsciousnessState] = None
        if consciousness_integration and ConsciousnessState:
            self.current_consciousness_state = ConsciousnessState.IDLE

        logger.info(
            f"SubAgentCoordinator initialized: "
            f"max_agents={max_agents}, "
            f"strategy={default_strategy.value}, "
            f"atomspace={'enabled' if atomspace else 'disabled'}"
        )

    # ========================================================================
    # Agent Management
    # ========================================================================

    def register_agent(
        self,
        agent_id: str,
        name: str,
        capabilities: Optional[Set[str]] = None,
        max_concurrent_tasks: int = 1,
    ) -> SubAgent:
        """
        Register a new subagent.

        Args:
            agent_id: Unique agent identifier
            name: Human-readable agent name
            capabilities: Set of capabilities this agent has
            max_concurrent_tasks: Maximum concurrent tasks for this agent

        Returns:
            SubAgent instance
        """
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Maximum agent limit ({self.max_agents}) reached")

        if agent_id in self.agents:
            logger.warning(f"Agent {agent_id} already registered, updating...")

        agent = SubAgent(
            agent_id=agent_id,
            name=name,
            capabilities=capabilities or set(),
            max_concurrent_tasks=max_concurrent_tasks,
        )

        self.agents[agent_id] = agent
        self.metrics.active_agents = len(self.agents)

        logger.info(f"Registered agent: {name} ({agent_id})")
        self._trigger_event('agent_registered', agent=agent)

        return agent

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister a subagent.

        Args:
            agent_id: Agent to unregister

        Returns:
            True if agent was unregistered, False if not found
        """
        if agent_id not in self.agents:
            return False

        agent = self.agents[agent_id]

        # Cancel any active tasks
        if agent.current_task:
            self.cancel_task(agent.current_task)

        del self.agents[agent_id]
        self.metrics.active_agents = len(self.agents)

        logger.info(f"Unregistered agent: {agent.name} ({agent_id})")
        self._trigger_event('agent_unregistered', agent_id=agent_id)

        return True

    def get_agent(self, agent_id: str) -> Optional[SubAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def list_agents(
        self,
        status: Optional[SubAgentStatus] = None,
        capability: Optional[str] = None,
    ) -> List[SubAgent]:
        """
        List agents, optionally filtered by status or capability.

        Args:
            status: Filter by agent status
            capability: Filter by capability

        Returns:
            List of matching agents
        """
        agents = list(self.agents.values())

        if status:
            agents = [a for a in agents if a.status == status]

        if capability:
            agents = [a for a in agents if capability in a.capabilities]

        return agents

    def get_best_agent_for_task(
        self,
        task: SubAgentTask,
        required_capability: Optional[str] = None,
    ) -> Optional[str]:
        """
        Select the best agent for a task using load balancing.

        Args:
            task: Task to assign
            required_capability: Required capability

        Returns:
            Agent ID or None if no suitable agent found
        """
        # Filter eligible agents
        eligible = [
            a for a in self.agents.values()
            if a.status in (SubAgentStatus.IDLE, SubAgentStatus.RUNNING)
            and (not required_capability or required_capability in a.capabilities)
        ]

        if not eligible:
            return None

        # Score agents based on:
        # 1. Success rate
        # 2. Current load
        # 3. Average execution time
        def score_agent(agent: SubAgent) -> float:
            load = 1.0 if agent.current_task else 0.0
            success_rate = agent.success_rate
            avg_time = agent.average_execution_time or 1.0

            # Lower is better
            return load + (1.0 - success_rate) + (avg_time / 10.0)

        best_agent = min(eligible, key=score_agent)
        return best_agent.agent_id

    # ========================================================================
    # Task Management
    # ========================================================================

    async def submit_task(
        self,
        function: Callable,
        *args,
        name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Submit a task for execution.

        Args:
            function: Callable to execute
            args: Positional arguments
            name: Task name
            priority: Task priority
            dependencies: List of task IDs this task depends on
            timeout: Execution timeout in seconds
            max_retries: Maximum retry attempts
            metadata: Additional metadata
            kwargs: Keyword arguments

        Returns:
            Task ID
        """
        task = SubAgentTask(
            task_id="",  # Will be generated in __post_init__
            name=name or function.__name__,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        # Store dependencies
        if task.dependencies:
            self.task_dependencies[task.task_id] = set(task.dependencies)

        # Add to queue (priority queue uses tuple: (priority, task))
        await self.task_queue.put((task.priority.value, task))
        self.active_tasks[task.task_id] = task
        self.metrics.total_tasks += 1

        logger.info(
            f"Task submitted: {task.name} ({task.task_id}) "
            f"priority={task.priority.value}"
        )
        self._trigger_event('task_submitted', task=task)

        return task.task_id

    async def execute_task(
        self,
        task: SubAgentTask,
        agent_id: str,
    ) -> TaskResult:
        """
        Execute a task on a specific agent.

        Args:
            task: Task to execute
            agent_id: Agent to execute on

        Returns:
            TaskResult
        """
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")

        result = TaskResult(
            task_id=task.task_id,
            agent_id=agent_id,
            status=SubAgentStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # Update agent status
        agent.status = SubAgentStatus.RUNNING
        agent.current_task = task.task_id
        agent.last_activity = datetime.now(timezone.utc)

        logger.debug(f"Executing task {task.name} on agent {agent.name}")

        try:
            # Execute with timeout
            if task.timeout:
                result.result = await asyncio.wait_for(
                    self._run_task_function(task),
                    timeout=task.timeout,
                )
            else:
                result.result = await self._run_task_function(task)

            result.status = SubAgentStatus.COMPLETED
            result.completed_at = datetime.now(timezone.utc)
            result.execution_time = (
                result.completed_at - result.started_at
            ).total_seconds()

            # Update agent metrics
            agent.tasks_completed += 1
            agent.total_execution_time += result.execution_time
            self.metrics.completed_tasks += 1

            logger.info(
                f"Task completed: {task.name} ({task.task_id}) "
                f"in {result.execution_time:.2f}s"
            )

        except asyncio.TimeoutError as e:
            result.status = SubAgentStatus.FAILED
            result.error = e
            result.error_traceback = traceback.format_exc()
            agent.tasks_failed += 1
            self.metrics.failed_tasks += 1
            logger.error(f"Task timeout: {task.name} ({task.task_id})")

        except Exception as e:
            result.status = SubAgentStatus.FAILED
            result.error = e
            result.error_traceback = traceback.format_exc()
            agent.tasks_failed += 1
            self.metrics.failed_tasks += 1
            logger.error(
                f"Task failed: {task.name} ({task.task_id}): {str(e)}",
                exc_info=True,
            )

        finally:
            # Update agent status
            agent.status = SubAgentStatus.IDLE
            agent.current_task = None
            agent.last_activity = datetime.now(timezone.utc)

            # Store result
            self.task_results[task.task_id] = result
            self.completed_tasks.add(task.task_id)

            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

            self._trigger_event('task_completed', task=task, result=result)

        return result

    async def _run_task_function(self, task: SubAgentTask) -> Any:
        """
        Run task function, handling both sync and async functions.

        Args:
            task: Task to run

        Returns:
            Task result
        """
        if asyncio.iscoroutinefunction(task.function):
            return await task.function(*task.args, **task.kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: task.function(*task.args, **task.kwargs),
            )

    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> TaskResult:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait

        Returns:
            TaskResult

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            ValueError: If task not found
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            if task_id in self.task_results:
                return self.task_results[task_id]

            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    raise asyncio.TimeoutError(
                        f"Timeout waiting for task {task_id}"
                    )

            await asyncio.sleep(0.1)

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        if task_id in self.completed_tasks:
            return False

        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]

            # Create cancelled result
            result = TaskResult(
                task_id=task_id,
                agent_id="",
                status=SubAgentStatus.CANCELLED,
            )
            self.task_results[task_id] = result
            del self.active_tasks[task_id]
            self.completed_tasks.add(task_id)
            self.metrics.cancelled_tasks += 1

            logger.info(f"Task cancelled: {task.name} ({task_id})")
            self._trigger_event('task_cancelled', task_id=task_id)

            return True

        return False

    # ========================================================================
    # Coordination Strategies
    # ========================================================================

    async def execute_parallel(
        self,
        tasks: List[SubAgentTask],
        return_exceptions: bool = False,
    ) -> List[TaskResult]:
        """
        Execute tasks in parallel.

        Args:
            tasks: Tasks to execute
            return_exceptions: If True, exceptions are returned as results

        Returns:
            List of TaskResults in same order as tasks
        """
        logger.info(f"Executing {len(tasks)} tasks in parallel")

        # Assign tasks to agents
        task_assignments = []
        for task in tasks:
            agent_id = self.get_best_agent_for_task(task)
            if not agent_id:
                logger.warning(f"No agent available for task {task.name}")
                continue
            task_assignments.append((task, agent_id))

        # Execute all tasks concurrently
        execution_coros = [
            self.execute_task(task, agent_id)
            for task, agent_id in task_assignments
        ]

        if return_exceptions:
            results = await asyncio.gather(*execution_coros, return_exceptions=True)
        else:
            results = await asyncio.gather(*execution_coros)

        return results

    async def execute_sequential(
        self,
        tasks: List[SubAgentTask],
    ) -> List[TaskResult]:
        """
        Execute tasks sequentially with dependency management.

        Args:
            tasks: Tasks to execute (may have dependencies)

        Returns:
            List of TaskResults
        """
        logger.info(f"Executing {len(tasks)} tasks sequentially")

        results = []

        # Topological sort based on dependencies
        sorted_tasks = self._topological_sort(tasks)

        for task in sorted_tasks:
            # Wait for dependencies to complete
            if task.dependencies:
                for dep_id in task.dependencies:
                    if dep_id not in self.completed_tasks:
                        await self.wait_for_task(dep_id)

            # Execute task
            agent_id = self.get_best_agent_for_task(task)
            if not agent_id:
                logger.error(f"No agent available for task {task.name}")
                result = TaskResult(
                    task_id=task.task_id,
                    agent_id="",
                    status=SubAgentStatus.FAILED,
                    error=Exception("No agent available"),
                )
                results.append(result)
                continue

            result = await self.execute_task(task, agent_id)
            results.append(result)

            # Stop on failure if dependency chain
            if not result.success and task.dependencies:
                logger.warning(
                    f"Task {task.name} failed, stopping sequential execution"
                )
                break

        return results

    async def execute_competitive(
        self,
        task: SubAgentTask,
        num_agents: int = 3,
        selection_strategy: str = 'first',
    ) -> TaskResult:
        """
        Execute same task on multiple agents, select best result.

        Args:
            task: Task to execute
            num_agents: Number of agents to compete
            selection_strategy: 'first', 'fastest', or 'best_quality'

        Returns:
            Best TaskResult
        """
        logger.info(
            f"Executing task {task.name} competitively on {num_agents} agents"
        )

        # Get available agents
        available = self.list_agents(status=SubAgentStatus.IDLE)
        if len(available) < num_agents:
            num_agents = len(available)

        if num_agents == 0:
            raise ValueError("No agents available for competitive execution")

        # Create copies of task for each agent
        agent_ids = [a.agent_id for a in available[:num_agents]]
        execution_coros = [
            self.execute_task(task, agent_id)
            for agent_id in agent_ids
        ]

        if selection_strategy == 'first':
            # Return first completed result
            done, pending = await asyncio.wait(
                execution_coros,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel remaining tasks
            for p in pending:
                p.cancel()

            best_result = list(done)[0].result()

        elif selection_strategy == 'fastest':
            # Wait for all, return fastest successful one
            results = await asyncio.gather(*execution_coros, return_exceptions=True)
            successful = [r for r in results if isinstance(r, TaskResult) and r.success]

            if not successful:
                best_result = results[0] if results else None
            else:
                best_result = min(successful, key=lambda r: r.execution_time or float('inf'))

        else:  # 'best_quality'
            # Wait for all, use custom quality metric
            results = await asyncio.gather(*execution_coros, return_exceptions=True)
            successful = [r for r in results if isinstance(r, TaskResult) and r.success]

            if not successful:
                best_result = results[0] if results else None
            else:
                # Use result metadata for quality score
                best_result = max(
                    successful,
                    key=lambda r: r.metadata.get('quality_score', 0.5),
                )

        logger.info(
            f"Competitive execution complete, best agent: {best_result.agent_id}"
        )
        self.metrics.consensus_achieved += 1

        return best_result

    async def execute_pipeline(
        self,
        tasks: List[SubAgentTask],
    ) -> TaskResult:
        """
        Execute tasks in pipeline (sequential with output passing).

        Args:
            tasks: Tasks to execute in pipeline order

        Returns:
            Final TaskResult
        """
        logger.info(f"Executing {len(tasks)} tasks in pipeline")

        previous_result = None

        for i, task in enumerate(tasks):
            # Pass previous result as input
            if previous_result is not None and previous_result.success:
                task.kwargs['input'] = previous_result.result

            agent_id = self.get_best_agent_for_task(task)
            if not agent_id:
                raise ValueError(f"No agent available for task {task.name}")

            result = await self.execute_task(task, agent_id)

            if not result.success:
                logger.error(f"Pipeline failed at task {i}: {task.name}")
                return result

            previous_result = result

        return previous_result

    async def execute_with_consensus(
        self,
        task: SubAgentTask,
        num_agents: int = 3,
        consensus_threshold: float = 0.66,
    ) -> TaskResult:
        """
        Execute task on multiple agents and require consensus.

        Args:
            task: Task to execute
            num_agents: Number of agents
            consensus_threshold: Fraction of agents that must agree (0.0-1.0)

        Returns:
            TaskResult with consensus result
        """
        logger.info(
            f"Executing task {task.name} with consensus "
            f"(threshold={consensus_threshold})"
        )

        # Get results from multiple agents
        results = await self.execute_parallel(
            [task] * num_agents,
            return_exceptions=True,
        )

        # Filter successful results
        successful = [r for r in results if isinstance(r, TaskResult) and r.success]

        if not successful:
            logger.error("All agents failed, no consensus possible")
            self.metrics.consensus_failed += 1
            return TaskResult(
                task_id=task.task_id,
                agent_id="consensus",
                status=SubAgentStatus.FAILED,
                error=Exception("All agents failed"),
            )

        # Group by result value
        result_groups = defaultdict(list)
        for result in successful:
            # Use string representation for grouping
            key = str(result.result)
            result_groups[key].append(result)

        # Find consensus
        required_votes = int(num_agents * consensus_threshold)
        consensus_result = None

        for key, group in result_groups.items():
            if len(group) >= required_votes:
                consensus_result = group[0]
                consensus_result.metadata['consensus_votes'] = len(group)
                consensus_result.metadata['total_agents'] = num_agents
                break

        if consensus_result:
            logger.info(
                f"Consensus achieved with {consensus_result.metadata['consensus_votes']} votes"
            )
            self.metrics.consensus_achieved += 1
        else:
            logger.warning("No consensus reached")
            self.metrics.consensus_failed += 1
            # Return most common result
            most_common_group = max(result_groups.values(), key=len)
            consensus_result = most_common_group[0]
            consensus_result.metadata['consensus_failed'] = True
            consensus_result.metadata['votes'] = len(most_common_group)

        return consensus_result

    def _topological_sort(self, tasks: List[SubAgentTask]) -> List[SubAgentTask]:
        """
        Topologically sort tasks based on dependencies.

        Args:
            tasks: Tasks to sort

        Returns:
            Sorted tasks
        """
        # Build dependency graph
        task_map = {t.task_id: t for t in tasks}
        in_degree = {t.task_id: 0 for t in tasks}
        graph = defaultdict(list)

        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in task_map:
                    graph[dep_id].append(task.task_id)
                    in_degree[task.task_id] += 1

        # Kahn's algorithm
        queue = [tid for tid, degree in in_degree.items() if degree == 0]
        sorted_ids = []

        while queue:
            current = queue.pop(0)
            sorted_ids.append(current)

            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(sorted_ids) != len(tasks):
            logger.warning("Circular dependencies detected, using original order")
            return tasks

        return [task_map[tid] for tid in sorted_ids]

    # ========================================================================
    # Communication Patterns
    # ========================================================================

    async def broadcast(
        self,
        sender_id: str,
        topic: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Broadcast message to all agents.

        Args:
            sender_id: Sender agent ID
            topic: Message topic
            content: Message content
            metadata: Additional metadata

        Returns:
            Number of agents message was sent to
        """
        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_ids=list(self.agents.keys()),
            topic=topic,
            content=content,
            pattern=CommunicationPattern.BROADCAST,
            metadata=metadata or {},
        )

        count = 0
        for agent_id in self.agents:
            if agent_id != sender_id:
                await self.message_queues[agent_id].put(message)
                count += 1

        self.metrics.messages_sent += count

        logger.debug(f"Broadcast from {sender_id}: {topic} to {count} agents")
        self._trigger_event('message_broadcast', message=message)

        # Store in Atomspace if enabled
        if self.atomspace and self.enable_atomspace_pubsub:
            await self._store_message_in_atomspace(message)

        return count

    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        topic: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send point-to-point message.

        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            topic: Message topic
            content: Message content
            metadata: Additional metadata

        Returns:
            True if sent successfully
        """
        if receiver_id not in self.agents:
            logger.warning(f"Receiver {receiver_id} not found")
            return False

        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_ids=[receiver_id],
            topic=topic,
            content=content,
            pattern=CommunicationPattern.POINT_TO_POINT,
            metadata=metadata or {},
        )

        await self.message_queues[receiver_id].put(message)
        self.metrics.messages_sent += 1

        logger.debug(f"Message from {sender_id} to {receiver_id}: {topic}")
        self._trigger_event('message_sent', message=message)

        if self.atomspace and self.enable_atomspace_pubsub:
            await self._store_message_in_atomspace(message)

        return True

    async def publish(
        self,
        sender_id: str,
        topic: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Publish message to topic subscribers.

        Args:
            sender_id: Publisher agent ID
            topic: Message topic
            content: Message content
            metadata: Additional metadata

        Returns:
            Number of subscribers message was sent to
        """
        subscribers = self.topic_subscribers.get(topic, set())

        if not subscribers:
            logger.debug(f"No subscribers for topic: {topic}")
            return 0

        message = Message(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            receiver_ids=list(subscribers),
            topic=topic,
            content=content,
            pattern=CommunicationPattern.PUBLISH_SUBSCRIBE,
            metadata=metadata or {},
        )

        count = 0
        for subscriber_id in subscribers:
            if subscriber_id in self.agents:
                await self.message_queues[subscriber_id].put(message)
                count += 1

        self.metrics.messages_sent += count

        logger.debug(
            f"Published to topic {topic} from {sender_id}: {count} subscribers"
        )
        self._trigger_event('message_published', message=message)

        if self.atomspace and self.enable_atomspace_pubsub:
            await self._store_message_in_atomspace(message)

        return count

    def subscribe(self, agent_id: str, topic: str) -> bool:
        """
        Subscribe agent to topic.

        Args:
            agent_id: Agent to subscribe
            topic: Topic to subscribe to

        Returns:
            True if subscribed
        """
        if agent_id not in self.agents:
            return False

        self.topic_subscribers[topic].add(agent_id)
        logger.debug(f"Agent {agent_id} subscribed to topic: {topic}")
        return True

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        """
        Unsubscribe agent from topic.

        Args:
            agent_id: Agent to unsubscribe
            topic: Topic to unsubscribe from

        Returns:
            True if unsubscribed
        """
        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(agent_id)
            logger.debug(f"Agent {agent_id} unsubscribed from topic: {topic}")
            return True
        return False

    async def receive_messages(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> List[Message]:
        """
        Receive all pending messages for an agent.

        Args:
            agent_id: Agent ID
            timeout: Timeout in seconds

        Returns:
            List of messages
        """
        if agent_id not in self.agents:
            return []

        messages = []
        queue = self.message_queues[agent_id]

        try:
            while True:
                if timeout is not None:
                    message = await asyncio.wait_for(queue.get(), timeout=timeout)
                else:
                    if queue.empty():
                        break
                    message = await queue.get()
                messages.append(message)
        except asyncio.TimeoutError:
            pass

        return messages

    async def request_reply(
        self,
        sender_id: str,
        receiver_id: str,
        topic: str,
        content: Any,
        timeout: float = 10.0,
    ) -> Optional[Any]:
        """
        Send request and wait for reply (synchronous RPC pattern).

        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID
            topic: Request topic
            content: Request content
            timeout: Reply timeout

        Returns:
            Reply content or None if timeout
        """
        request_id = str(uuid.uuid4())

        # Create future for reply
        reply_future = asyncio.Future()
        self.pending_requests[request_id] = reply_future

        # Send request
        await self.send_message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            topic=topic,
            content=content,
            metadata={'request_id': request_id, 'expects_reply': True},
        )

        try:
            # Wait for reply
            reply = await asyncio.wait_for(reply_future, timeout=timeout)
            return reply
        except asyncio.TimeoutError:
            logger.warning(f"Request {request_id} timed out")
            return None
        finally:
            del self.pending_requests[request_id]

    async def send_reply(
        self,
        sender_id: str,
        request_message: Message,
        content: Any,
    ) -> bool:
        """
        Send reply to a request.

        Args:
            sender_id: Reply sender
            request_message: Original request message
            content: Reply content

        Returns:
            True if sent successfully
        """
        request_id = request_message.metadata.get('request_id')
        if not request_id:
            logger.warning("Cannot reply to message without request_id")
            return False

        # If there's a pending future, resolve it
        if request_id in self.pending_requests:
            self.pending_requests[request_id].set_result(content)
            return True

        # Otherwise, send regular message
        return await self.send_message(
            sender_id=sender_id,
            receiver_id=request_message.sender_id,
            topic=f"reply_{request_message.topic}",
            content=content,
            metadata={'reply_to': request_id},
        )

    async def _store_message_in_atomspace(self, message: Message):
        """Store message in Atomspace for pub-sub persistence"""
        if not self.atomspace or not Atom:
            return

        try:
            atom = Atom(
                id=f"msg_{message.message_id}",
                type=AtomType.PERCEPTION,  # Using PERCEPTION for messages
                content={
                    'sender_id': message.sender_id,
                    'receiver_ids': message.receiver_ids,
                    'topic': message.topic,
                    'content': message.content,
                    'pattern': message.pattern.value,
                    'metadata': message.metadata,
                },
                timestamp=message.timestamp,
            )
            self.atomspace.add_atom(atom)
        except Exception as e:
            logger.error(f"Failed to store message in Atomspace: {e}")

    # ========================================================================
    # Main Coordination Loop
    # ========================================================================

    async def start(self):
        """Start the coordinator"""
        if self.running:
            logger.warning("Coordinator already running")
            return

        self.running = True
        logger.info("SubAgentCoordinator started")

        # Start worker loops
        worker_tasks = [
            asyncio.create_task(self._worker_loop())
            for _ in range(min(self.max_agents, len(self.agents)))
        ]

        try:
            await asyncio.gather(*worker_tasks)
        except asyncio.CancelledError:
            logger.info("Coordinator stopped")

    async def stop(self):
        """Stop the coordinator"""
        self.running = False
        logger.info("SubAgentCoordinator stopping...")

        # Wait for active tasks to complete
        while self.active_tasks:
            await asyncio.sleep(0.1)

        logger.info("SubAgentCoordinator stopped")

    async def _worker_loop(self):
        """Main worker loop that processes tasks"""
        while self.running:
            try:
                # Get task from queue (non-blocking with timeout)
                try:
                    priority, task = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Check dependencies
                if task.dependencies:
                    deps_ready = all(
                        dep_id in self.completed_tasks
                        for dep_id in task.dependencies
                    )
                    if not deps_ready:
                        # Re-queue task
                        await self.task_queue.put((priority, task))
                        await asyncio.sleep(0.5)
                        continue

                # Get best agent for task
                agent_id = self.get_best_agent_for_task(task)
                if not agent_id:
                    # No agent available, re-queue
                    await self.task_queue.put((priority, task))
                    await asyncio.sleep(0.5)
                    continue

                # Execute task (in background, don't block worker loop)
                asyncio.create_task(self._execute_with_retry(task, agent_id))

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _execute_with_retry(
        self,
        task: SubAgentTask,
        agent_id: str,
    ):
        """Execute task with retry logic"""
        for attempt in range(task.max_retries + 1):
            try:
                result = await self.execute_task(task, agent_id)

                if result.success:
                    return result

                # Task failed, retry if attempts remain
                if attempt < task.max_retries:
                    logger.info(
                        f"Retrying task {task.name} "
                        f"(attempt {attempt + 1}/{task.max_retries})"
                    )
                    result.status = SubAgentStatus.RETRYING
                    result.retry_count = attempt + 1
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"Task {task.name} failed after {task.max_retries} retries"
                    )
                    return result

            except Exception as e:
                logger.error(
                    f"Error executing task {task.name} (attempt {attempt + 1}): {e}",
                    exc_info=True,
                )

                if attempt >= task.max_retries:
                    # Create failed result
                    result = TaskResult(
                        task_id=task.task_id,
                        agent_id=agent_id,
                        status=SubAgentStatus.FAILED,
                        error=e,
                        error_traceback=traceback.format_exc(),
                        retry_count=attempt,
                    )
                    self.task_results[task.task_id] = result
                    return result

                await asyncio.sleep(2 ** attempt)

    # ========================================================================
    # Consciousness Integration
    # ========================================================================

    def set_consciousness_state(self, state: ConsciousnessState):
        """
        Update consciousness state and adjust coordination behavior.

        Args:
            state: New consciousness state
        """
        if not self.consciousness_integration or not ConsciousnessState:
            return

        old_state = self.current_consciousness_state
        self.current_consciousness_state = state

        logger.info(
            f"Consciousness state changed: "
            f"{old_state.value if old_state else 'None'} -> {state.value}"
        )

        # Adjust coordination based on state
        if state == ConsciousnessState.SLEEPING:
            # Consolidation mode - sequential processing
            self.default_strategy = CoordinationStrategy.SEQUENTIAL
        elif state == ConsciousnessState.EXPLORING:
            # Exploration mode - parallel processing
            self.default_strategy = CoordinationStrategy.PARALLEL
        elif state == ConsciousnessState.CONVERSING:
            # Interactive mode - competitive for best responses
            self.default_strategy = CoordinationStrategy.COMPETITIVE
        elif state == ConsciousnessState.IDLE:
            # Background processing
            self.default_strategy = CoordinationStrategy.PARALLEL

        self._trigger_event(
            'consciousness_state_changed',
            old_state=old_state,
            new_state=state,
        )

    # ========================================================================
    # Monitoring and Debugging
    # ========================================================================

    def get_metrics(self) -> CoordinationMetrics:
        """Get current coordination metrics"""
        self.metrics.active_agents = len([
            a for a in self.agents.values()
            if a.status != SubAgentStatus.IDLE
        ])
        self.metrics.idle_agents = len(self.agents) - self.metrics.active_agents

        # Calculate average execution time
        if self.metrics.completed_tasks > 0:
            total_time = sum(
                r.execution_time
                for r in self.task_results.values()
                if r.execution_time
            )
            self.metrics.average_execution_time = (
                total_time / self.metrics.completed_tasks
            )

        return self.metrics

    def get_status(self) -> Dict[str, Any]:
        """Get detailed coordinator status"""
        return {
            'running': self.running,
            'agents': {
                agent_id: {
                    'name': agent.name,
                    'status': agent.status.value,
                    'current_task': agent.current_task,
                    'tasks_completed': agent.tasks_completed,
                    'tasks_failed': agent.tasks_failed,
                    'success_rate': agent.success_rate,
                    'average_execution_time': agent.average_execution_time,
                }
                for agent_id, agent in self.agents.items()
            },
            'active_tasks': len(self.active_tasks),
            'pending_tasks': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'metrics': {
                'total_tasks': self.metrics.total_tasks,
                'completed_tasks': self.metrics.completed_tasks,
                'failed_tasks': self.metrics.failed_tasks,
                'cancelled_tasks': self.metrics.cancelled_tasks,
                'average_execution_time': self.metrics.average_execution_time,
                'messages_sent': self.metrics.messages_sent,
                'consensus_achieved': self.metrics.consensus_achieved,
                'consensus_failed': self.metrics.consensus_failed,
            },
            'consciousness_state': (
                self.current_consciousness_state.value
                if self.current_consciousness_state
                else None
            ),
            'default_strategy': self.default_strategy.value,
        }

    def on(self, event: str, handler: Callable):
        """
        Register event handler.

        Args:
            event: Event name
            handler: Handler function
        """
        self.event_handlers[event].append(handler)

    def off(self, event: str, handler: Callable):
        """
        Unregister event handler.

        Args:
            event: Event name
            handler: Handler function
        """
        if event in self.event_handlers:
            self.event_handlers[event].remove(handler)

    def _trigger_event(self, event: str, **kwargs):
        """Trigger event handlers"""
        for handler in self.event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(**kwargs))
                else:
                    handler(**kwargs)
            except Exception as e:
                logger.error(f"Error in event handler for {event}: {e}")

    def debug_info(self) -> str:
        """Get formatted debug information"""
        status = self.get_status()
        metrics = self.get_metrics()

        info = [
            "=" * 60,
            "SubAgentCoordinator Debug Info",
            "=" * 60,
            f"Status: {'Running' if status['running'] else 'Stopped'}",
            f"Strategy: {status['default_strategy']}",
            f"Consciousness State: {status['consciousness_state']}",
            "",
            "Agents:",
            f"  Total: {len(self.agents)}",
            f"  Active: {metrics.active_agents}",
            f"  Idle: {metrics.idle_agents}",
            "",
            "Tasks:",
            f"  Total Submitted: {metrics.total_tasks}",
            f"  Active: {status['active_tasks']}",
            f"  Pending: {status['pending_tasks']}",
            f"  Completed: {metrics.completed_tasks}",
            f"  Failed: {metrics.failed_tasks}",
            f"  Cancelled: {metrics.cancelled_tasks}",
            "",
            "Performance:",
            f"  Avg Execution Time: {metrics.average_execution_time:.3f}s",
            f"  Messages Sent: {metrics.messages_sent}",
            f"  Consensus Achieved: {metrics.consensus_achieved}",
            f"  Consensus Failed: {metrics.consensus_failed}",
            "",
            "Agent Details:",
        ]

        for agent_id, agent_status in status['agents'].items():
            info.append(
                f"  {agent_status['name']} ({agent_id}): "
                f"{agent_status['status']} - "
                f"{agent_status['tasks_completed']} completed, "
                f"{agent_status['tasks_failed']} failed, "
                f"success rate: {agent_status['success_rate']:.1%}"
            )

        info.append("=" * 60)

        return "\n".join(info)
