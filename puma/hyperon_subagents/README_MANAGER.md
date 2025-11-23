# Hyperon SubAgent Manager

Comprehensive multi-agent Hyperon MeTTa system for parallel reasoning, pattern matching, memory retrieval, and goal planning within PUMA's cognitive architecture.

## Overview

The SubAgent Manager provides a scalable framework for coordinating multiple Hyperon MeTTa interpreter instances, enabling distributed reasoning and emergent collective intelligence through:

- **Parallel Execution**: Execute multiple MeTTa programs simultaneously across agent pool
- **Capability-Based Routing**: Route tasks to agents based on specialized capabilities
- **Inter-Agent Communication**: Message passing and shared Atomspace for coordination
- **State Management**: Track agent lifecycle (IDLE, RUNNING, WAITING, COMPLETED, FAILED)
- **Integration**: Seamless integration with PUMA's consciousness, memory, and goal systems

## Architecture

### Components

1. **HyperonSubAgent**
   - Individual MeTTa interpreter instance
   - Specialized capabilities (reasoning, pattern matching, etc.)
   - Independent task execution with state management
   - Performance metrics tracking

2. **SubAgentManager**
   - Coordinates pool of subagents
   - Task queue and scheduling
   - Load balancing and resource management
   - Message bus for inter-agent communication
   - Integration with PUMA's cognitive systems

### Agent States

```python
class SubAgentState(Enum):
    IDLE = "idle"          # Available for new tasks
    RUNNING = "running"    # Currently executing a task
    WAITING = "waiting"    # Waiting for resources/dependencies
    COMPLETED = "completed" # Task execution completed
    FAILED = "failed"      # Task execution failed
    SUSPENDED = "suspended" # Temporarily suspended
```

### Agent Capabilities

```python
class AgentCapability(Enum):
    REASONING = "reasoning"                    # Forward/backward chaining
    PATTERN_MATCHING = "pattern_matching"      # Pattern discovery and matching
    MEMORY_RETRIEVAL = "memory_retrieval"      # Episodic memory queries
    GOAL_PLANNING = "goal_planning"           # Goal decomposition and planning
    RELATIONAL_FRAMING = "relational_framing" # RFT-based relational reasoning
    ABSTRACTION = "abstraction"               # Abstract concept formation
    ANALOGY_MAKING = "analogy_making"         # Analogical reasoning
    CONCEPT_SYNTHESIS = "concept_synthesis"   # Creative concept combination
```

## Usage Examples

### Basic Setup

```python
from puma.hyperon_subagents import (
    SubAgentManager,
    HyperonSubAgent,
    SubAgentTask,
    AgentCapability,
    SubAgentState
)
from atomspace_db.core import bootstrap_atomspace

# Initialize shared atomspace
atomspace = bootstrap_atomspace()

# Create manager with up to 10 agents
manager = SubAgentManager(
    atomspace=atomspace,
    max_agents=10
)

# Create specialized agent pool
manager.create_specialized_agents()

# Check pool status
status = manager.get_pool_status()
print(f"Total agents: {status['total_agents']}")
print(f"Capability distribution: {status['capability_distribution']}")
```

### Creating Custom Agents

```python
# Create a custom reasoning agent
reasoning_agent = manager.create_agent(
    capabilities={
        AgentCapability.REASONING,
        AgentCapability.RELATIONAL_FRAMING
    },
    name="CustomReasoner"
)

# Create a multi-capability agent
generalist = manager.create_agent(
    capabilities={
        AgentCapability.REASONING,
        AgentCapability.PATTERN_MATCHING,
        AgentCapability.ANALOGY_MAKING
    },
    name="Generalist"
)
```

### Task Execution

#### Single Task Execution

```python
# Create a reasoning task
task = SubAgentTask(
    task_type="reasoning",
    metta_program="""
    ; Forward chaining inference
    (= (premise) A)
    (= (rule) (implies A B))
    (infer (premise) (rule))
    """,
    context={'domain': 'logic'},
    priority=0.8
)

# Execute on any available reasoning agent
result = await manager.execute_task(
    task,
    required_capability=AgentCapability.REASONING
)

if result.success:
    print(f"Result atoms: {result.output_atoms}")
    print(f"Execution time: {result.execution_time}s")
else:
    print(f"Error: {result.error}")
```

#### Parallel Task Execution

```python
# Create multiple pattern matching tasks
tasks = []
for pattern in ["(shape square)", "(color red)", "(size large)"]:
    task = SubAgentTask(
        task_type="pattern_matching",
        metta_program=f"(find-pattern {pattern})",
        priority=0.7
    )
    tasks.append(task)

# Execute all tasks in parallel
results = await manager.execute_parallel(tasks)

# Process results
successful_results = [r for r in results if r.success]
print(f"Completed {len(successful_results)}/{len(tasks)} tasks")
```

### Map-Reduce Reasoning

```python
# Define map programs (execute in parallel)
map_programs = [
    "(match &self (pattern1 $x) $x)",
    "(match &self (pattern2 $y) $y)",
    "(match &self (pattern3 $z) $z)",
]

# Define reduce program (combine results)
reduce_program = """
(= (combine-results $results)
   (synthesize-concept $results))
"""

# Execute map-reduce
result = await manager.map_reduce_reasoning(
    map_programs,
    reduce_program,
    context={'operation': 'pattern_synthesis'}
)

print(f"Combined result: {result.output_atoms}")
```

### Inter-Agent Communication

```python
# Broadcast message to all agents
manager.broadcast_message(
    message={'type': 'update', 'data': 'new_knowledge'},
    sender_id='control_system'
)

# Send message to specific agent
manager.send_message(
    recipient_id=reasoning_agent.id,
    message={'type': 'task_hint', 'hint': 'try_backward_chaining'},
    sender_id='planner'
)

# Agent receives messages
messages = manager.get_messages(reasoning_agent.id, clear=True)
for msg in messages:
    print(f"From {msg['sender']}: {msg['message']}")
```

### Memory Retrieval Tasks

```python
# Create memory retrieval task
memory_task = SubAgentTask(
    task_type="memory_retrieval",
    metta_program="""
    ; Retrieve episodes from last hour
    (temporal-query
        (- (current-time) 3600)
        (current-time))
    """,
    context={'query_type': 'temporal'},
    priority=0.9
)

result = await manager.execute_task(
    memory_task,
    required_capability=AgentCapability.MEMORY_RETRIEVAL
)

episodes = result.output_atoms
print(f"Retrieved {len(episodes)} episodic memories")
```

### Goal Planning Tasks

```python
# Create goal planning task
planning_task = SubAgentTask(
    task_type="goal_planning",
    metta_program="""
    ; Decompose high-level goal into subgoals
    (= (main-goal) (learn-about quantum-computing))
    (decompose-goal (main-goal))
    """,
    context={'planning_horizon': 7},  # days
    priority=0.85
)

result = await manager.execute_task(
    planning_task,
    required_capability=AgentCapability.GOAL_PLANNING
)

subgoals = result.output_atoms
print(f"Goal decomposed into {len(subgoals)} subgoals")
```

### Performance Monitoring

```python
# Get pool status
status = manager.get_pool_status()
print(f"Average success rate: {status['average_success_rate']:.2%}")
print(f"Pending tasks: {status['pending_tasks']}")
print(f"Completed tasks: {status['completed_tasks']}")

# Get individual agent metrics
metrics = manager.get_agent_metrics()
for agent_metrics in metrics:
    print(f"\nAgent: {agent_metrics['name']}")
    print(f"  Executions: {agent_metrics['execution_count']}")
    print(f"  Success rate: {agent_metrics['success_rate']:.2%}")
    print(f"  Avg execution time: {agent_metrics['average_execution_time']:.3f}s")
```

### Finding Agents

```python
# Find any idle agent with reasoning capability
agent = manager.find_capable_agent(
    required_capability=AgentCapability.REASONING,
    prefer_idle=True
)

# Find all agents with pattern matching capability
pattern_matchers = manager.find_agents_with_capability(
    AgentCapability.PATTERN_MATCHING
)
print(f"Found {len(pattern_matchers)} pattern matching agents")
```

## Integration with PUMA Consciousness System

```python
from puma.consciousness.state_machine import ConsciousnessStateMachine
from puma.memory.episodic import EpisodicMemorySystem
from puma.goals.formation import GoalFormationSystem

# Create PUMA systems
memory_system = EpisodicMemorySystem(atomspace=atomspace)
goal_system = GoalFormationSystem()
consciousness = ConsciousnessStateMachine(
    memory_system=memory_system,
    goal_system=goal_system
)

# Create integrated manager
manager = SubAgentManager(
    atomspace=atomspace,
    consciousness_state_machine=consciousness,
    memory_system=memory_system,
    goal_system=goal_system,
    max_agents=10
)

# Manager automatically records task execution in memory system
# and can interact with consciousness states
```

## Relational Frame Theory (RFT) Integration

The manager integrates PUMA's RFT-based cognitive architecture:

```python
# RFT-based relational reasoning task
rft_task = SubAgentTask(
    task_type="relational_framing",
    metta_program="""
    ; Derive new relations through relational frames
    (= (trained-frame) (is-bigger-than A B))
    (= (trained-frame) (is-bigger-than B C))

    ; Derive transitive relation (A is bigger than C)
    (derive-relation A C is-bigger-than)
    """,
    context={'frame_type': 'comparative'},
    priority=0.8
)

result = await manager.execute_task(
    rft_task,
    required_capability=AgentCapability.RELATIONAL_FRAMING
)
```

## Advanced Features

### Custom Capability Initialization

```python
# Add capability to existing agent
agent = manager.agents[some_agent_id]
agent.add_capability(AgentCapability.CONCEPT_SYNTHESIS)

# Agent automatically initializes MeTTa programs for new capability
```

### Graceful Degradation

The system gracefully handles missing Hyperon installation:

```python
from puma.hyperon_subagents import HYPERON_AVAILABLE

if HYPERON_AVAILABLE:
    print("Hyperon available - using full MeTTa reasoning")
else:
    print("Hyperon not available - using simulation mode")
    # System still works with simulated results for testing
```

### Shutdown

```python
# Gracefully shutdown all agents and thread pool
manager.shutdown()
```

## Files Created

- `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/manager.py`
  - Core implementation of HyperonSubAgent and SubAgentManager classes

- `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/__init__.py`
  - Module initialization and exports

## Type Hints and Documentation

All classes and methods include:
- Comprehensive docstrings
- Full type hints for parameters and return values
- Usage examples in docstrings
- Detailed error descriptions

## Thread Safety

- All shared state access is protected by locks
- Thread-safe agent pool management
- Thread-safe message bus operations
- Safe parallel task execution

## Performance Characteristics

- **Parallel Execution**: O(n/m) where n=tasks, m=agents
- **Agent Lookup**: O(a) where a=number of agents
- **Memory**: O(a + t) where a=agents, t=tasks in history
- **Thread Pool**: Configurable max workers (default: max_agents)

## Future Enhancements

Potential extensions:
- Dynamic agent creation based on workload
- Agent specialization through learning
- Advanced load balancing strategies
- Distributed execution across network
- Agent persistence and recovery
- Priority-based task scheduling
- Task dependency resolution
- Hierarchical agent organization
