# Hyperon SubAgent Manager - Quick Reference

## Import Statement
```python
from puma.hyperon_subagents import (
    SubAgentManager,
    HyperonSubAgent,
    SubAgentTask,
    SubAgentResult,
    SubAgentState,
    AgentCapability,
    HYPERON_AVAILABLE
)
```

## Quick Start

### 1. Create Manager
```python
manager = SubAgentManager(max_agents=10)
manager.create_specialized_agents()
```

### 2. Create Task
```python
task = SubAgentTask(
    task_type="reasoning",
    metta_program="(infer (premise A) (rule implies))",
    priority=0.8
)
```

### 3. Execute Task
```python
result = await manager.execute_task(task, AgentCapability.REASONING)
```

## Key Classes

### SubAgentState
- `IDLE` - Ready for tasks
- `RUNNING` - Executing
- `WAITING` - Blocked
- `COMPLETED` - Done
- `FAILED` - Error
- `SUSPENDED` - Paused

### AgentCapability
- `REASONING` - Logic inference
- `PATTERN_MATCHING` - Pattern discovery
- `MEMORY_RETRIEVAL` - Memory queries
- `GOAL_PLANNING` - Goal decomposition
- `RELATIONAL_FRAMING` - RFT reasoning
- `ABSTRACTION` - Concept formation
- `ANALOGY_MAKING` - Analogies
- `CONCEPT_SYNTHESIS` - Creative synthesis

## Common Patterns

### Parallel Execution
```python
tasks = [SubAgentTask(...) for _ in range(5)]
results = await manager.execute_parallel(tasks)
```

### Map-Reduce
```python
result = await manager.map_reduce_reasoning(
    map_programs=["(p1)", "(p2)"],
    reduce_program="(combine $x)"
)
```

### Communication
```python
manager.broadcast_message({'type': 'update', 'data': 'new'})
manager.send_message(agent_id, {'type': 'hint'})
messages = manager.get_messages(agent_id)
```

### Monitoring
```python
status = manager.get_pool_status()
metrics = manager.get_agent_metrics()
```

## File Locations

- **Core**: `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/manager.py`
- **Init**: `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/__init__.py`
- **Docs**: `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_subagents/README_MANAGER.md`
- **Demo**: `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/examples/hyperon_subagents_demo.py`
