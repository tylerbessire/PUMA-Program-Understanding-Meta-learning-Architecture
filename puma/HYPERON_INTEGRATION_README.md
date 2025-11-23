# Hyperon-PUMA Integration

Complete integration of OpenCog Hyperon's MeTTa reasoning engine with PUMA's cognitive architecture.

## Overview

This integration brings symbolic reasoning, parallel distributed processing, and advanced pattern matching to PUMA through Hyperon's MeTTa language and subagent architecture.

### Key Components

1. **HyperonPUMAIntegration** (`puma/hyperon_integration.py`)
   - Main integration class coordinating all Hyperon components
   - Provides high-level workflow methods
   - Manages subagent lifecycle and resources
   - Integrates with consciousness states and memory

2. **Bootstrap Integration** (`bootstrap/bootstrap.py`)
   - Updated to initialize Hyperon components during consciousness bootstrap
   - Maintains backward compatibility (Hyperon is optional)
   - Configurable through `HyperonConfig`

3. **Subagent Systems** (`puma/hyperon_subagents/`)
   - SubAgentManager: Pool management for parallel reasoning
   - SubAgentCoordinator: Task coordination and communication
   - RFTHyperonBridge: RFT ↔ MeTTa conversion
   - MeTTaExecutionEngine: Core MeTTa execution

## Installation

```bash
# Install Hyperon (optional)
pip install hyperon

# PUMA will work without Hyperon, but integration features will be disabled
```

## Quick Start

### Basic Usage

```python
from bootstrap.bootstrap import bootstrap_new_consciousness
from puma.hyperon_integration import HyperonConfig
from pathlib import Path

# Bootstrap PUMA with Hyperon integration
consciousness = bootstrap_new_consciousness(
    atomspace_path=Path("./atomspace-db/default"),
    enable_hyperon=True,
    hyperon_config=HyperonConfig(
        max_agents=10,
        create_specialized_pool=True
    )
)

# Access Hyperon integration
integration = consciousness.hyperon_integration

# Initialize (async)
await integration.initialize()

# Get status
status = integration.get_status()
print(f"Subagents: {status['num_subagents']}")
print(f"RFT Bridge: {status['rft_bridge_enabled']}")
```

### Using the Integration

```python
# Solve ARC task
arc_task = {
    "train": [...],
    "test": [...]
}
result = await integration.solve_arc_task(arc_task)

# RFT reasoning
from puma.rft.reasoning import RelationType

frames = await integration.reason_with_rft(
    source="cat",
    target="dog",
    relation_type=RelationType.COORDINATION,
    use_subagents=True
)

# Frequency analysis
pattern_data = {...}
signature = await integration.analyze_frequencies(
    pattern_data=pattern_data,
    use_metta_inference=True
)

# Consciousness-aware task execution
from puma.hyperon_subagents import SubAgentTask
from puma.consciousness.state_machine import ConsciousnessState

task = SubAgentTask(
    task_type="reasoning",
    metta_program="(infer (premise) (rule))",
    priority=0.8
)

result = await integration.coordinate_consciousness_aware_task(
    task=task,
    required_state=ConsciousnessState.EXPLORING
)
```

## Configuration

### HyperonConfig Options

```python
from puma.hyperon_integration import HyperonConfig
from puma.hyperon_subagents import CoordinationStrategy, CommunicationPattern

config = HyperonConfig(
    # Subagent pool
    max_agents=10,
    create_specialized_pool=True,
    default_timeout=30.0,

    # Coordination
    default_coordination_strategy=CoordinationStrategy.PARALLEL,
    default_communication_pattern=CommunicationPattern.SHARED_MEMORY,

    # Performance
    enable_metrics=True,
    enable_caching=True,
    cache_size=1000,

    # Integration
    integrate_with_consciousness=True,
    integrate_with_memory=True,
    enable_frequency_ledger=True
)
```

## Example Workflows

Three comprehensive example workflows are provided in `examples/hyperon_integration_workflows.py`:

### 1. ARC Task Solving

Demonstrates distributed reasoning for visual pattern problems:

```bash
python examples/hyperon_integration_workflows.py --workflow arc
```

Features:
- Pattern frequency analysis
- Parallel task distribution
- Solution synthesis from multiple agents
- Reasoning trace visualization

### 2. RFT Reasoning

Shows relational frame theory reasoning with MeTTa:

```bash
python examples/hyperon_integration_workflows.py --workflow rft
```

Features:
- Frame to MeTTa conversion
- Distributed relational inference
- Relation composition
- Multiple relation types (coordination, opposition, hierarchy, etc.)

### 3. Frequency Analysis

Demonstrates pattern frequency analysis with symbolic inference:

```bash
python examples/hyperon_integration_workflows.py --workflow frequency
```

Features:
- Symbolic pattern matching
- Frequency signature generation
- Pattern correlation analysis
- MeTTa-based pattern extraction

### Run All Workflows

```bash
python examples/hyperon_integration_workflows.py
```

## Architecture

### Integration Points

```
PUMA Consciousness
├── Atomspace ←→ Hyperon Grounding Space
├── RFT Engine ←→ RFTHyperonBridge ←→ MeTTa
├── Memory System ←→ SubAgentCoordinator
├── Consciousness States ←→ Task Routing
└── Frequency Ledger ←→ MeTTa Inference
```

### Subagent Capabilities

Each subagent can have specialized capabilities:

- **REASONING**: Forward/backward chaining, inference
- **PATTERN_MATCHING**: Pattern recognition and extraction
- **MEMORY_RETRIEVAL**: Atomspace query and retrieval
- **GOAL_PLANNING**: Goal decomposition and planning
- **RELATIONAL_FRAMING**: RFT relation inference
- **ABSTRACTION**: Concept abstraction and generalization
- **ANALOGY_MAKING**: Analogical reasoning
- **CONCEPT_SYNTHESIS**: Concept combination and synthesis

### Communication Patterns

- **BROADCAST**: One-to-all communication
- **POINT_TO_POINT**: Direct agent-to-agent
- **PUBLISH_SUBSCRIBE**: Topic-based messaging
- **REQUEST_REPLY**: Synchronous request-response
- **SHARED_MEMORY**: Communication via Atomspace

### Coordination Strategies

- **PARALLEL**: Execute all tasks concurrently
- **SEQUENTIAL**: Execute with dependencies
- **COMPETITIVE**: Multiple agents, best wins
- **PIPELINE**: Sequential with output passing
- **HIERARCHICAL**: Tree-based delegation
- **CONSENSUS**: Require agreement from multiple agents

## API Reference

### HyperonPUMAIntegration

Main integration class:

```python
class HyperonPUMAIntegration:
    async def initialize() -> None
    async def solve_arc_task(task_data, **kwargs) -> Dict
    async def reason_with_rft(source, target, **kwargs) -> List[RelationalFrame]
    async def analyze_frequencies(pattern_data, **kwargs) -> FrequencySignature
    async def coordinate_consciousness_aware_task(task, **kwargs) -> SubAgentResult
    def get_status() -> Dict
    async def shutdown() -> None
```

### Convenience Functions

```python
async def create_integration(**kwargs) -> HyperonPUMAIntegration
    """Create and initialize integration in one call"""
```

## Performance Considerations

### Optimal Configuration

For best performance:

1. **Agent Pool Size**: 5-10 agents for most tasks
2. **Caching**: Enable for repeated pattern matching
3. **Communication**: Use SHARED_MEMORY with Atomspace
4. **Coordination**: PARALLEL for independent tasks

### Monitoring

```python
# Get pool status
status = integration.subagent_manager.get_pool_status()
print(f"Completed tasks: {status['completed_tasks']}")
print(f"Success rate: {status['average_success_rate']:.2%}")

# Get agent metrics
metrics = integration.subagent_manager.get_agent_metrics()
for agent in metrics:
    print(f"{agent['name']}: {agent['execution_count']} executions")
```

## Backward Compatibility

The integration is fully backward compatible:

- **Hyperon optional**: PUMA works without Hyperon installed
- **Graceful degradation**: Features disable if Hyperon unavailable
- **Existing code unchanged**: No changes needed to existing PUMA code
- **Optional initialization**: Set `enable_hyperon=False` to disable

## Troubleshooting

### Hyperon Not Available

If you see "Hyperon not available" messages:

```bash
# Install Hyperon
pip install hyperon

# Or disable Hyperon integration
consciousness = bootstrap_new_consciousness(enable_hyperon=False)
```

### Import Errors

If you encounter import errors:

```python
# Check Hyperon availability
from puma.hyperon_subagents import HYPERON_AVAILABLE
print(f"Hyperon available: {HYPERON_AVAILABLE}")

# Check integration status
status = integration.get_status()
print(status)
```

### Performance Issues

If subagents are slow:

1. Reduce `max_agents` in config
2. Enable caching
3. Use PARALLEL coordination for independent tasks
4. Check agent metrics to identify bottlenecks

## Development

### Adding New Workflows

Create new workflows in `examples/`:

```python
async def my_custom_workflow():
    consciousness = bootstrap_new_consciousness(enable_hyperon=True)
    integration = consciousness.hyperon_integration
    await integration.initialize()

    # Your workflow logic here

    await integration.shutdown()
```

### Extending Capabilities

Add new agent capabilities in `puma/hyperon_subagents/manager.py`:

```python
class AgentCapability(Enum):
    MY_NEW_CAPABILITY = "my_new_capability"
```

### Custom Coordination Strategies

Implement custom strategies in `puma/hyperon_subagents/coordinator.py`.

## Files Created/Modified

### Created Files

1. `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/hyperon_integration.py`
   - Main integration module (800+ lines)
   - HyperonPUMAIntegration class
   - Workflow methods and utilities

2. `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/examples/hyperon_integration_workflows.py`
   - Example workflows (600+ lines)
   - ARC task solving demo
   - RFT reasoning demo
   - Frequency analysis demo
   - Comprehensive integration demo

3. `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/puma/HYPERON_INTEGRATION_README.md`
   - This documentation file

### Modified Files

1. `/home/user/PUMA-Program-Understanding-Meta-learning-Architecture/bootstrap/bootstrap.py`
   - Added Hyperon integration initialization
   - Added `enable_hyperon` and `hyperon_config` parameters
   - Updated Consciousness class with `hyperon_integration` attribute
   - Maintains full backward compatibility

## Further Reading

- Hyperon documentation: https://github.com/trueagi-io/hyperon-experimental
- MeTTa language guide: https://github.com/trueagi-io/metta-lang
- PUMA architecture: See main README.md
- Subagent system: See `puma/hyperon_subagents/README_MANAGER.md`
- RFT integration: See `puma/rft/README.md`

## Support

For issues or questions:

1. Check this README
2. Review example workflows
3. Check existing Hyperon subagent demos
4. Review integration status with `get_status()`

## License

Same as PUMA project license.
