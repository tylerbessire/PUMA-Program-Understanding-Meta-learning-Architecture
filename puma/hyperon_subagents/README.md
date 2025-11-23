# Hyperon Subagents - MeTTa Execution Engine

## Overview

The MeTTa Execution Engine provides symbolic reasoning capabilities for PUMA's cognitive architecture through integration with OpenCog Hyperon's MeTTa language. This module bridges PUMA's Relational Frame Theory (RFT) system with symbolic program execution, enabling:

- **Symbolic reasoning** over relational frames and patterns
- **Pattern matching** for ARC-AGI grid analysis
- **Knowledge representation** in Atomspace
- **DSL-to-MeTTa compilation** for PUMA operations
- **Multi-modal execution** (interactive, batch, async)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PUMA Cognitive Layer                    │
│    (RFT Frames, Entities, Context, Goals)               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             MeTTa Execution Engine                       │
│  • DSL Compiler                                         │
│  • RFT-to-MeTTa Translator                             │
│  • Execution Modes (Interactive/Batch/Async)           │
│  • Atomspace Integration                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Hyperon/MeTTa Runtime                       │
│    (Atomspace, Pattern Matching, Inference)             │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Ensure Hyperon is installed
pip install hyperon>=0.3.0

# The module is part of PUMA
# No additional installation needed if PUMA is already set up
```

## Key Components

### MeTTaExecutionEngine

The main execution engine class providing comprehensive MeTTa program execution.

**Key Features:**
- Execute MeTTa programs in multiple modes
- Load MeTTa files
- Register custom atoms
- Query Atomspace with patterns
- Convert PUMA DSL to MeTTa
- Integrate with RFT system

### Execution Modes

1. **BATCH** - Execute entire program at once (fastest)
2. **INTERACTIVE** - Step-by-step execution with inspection
3. **ASYNC** - Asynchronous execution with callbacks

### RFT Integration

Seamlessly convert PUMA's relational frames to MeTTa expressions:

- **RelationalFrame → MeTTa**: Convert coordination, hierarchy, causal, and other frame types
- **Context → MeTTa KB**: Transform RFT context into queryable knowledge base
- **Entity → MeTTa Atom**: Represent PUMA entities as MeTTa atoms

## Usage Examples

### Basic Execution

```python
from puma.hyperon_subagents import MeTTaExecutionEngine, ExecutionMode

# Initialize engine
engine = MeTTaExecutionEngine(execution_mode=ExecutionMode.BATCH)

# Execute simple program
result = engine.execute_program("(+ 2 3)")
print(result.results)  # [5]

# Execute pattern matching
program = """
(cell 0 0 blue)
(cell 1 0 red)
!(match &self (cell ?x ?y blue) $result)
"""
result = engine.execute_program(program)
```

### RFT to MeTTa Conversion

```python
from puma.rft import RelationalFrame, RelationType
from puma.hyperon_subagents import MeTTaExecutionEngine

engine = MeTTaExecutionEngine()

# Create relational frame
frame = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="square",
    target="rectangle",
    strength=0.8
)

# Convert to MeTTa
metta_expr = engine.rft_to_metta(frame)
# Returns: "(RelFrame coordination square rectangle 0.8)"

# Execute it
result = engine.execute_program(metta_expr)
```

### DSL Compilation

```python
# Define PUMA DSL operation
dsl_operation = {
    "operation": "pattern_match",
    "params": {
        "pattern": "(cell ?x ?y blue)",
        "target": "(cell 0 0 blue)"
    }
}

# Compile to MeTTa
metta_code = engine.compile_dsl_to_metta(dsl_operation)
# Returns: "!(match &self (cell ?x ?y blue) (cell 0 0 blue))"

# Execute compiled code
result = engine.execute_program(metta_code)
```

### Context to Knowledge Base

```python
from puma.rft import Context, Limits

# Create RFT context
context = Context(
    state={"grid_size": (3, 3), "pattern_count": 5},
    history=[],
    constraints={"max_steps": 100},
    goal_test=lambda s: s.get("pattern_count", 0) >= 5,
    limits=Limits(pliance_steps=50, tracking_budget=20, thresh=0.7, outer_budget=10)
)

# Convert to MeTTa knowledge base
metta_kb = engine.context_to_metta(context)

# Execute knowledge base
result = engine.execute_program(metta_kb)
```

### Load MeTTa Files

```python
# Load and execute .metta file
result = engine.load_metta_file("programs/reasoning.metta")
print(f"Execution time: {result.execution_time}s")
print(f"Results: {result.results}")
```

### Register Custom Atoms

```python
# Register different types of atoms
engine.register_atom("learning_rate", 0.001)
engine.register_atom("model_name", "puma_transformer")
engine.register_atom("config", {
    "layers": 12,
    "hidden_size": 768
})
```

### Query Atomspace

```python
# Query with pattern
results = engine.query_atomspace(
    "(RelFrame coordination ?source ?target ?strength)"
)

for result in results:
    print(f"Found: {result}")
```

## Sample Programs

The module includes comprehensive sample programs demonstrating:

1. **Pattern Matching** - Grid cell analysis
2. **Relational Reasoning** - RFT frame queries
3. **Frequency Analysis** - PUMA's core innovation
4. **Transformations** - Pattern-based rewriting
5. **Causal Reasoning** - Temporal and causal chains
6. **Derivational Reasoning** - Transitive relations
7. **Comparative Reasoning** - Magnitude comparisons
8. **Spatial Reasoning** - Location and proximity
9. **Episodic Memory** - Experience tracking
10. **Goal-Directed Reasoning** - Planning and intentions
11. **Meta-Learning** - Learning to learn
12. **Self-Modification** - Code introspection

See `sample_programs.metta` for full examples.

## PUMA DSL Operations

Supported DSL operations for compilation:

| Operation | Description | Example |
|-----------|-------------|---------|
| `pattern_match` | Match patterns in atomspace | Find all blue cells |
| `transform` | Pattern-based rewriting | Convert blue → red |
| `frequency_analysis` | Group by frequency (core PUMA) | Cluster by occurrence count |
| `relational_query` | Query relational frames | Find coordination frames |
| `custom` | Direct MeTTa code | Any valid MeTTa |

## API Reference

### MeTTaExecutionEngine

#### Constructor

```python
MeTTaExecutionEngine(
    atomspace: Optional[GroundingSpace] = None,
    execution_mode: ExecutionMode = ExecutionMode.BATCH,
    enable_logging: bool = True
)
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `execute_program(code, mode, timeout)` | Execute MeTTa program | ExecutionResult |
| `load_metta_file(filepath)` | Load and execute .metta file | ExecutionResult |
| `register_atom(name, value, type)` | Register custom atom | HyperonAtom |
| `query_atomspace(pattern)` | Query with pattern | List[Dict] |
| `compile_dsl_to_metta(dsl_op)` | Compile DSL to MeTTa | str |
| `rft_to_metta(frame)` | Convert RFT frame | str |
| `context_to_metta(context)` | Convert RFT context | str |
| `entity_to_metta(entity)` | Convert PUMA entity | str |
| `get_sample_programs()` | Get example programs | Dict[str, str] |
| `get_statistics()` | Get execution stats | Dict |
| `reset()` | Reset engine state | None |

### ExecutionResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Execution succeeded |
| `results` | List[Any] | Execution results |
| `execution_time` | float | Time in seconds |
| `mode` | ExecutionMode | Execution mode used |
| `error` | Optional[str] | Error message if failed |
| `metadata` | Dict | Additional metadata |
| `timestamp` | datetime | Execution timestamp |

## Integration with PUMA Systems

### Frequency Ledger System

The engine supports PUMA's core innovation - frequency-based analysis:

```metta
; Group objects by frequency attribute
!(group-by-frequency
    (object obj1 (frequency 3))
    (object obj2 (frequency 1))
    (object obj3 (frequency 3)))
```

### RFT Reasoning

All RFT relation types are supported:

- **Coordination** - Similarity (X is like Y)
- **Opposition** - Difference (X is opposite of Y)
- **Hierarchy** - Categorization (X is a type of Y)
- **Temporal** - Sequence (X before Y)
- **Causal** - If-then (X causes Y)
- **Comparative** - Magnitude (X > Y)
- **Spatial** - Location (X near Y)

### ARC-AGI Integration

Designed for ARC-AGI puzzle solving:

```python
# Analyze grid patterns
result = engine.execute_program("""
(cell 0 0 blue)
(cell 1 0 blue)
(cell 2 0 red)
!(match &self (cell ?x ?y blue) $result)
""")

# Apply transformations
transform_dsl = {
    "operation": "transform",
    "params": {
        "input_pattern": "(cell ?x ?y blue)",
        "output_pattern": "(cell ?x ?y red)",
        "target": "$grid"
    }
}
metta_code = engine.compile_dsl_to_metta(transform_dsl)
```

## Performance

- **Batch mode**: Fastest for production use
- **Interactive mode**: Best for debugging and inspection
- **Async mode**: Non-blocking for long-running operations

Typical execution times (on reference hardware):
- Simple arithmetic: <1ms
- Pattern matching (10 patterns): 5-10ms
- Complex reasoning (100+ frames): 50-100ms

## Error Handling

The engine provides comprehensive error handling:

```python
from puma.hyperon_subagents import (
    MeTTaEngineError,
    HyperonNotAvailableError,
    ExecutionError,
    CompilationError
)

try:
    result = engine.execute_program(code)
except HyperonNotAvailableError:
    print("Hyperon not installed")
except ExecutionError as e:
    print(f"Execution failed: {e}")
except CompilationError as e:
    print(f"DSL compilation failed: {e}")
```

## Logging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("puma.hyperon_subagents.metta_engine")

# Now all engine operations are logged
```

## Testing

Run the example usage script:

```bash
python puma/hyperon_subagents/example_usage.py
```

Run comprehensive tests:

```bash
pytest tests/test_metta_engine.py -v
```

## Future Enhancements

- [ ] Parallel MeTTa execution across multiple cores
- [ ] GPU-accelerated pattern matching
- [ ] Distributed Atomspace for large-scale reasoning
- [ ] Advanced query optimization
- [ ] MeTTa-to-DSL reverse compilation
- [ ] Visual debugging interface
- [ ] Integration with neural guidance models

## Contributing

When extending the MeTTa engine:

1. Follow PUMA's RFT principles
2. Add comprehensive docstrings
3. Include usage examples
4. Write tests for new features
5. Update this README

## References

- [OpenCog Hyperon Documentation](https://wiki.opencog.org/w/Hyperon)
- [MeTTa Language Specification](https://github.com/trueagi-io/hyperon-experimental)
- [PUMA RFT Architecture](../../README.md)
- [Relational Frame Theory](https://en.wikipedia.org/wiki/Relational_frame_theory)

## License

Part of the PUMA cognitive architecture project.

## Authors

PUMA Development Team - Integration with Hyperon/MeTTa symbolic reasoning

---

**Note**: This module requires `hyperon>=0.3.0`. Install with:
```bash
pip install hyperon
```
