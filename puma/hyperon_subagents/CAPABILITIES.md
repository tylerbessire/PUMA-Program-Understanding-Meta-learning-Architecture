# MeTTa Execution Engine - Comprehensive Capabilities

## Overview

The MeTTa Execution Engine is a comprehensive symbolic reasoning module for PUMA's cognitive architecture, providing seamless integration between Relational Frame Theory (RFT) and MeTTa symbolic program execution.

## Core Capabilities

### 1. Program Execution (Multiple Modes)

**Supported Modes:**
- **Batch Mode**: Execute entire programs at once (optimized for speed)
- **Interactive Mode**: Step-by-step execution with inspection capabilities
- **Async Mode**: Non-blocking asynchronous execution

**Features:**
- Timeout support for all execution modes
- Comprehensive error handling
- Execution history tracking
- Performance metrics collection

### 2. RFT Integration

**Relational Frame Types Supported:**
- `COORDINATION` - Similarity relations (X is like Y)
- `OPPOSITION` - Difference relations (X is opposite of Y)
- `HIERARCHY` - Categorization (X is a type of Y)
- `TEMPORAL` - Sequence relations (X before Y)
- `CAUSAL` - If-then relations (X causes Y)
- `COMPARATIVE` - Magnitude relations (X > Y)
- `SPATIAL` - Location relations (X near Y)

**Translation Functions:**
- `rft_to_metta()` - Convert RelationalFrame to MeTTa expression
- `context_to_metta()` - Convert RFT Context to MeTTa knowledge base
- `entity_to_metta()` - Convert PUMA Entity to MeTTa atom

### 3. PUMA DSL Compilation

**Supported DSL Operations:**

| Operation | MeTTa Output | Purpose |
|-----------|--------------|---------|
| `pattern_match` | `!(match ...)` | Pattern matching in atomspace |
| `transform` | `!(transform-by-pattern ...)` | Pattern-based rewriting |
| `frequency_analysis` | `!(group-by-frequency ...)` | PUMA's frequency ledger |
| `relational_query` | `!(match &self (RelFrame ...))` | Query relational frames |
| `custom` | User-provided MeTTa | Direct MeTTa code |

### 4. Atomspace Management

**Features:**
- Register custom atoms (strings, numbers, dicts, objects)
- Query atomspace with patterns
- Persistent knowledge representation
- Atom indexing and retrieval
- Frame storage and tracking

### 5. File Operations

**Capabilities:**
- Load and execute .metta files
- Sample program library included
- Batch file processing
- Error reporting for file operations

### 6. Pattern Matching

**MeTTa Pattern Matching:**
- Variable binding with `?variable`
- Wildcard matching
- Nested pattern support
- Result extraction and binding

**Example:**
```metta
!(match &self (cell ?x ?y blue) (cell ?x ?y blue))
```

### 7. Frequency Ledger System

**PUMA's Core Innovation:**
- Group objects by frequency attributes
- Frequency-based pattern discovery
- Abstract grouping operations
- Numerical relationship analysis

**Example:**
```metta
!(group-by-frequency
    (object obj1 (frequency 3))
    (object obj2 (frequency 1))
    (object obj3 (frequency 3)))
```

### 8. Transformation Rules

**Pattern-Based Rewriting:**
- Input pattern specification
- Output pattern generation
- Grid transformation support
- Multi-step transformations

**Example:**
```metta
!(transform-by-pattern
    (cell ?x ?y blue)
    (cell ?x ?y red)
    $grid)
```

### 9. Logging and Monitoring

**Comprehensive Logging:**
- Execution events
- Performance metrics
- Error tracking
- Debug information

**Statistics Collection:**
- Total executions
- Success/failure rates
- Average execution time
- Atomspace size
- Frame count

### 10. Error Handling

**Exception Hierarchy:**
- `MeTTaEngineError` - Base exception
- `HyperonNotAvailableError` - Missing dependency
- `ExecutionError` - Program execution failure
- `CompilationError` - DSL compilation failure

**Features:**
- Graceful error recovery
- Detailed error messages
- Error logging
- Safe fallback behavior

## Advanced Features

### Type Conversions

**Python ↔ MeTTa:**
- String → MeTTa string atom
- Number → MeTTa number atom
- Bool → MeTTa boolean
- List/Tuple → MeTTa expression
- Dict → MeTTa structured atom

### Sample Programs Library

**12 Categories of Sample Programs:**
1. Pattern Matching - Grid analysis
2. Relational Reasoning - RFT frame queries
3. Frequency Analysis - Ledger operations
4. Transformations - Pattern rewriting
5. Causal Reasoning - Temporal chains
6. Derivational Reasoning - Transitivity
7. Comparative Reasoning - Magnitudes
8. Spatial Reasoning - Locations
9. Episodic Memory - Experience tracking
10. Goal-Directed Reasoning - Planning
11. Meta-Learning - Learning to learn
12. Self-Modification - Code introspection

### Query Interface

**Pattern-Based Queries:**
- Variable binding
- Constraint satisfaction
- Multi-pattern matching
- Result extraction

### Integration Points

**PUMA Systems:**
- RFT Engine
- Frequency Ledger
- Episodic Memory
- Goal System
- Self-Modification (Shop)
- Consciousness Layer

**External Systems:**
- OpenCog Hyperon
- Atomspace persistence
- Neural guidance models
- ARC-AGI solvers

## Performance Characteristics

**Execution Speed (Reference Hardware):**
- Simple arithmetic: <1ms
- Pattern matching (10 patterns): 5-10ms
- Complex reasoning (100 frames): 50-100ms
- File loading: Variable (depends on file size)

**Memory Usage:**
- Engine overhead: ~10-20MB
- Per atom: ~1-5KB
- Per frame: ~2-8KB
- Execution history: Configurable retention

**Scalability:**
- Tested with 1000+ atoms
- Tested with 500+ relational frames
- Batch processing optimized
- Async execution for long operations

## Use Cases

### 1. ARC-AGI Puzzle Solving
- Grid pattern analysis
- Transformation rule discovery
- Analogical reasoning
- Frequency-based grouping

### 2. Knowledge Representation
- Persistent memory storage
- Relational knowledge graphs
- Hierarchical categorization
- Temporal event sequences

### 3. Abstract Reasoning
- Pattern matching and recognition
- Rule-based inference
- Analogical transfer
- Meta-learning

### 4. Cognitive Architecture
- RFT-based reasoning
- Goal-directed behavior
- Self-modification support
- Experience integration

## API Summary

### Core Methods

```python
# Execution
execute_program(code, mode, timeout) -> ExecutionResult
load_metta_file(filepath) -> ExecutionResult

# Atomspace
register_atom(name, value, type) -> HyperonAtom
query_atomspace(pattern) -> List[Dict]

# Compilation
compile_dsl_to_metta(dsl_operation) -> str

# Translation
rft_to_metta(frame) -> str
context_to_metta(context) -> str
entity_to_metta(entity) -> str

# Utilities
get_sample_programs() -> Dict[str, str]
get_statistics() -> Dict[str, Any]
reset() -> None
```

### Data Structures

```python
ExecutionResult(
    success: bool,
    results: List[Any],
    execution_time: float,
    mode: ExecutionMode,
    error: Optional[str],
    metadata: Dict[str, Any],
    timestamp: datetime
)
```

## Extension Points

**How to Extend:**

1. **Custom Operations**: Add new DSL operations
2. **Custom Atoms**: Register domain-specific atoms
3. **Custom Functions**: Define MeTTa functions
4. **Custom Queries**: Create specialized query patterns
5. **Custom Transformations**: Add transformation rules

## Testing

**Test Coverage:**
- Unit tests for all core methods
- Integration tests for RFT workflow
- Edge case testing
- Error handling verification
- Performance benchmarks

**Test Suite:**
- 40+ test cases
- Multiple test classes
- Comprehensive edge cases
- Integration scenarios

## Future Enhancements

**Planned Features:**
1. Parallel execution across cores
2. GPU-accelerated pattern matching
3. Distributed atomspace
4. Query optimization
5. Visual debugging interface
6. MeTTa-to-DSL reverse compilation

## Documentation

**Available Resources:**
- Full README: `README.md`
- Quick Start: `QUICK_START.md`
- Capabilities: `CAPABILITIES.md` (this file)
- Examples: `example_usage.py`
- Sample Programs: `sample_programs.metta`
- Tests: `tests/test_metta_engine.py`

## Compatibility

**Dependencies:**
- Python 3.11+
- hyperon >= 0.3.0
- PUMA RFT module
- Standard library (asyncio, logging, etc.)

**Platforms:**
- Linux ✓
- macOS ✓
- Windows ✓ (with Hyperon support)

## License

Part of the PUMA cognitive architecture project.

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-23  
**Maintainer**: PUMA Development Team
