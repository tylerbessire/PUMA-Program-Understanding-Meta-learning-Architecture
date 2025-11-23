# MeTTa Execution Engine - Quick Start Guide

## Installation

```bash
# Ensure Hyperon is installed
pip install hyperon>=0.3.0

# All set! The module is ready to use.
```

## 5-Minute Quick Start

### 1. Basic Execution

```python
from puma.hyperon_subagents import MeTTaExecutionEngine

# Initialize
engine = MeTTaExecutionEngine()

# Execute simple program
result = engine.execute_program("(+ 2 3)")
print(result.results)  # Output: [5]
```

### 2. RFT Integration

```python
from puma.rft import RelationalFrame, RelationType

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

# Execute
result = engine.execute_program(metta_expr)
```

### 3. DSL Compilation

```python
# Define PUMA DSL operation
dsl_op = {
    "operation": "pattern_match",
    "params": {
        "pattern": "(cell ?x ?y blue)",
        "target": "(cell 0 0 blue)"
    }
}

# Compile to MeTTa
metta_code = engine.compile_dsl_to_metta(dsl_op)

# Execute
result = engine.execute_program(metta_code)
```

### 4. Load MeTTa Files

```python
# Load and execute .metta file
result = engine.load_metta_file("sample_programs.metta")
print(f"Execution time: {result.execution_time}s")
```

### 5. Query Atomspace

```python
# Add some data
engine.execute_program("""
(RelFrame coordination square rectangle 0.8)
(RelFrame coordination circle ellipse 0.7)
""")

# Query for coordination frames
results = engine.query_atomspace(
    "(RelFrame coordination ?source ?target ?strength)"
)

for result in results:
    print(result)
```

## Key Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `execute_program(code)` | Execute MeTTa code | `engine.execute_program("(+ 1 2)")` |
| `load_metta_file(path)` | Load .metta file | `engine.load_metta_file("program.metta")` |
| `compile_dsl_to_metta(dsl)` | Compile PUMA DSL | `engine.compile_dsl_to_metta(dsl_op)` |
| `rft_to_metta(frame)` | Convert RFT frame | `engine.rft_to_metta(frame)` |
| `query_atomspace(pattern)` | Query with pattern | `engine.query_atomspace("(atom ?x)")` |
| `register_atom(name, value)` | Add custom atom | `engine.register_atom("my_atom", 42)` |

## Execution Modes

```python
from puma.hyperon_subagents import ExecutionMode

# Batch mode (fastest, default)
engine = MeTTaExecutionEngine(execution_mode=ExecutionMode.BATCH)

# Interactive mode (step-by-step)
engine = MeTTaExecutionEngine(execution_mode=ExecutionMode.INTERACTIVE)

# Async mode (non-blocking)
engine = MeTTaExecutionEngine(execution_mode=ExecutionMode.ASYNC)
```

## Sample Programs

```python
# Get all sample programs
samples = engine.get_sample_programs()

print(samples["pattern_matching"])
print(samples["transformation"])
print(samples["frequency_analysis"])
```

## Error Handling

```python
from puma.hyperon_subagents import ExecutionError, CompilationError

try:
    result = engine.execute_program(code)
    if not result.success:
        print(f"Execution failed: {result.error}")
except ExecutionError as e:
    print(f"Error: {e}")
```

## Statistics

```python
# Get execution statistics
stats = engine.get_statistics()

print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Average time: {stats['average_execution_time']:.4f}s")
```

## Complete Example: ARC-AGI Grid Analysis

```python
from puma.hyperon_subagents import MeTTaExecutionEngine
from puma.rft import RelationalFrame, RelationType

# Initialize engine
engine = MeTTaExecutionEngine()

# Define grid cells
grid_program = """
(cell 0 0 blue)
(cell 1 0 blue)
(cell 2 0 red)
(cell 0 1 green)
(cell 1 1 blue)
(cell 2 1 blue)
"""

# Execute to populate atomspace
engine.execute_program(grid_program)

# Query for blue cells
blue_cells = engine.query_atomspace("(cell ?x ?y blue)")
print(f"Found {len(blue_cells)} blue cells")

# Add relational frames for pattern recognition
frames = [
    RelationalFrame(
        relation_type=RelationType.COORDINATION,
        source="pattern_1",
        target="pattern_2",
        strength=0.9
    )
]

# Convert frames to MeTTa
for frame in frames:
    metta_expr = engine.rft_to_metta(frame)
    engine.execute_program(metta_expr)

# Compile DSL for transformation
transform_dsl = {
    "operation": "transform",
    "params": {
        "input_pattern": "(cell ?x ?y blue)",
        "output_pattern": "(cell ?x ?y red)",
        "target": "$grid"
    }
}

transform_code = engine.compile_dsl_to_metta(transform_dsl)
print(f"Transformation: {transform_code}")

# Get statistics
stats = engine.get_statistics()
print(f"\nStatistics: {stats}")
```

## Next Steps

1. Read the full README: `README.md`
2. Run examples: `python example_usage.py`
3. Load sample programs: `sample_programs.metta`
4. Run tests: `pytest tests/test_metta_engine.py`

## Common Use Cases

### Pattern Matching for ARC-AGI
```metta
!(match &self (cell ?x ?y blue) $result)
```

### Frequency Analysis (PUMA's Core Innovation)
```metta
!(group-by-frequency
    (object obj1 (frequency 3))
    (object obj2 (frequency 1))
    (object obj3 (frequency 3)))
```

### Relational Reasoning
```metta
!(match &self
    (RelFrame coordination ?source ?target ?strength)
    (> ?strength 0.7))
```

### Transformation Rules
```metta
!(transform-by-pattern
    (cell ?x ?y blue)
    (cell ?x ?y red)
    $grid)
```

## Resources

- Full Documentation: `README.md`
- Example Code: `example_usage.py`
- Sample Programs: `sample_programs.metta`
- Test Suite: `tests/test_metta_engine.py`
- PUMA Documentation: `../../README.md`
- Hyperon Docs: https://wiki.opencog.org/w/Hyperon

## Support

For issues or questions:
1. Check the full README
2. Review example_usage.py
3. Run the test suite
4. Consult PUMA documentation

---

**Happy Reasoning!** 🧠
