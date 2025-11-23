# RFT-Hyperon Bridge Quick Start Guide

## Installation

1. **Install dependencies**:
```bash
pip install hyperon numpy
```

2. **Verify installation**:
```bash
python -c "from hyperon import MeTTa; print('Hyperon installed successfully')"
```

## 5-Minute Quick Start

### Example 1: Basic Conversion

```python
from puma.hyperon_subagents.rft_bridge import RFTHyperonBridge
from puma.rft.reasoning import RelationalFrame, RelationType

# Initialize bridge
bridge = RFTHyperonBridge()

# Create an RFT frame (similarity relation)
frame = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="red_square",
    target="red_circle",
    strength=0.85,
    context=["same_color"]
)

# Convert to MeTTa
metta_expr = bridge.rft_frame_to_metta(frame)
print(f"MeTTa: {metta_expr}")
# Output: (with-context ((same-as red_square red_circle 0.85)) ("same_color"))

# Convert back to RFT
reconstructed = bridge.metta_to_rft_frame("(same-as red_square red_circle 0.85)")
print(f"Reconstructed: {reconstructed}")
```

### Example 2: Derive Relations Through Transitivity

```python
# If A is similar to B, and B is similar to C, then A is similar to C

frame1 = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="A",
    target="B",
    strength=0.9
)

frame2 = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="B",
    target="C",
    strength=0.8
)

# Compose frames via transitivity
derived = bridge.compose_frames(frame1, frame2)

print(f"Given: {frame1.source} → {frame1.target} (strength: {frame1.strength})")
print(f"Given: {frame2.source} → {frame2.target} (strength: {frame2.strength})")
print(f"Derived: {derived.source} → {derived.target} (strength: {derived.strength})")
# Output: Derived: A → C (strength: 0.64)
```

### Example 3: Frequency Ledger Integration

```python
from arc_solver.frequency_ledger import FrequencyLedger, FrequencySignature

# Create frequency ledger
ledger = FrequencyLedger()
ledger.color_frequencies = {1: 10, 2: 5}
ledger.size_frequencies = {9: 8, 4: 6}

# Create similar objects
sig1 = FrequencySignature(color=1, size=9, occurrence_count=8,
                          shape_frequency=8, color_frequency=10)
sig2 = FrequencySignature(color=1, size=9, occurrence_count=7,
                          shape_frequency=8, color_frequency=10)

ledger.object_signatures = [sig1, sig2]
ledger.relational_groupings = {'group_0': [sig1, sig2]}

# Derive relations from frequency patterns
derived_frames = bridge.derive_frequency_relations(ledger)

print(f"Derived {len(derived_frames)} frames from frequency patterns:")
for frame in derived_frames:
    print(f"  {frame.source} → {frame.target} ({frame.relation_type.value})")
```

### Example 4: Batch Inference

```python
# Set of known relations
known_frames = [
    RelationalFrame(RelationType.COORDINATION, "A", "B", 0.9),
    RelationalFrame(RelationType.COORDINATION, "B", "C", 0.8),
    RelationalFrame(RelationType.COORDINATION, "C", "D", 0.7),
]

# Infer all derived relations
derived = bridge.infer_derived_relations(known_frames, max_depth=2)

print(f"Known: {len(known_frames)} frames")
print(f"Derived: {len(derived)} new frames")

for frame in derived:
    print(f"  {frame.source} → {frame.target} "
          f"(strength: {frame.strength:.2f}, context: {frame.context})")
```

### Example 5: ARC Task Integration

```python
from arc_solver.rft import RelationalFrameAnalyzer
import numpy as np

# Sample ARC input-output pair
input_grid = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])

output_grid = np.array([
    [0, 2, 0],
    [2, 2, 2],
    [0, 2, 0]
])

# Analyze with RFT
analyzer = RelationalFrameAnalyzer()
facts = analyzer.analyze([(input_grid, output_grid)])

# Convert spatial facts to MeTTa
print("Spatial relations in MeTTa:")
for fact in facts.get('spatial', [])[:5]:
    metta_expr = bridge.rft_fact_to_metta(fact)
    print(f"  {metta_expr}")

# Convert transformation facts to MeTTa
print("\nTransformation relations in MeTTa:")
for fact in facts.get('transformation', [])[:5]:
    metta_expr = bridge.rft_fact_to_metta(fact)
    print(f"  {metta_expr}")
```

## Common Use Cases

### Use Case 1: Analogical Reasoning

```python
# Find similar patterns across different domains

# Domain 1: Colors
color_frames = [
    RelationalFrame(RelationType.COORDINATION, "red", "crimson", 0.9),
    RelationalFrame(RelationType.COORDINATION, "blue", "navy", 0.9),
]

# Domain 2: Sizes
size_frames = [
    RelationalFrame(RelationType.COORDINATION, "large", "huge", 0.9),
    RelationalFrame(RelationType.COORDINATION, "small", "tiny", 0.9),
]

# Derive that both show similar relationship patterns
all_frames = color_frames + size_frames
derived = bridge.infer_derived_relations(all_frames)

# Use derived relations for transfer learning...
```

### Use Case 2: Hierarchical Knowledge

```python
# Build taxonomy through part-of relations

taxonomy = [
    RelationalFrame(RelationType.HIERARCHY, "poodle", "dog", 1.0),
    RelationalFrame(RelationType.HIERARCHY, "dog", "mammal", 1.0),
    RelationalFrame(RelationType.HIERARCHY, "mammal", "animal", 1.0),
]

# Derive: poodle is-a animal (via transitivity)
derived = bridge.infer_derived_relations(taxonomy)

for frame in derived:
    if frame.source == "poodle" and frame.target == "animal":
        print(f"Derived: {frame.source} is a {frame.target}")
        print(f"Confidence: {frame.strength}")
```

### Use Case 3: Comparison Chains

```python
# Build comparison hierarchy

sizes = [
    RelationalFrame(RelationType.COMPARATIVE, "tiny", "small", 1.0),
    RelationalFrame(RelationType.COMPARATIVE, "small", "medium", 1.0),
    RelationalFrame(RelationType.COMPARATIVE, "medium", "large", 1.0),
    RelationalFrame(RelationType.COMPARATIVE, "large", "huge", 1.0),
]

# Derive all comparison relations
derived = bridge.infer_derived_relations(sizes, max_depth=3)

# Now can compare any two sizes
for frame in derived:
    if frame.source == "tiny" and frame.target == "huge":
        print(f"{frame.source} less-than {frame.target}")
        print(f"Confidence: {frame.strength}")
```

## API Reference

### Core Methods

#### `rft_frame_to_metta(frame: RelationalFrame) → str`
Convert RFT frame to MeTTa expression.

**Example**:
```python
metta_expr = bridge.rft_frame_to_metta(frame)
```

#### `metta_to_rft_frame(metta_expr: str) → RelationalFrame`
Parse MeTTa expression back to RFT frame.

**Example**:
```python
frame = bridge.metta_to_rft_frame("(same-as A B 0.9)")
```

#### `compose_frames(frame1: RelationalFrame, frame2: RelationalFrame) → RelationalFrame`
Compose two frames via transitivity (if valid).

**Example**:
```python
composed = bridge.compose_frames(frame_AB, frame_BC)
# Returns frame_AC if valid
```

#### `infer_derived_relations(known_frames: List[RelationalFrame], max_depth: int = 3) → List[RelationalFrame]`
Infer new relations through symmetry, transitivity, and composition.

**Example**:
```python
derived = bridge.infer_derived_relations(known_frames, max_depth=2)
```

#### `derive_frequency_relations(ledger: FrequencyLedger) → List[RelationalFrame]`
Derive relational frames from frequency patterns.

**Example**:
```python
frames = bridge.derive_frequency_relations(ledger)
```

## Tips and Best Practices

### 1. Confidence Management

```python
# High confidence for direct observations
direct_frame = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="A",
    target="B",
    strength=1.0  # Direct observation
)

# Lower confidence for inferred relations
# (Automatically handled by compose_frames)
derived_frame = bridge.compose_frames(frame1, frame2)
# derived_frame.strength < min(frame1.strength, frame2.strength)
```

### 2. Controlling Inference Depth

```python
# Shallow inference (faster, fewer relations)
derived = bridge.infer_derived_relations(frames, max_depth=1)

# Deep inference (slower, more relations)
derived = bridge.infer_derived_relations(frames, max_depth=3)
```

### 3. Caching for Performance

```python
# Relations are automatically cached
metta_expr1 = bridge.rft_frame_to_metta(frame)  # Computed and cached
metta_expr2 = bridge.rft_frame_to_metta(frame)  # Retrieved from cache

# Check cache
stats = bridge.get_bridge_statistics()
print(f"Cached relations: {stats['cached_relations']}")
```

### 4. Export for Persistence

```python
# Export frames to MeTTa file for later use
frames = [frame1, frame2, frame3]
bridge.export_to_metta_file(frames, "my_knowledge.metta")

# Load in another session
# (Use MeTTa's load functionality)
```

## Troubleshooting

### Issue: ImportError for Hyperon

**Solution**: Install Hyperon
```bash
pip install hyperon
```

### Issue: ImportError for numpy

**Solution**: Install numpy
```bash
pip install numpy
```

### Issue: Frames won't compose

**Check**:
1. Are the relation types the same?
2. Does `frame1.target == frame2.source`?
3. Is the relation type transitive? (COORDINATION, HIERARCHY, COMPARATIVE, TEMPORAL)

```python
# Won't compose - different types
frame1 = RelationalFrame(RelationType.COORDINATION, "A", "B", 0.9)
frame2 = RelationalFrame(RelationType.HIERARCHY, "B", "C", 0.8)
composed = bridge.compose_frames(frame1, frame2)  # Returns None

# Won't compose - disconnected
frame1 = RelationalFrame(RelationType.COORDINATION, "A", "B", 0.9)
frame2 = RelationalFrame(RelationType.COORDINATION, "X", "Y", 0.8)
composed = bridge.compose_frames(frame1, frame2)  # Returns None

# Will compose - same type, connected, transitive
frame1 = RelationalFrame(RelationType.COORDINATION, "A", "B", 0.9)
frame2 = RelationalFrame(RelationType.COORDINATION, "B", "C", 0.8)
composed = bridge.compose_frames(frame1, frame2)  # Returns frame A→C
```

## Next Steps

1. **Read the full documentation**: `RFT_BRIDGE_README.md`
2. **Run the test suite**: `pytest test_rft_bridge.py -v`
3. **Run the examples**: `python rft_bridge.py`
4. **Integrate with your PUMA application**

## Support and Resources

- **Documentation**: `RFT_BRIDGE_README.md`
- **Integration Summary**: `INTEGRATION_SUMMARY.md`
- **Test Suite**: `test_rft_bridge.py`
- **Source Code**: `rft_bridge.py`

---

**Happy reasoning with RFT and Hyperon!** 🧠✨
