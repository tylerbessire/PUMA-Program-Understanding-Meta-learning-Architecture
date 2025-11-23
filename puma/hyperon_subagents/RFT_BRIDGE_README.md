# RFT-Hyperon Bridge: Integration Guide

## Overview

The RFT-Hyperon Bridge (`rft_bridge.py`) connects PUMA's Relational Frame Theory (RFT) system with Hyperon's MeTTa reasoning capabilities, creating a hybrid cognitive architecture that combines behavioral analysis with symbolic reasoning.

## Architecture

### Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      RFTHyperonBridge                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐         ┌─────────────────────┐         │
│  │  RFT System      │  <--->  │  MeTTa Engine       │         │
│  │  (Behavioral)    │         │  (Symbolic)         │         │
│  └──────────────────┘         └─────────────────────┘         │
│         ↑                              ↑                        │
│         │                              │                        │
│         v                              v                        │
│  ┌──────────────────┐         ┌─────────────────────┐         │
│  │ Frequency Ledger │         │  Atomspace          │         │
│  │ (Patterns)       │         │  (Knowledge)        │         │
│  └──────────────────┘         └─────────────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Integration Points

1. **RFT Frame ↔ MeTTa Conversion**
   - Convert RFT relational frames to MeTTa expressions
   - Parse MeTTa results back to RFT frames
   - Bidirectional translation layer

2. **Frequency Ledger Integration**
   - Convert frequency signatures to MeTTa knowledge
   - Derive relational frames from frequency patterns
   - MeTTa-based frequency analysis

3. **Relational Frame Composition**
   - Transitivity inference (A→B, B→C ⟹ A→C)
   - Symmetry inference (A↔B ⟹ B↔A)
   - Compositional reasoning

4. **Derived Relation Inference**
   - Use Hyperon's reasoning engine for inference
   - Apply logical rules to derive new relations
   - Automatic relation discovery

## RFT Relation Types

The bridge supports all RFT relation types:

### 1. Coordination (Same-As Relations)

**Behavioral Meaning**: Similarity, equivalence
**MeTTa Predicate**: `same-as`
**Example**:

```python
frame = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="red_square",
    target="red_circle",
    strength=0.85,
    context=["same_color"]
)

# Converts to MeTTa:
# (same-as red_square red_circle 0.85)
```

**Inference Rules**:
- **Symmetry**: `(same-as A B) → (same-as B A)`
- **Transitivity**: `(same-as A B) ∧ (same-as B C) → (same-as A C)`

### 2. Opposition (Opposite-Of Relations)

**Behavioral Meaning**: Difference, contrast
**MeTTa Predicate**: `opposite-of`
**Example**:

```python
frame = RelationalFrame(
    relation_type=RelationType.OPPOSITION,
    source="large",
    target="small",
    strength=1.0
)

# Converts to MeTTa:
# (opposite-of large small 1.0)
```

**Inference Rules**:
- **Symmetry**: `(opposite-of A B) → (opposite-of B A)`

### 3. Comparison (More-Than, Less-Than)

**Behavioral Meaning**: Magnitude relations
**MeTTa Predicates**: `more-than`, `less-than`
**Example**:

```python
frame = RelationalFrame(
    relation_type=RelationType.COMPARATIVE,
    source="large",
    target="medium",
    strength=1.0
)

# Converts to MeTTa:
# (more-than large medium 1.0)
```

**Inference Rules**:
- **Transitivity**: `(more-than A B) ∧ (more-than B C) → (more-than A C)`
- **Inverse**: `(more-than A B) → (less-than B A)`

### 4. Hierarchical (Contains, Part-Of)

**Behavioral Meaning**: Categorization, containment
**MeTTa Predicates**: `part-of`, `contains`
**Example**:

```python
frame = RelationalFrame(
    relation_type=RelationType.HIERARCHY,
    source="square",
    target="shape",
    strength=1.0,
    context=["category"]
)

# Converts to MeTTa:
# (part-of square shape 1.0)
```

**Inference Rules**:
- **Transitivity**: `(part-of A B) ∧ (part-of B C) → (part-of A C)`

### 5. Spatial Relations

**Behavioral Meaning**: Location, proximity
**MeTTa Predicate**: `near`
**Example**:

```python
fact = RelationalFact(
    relation="spatial_transform",
    subject=(1, 3, 3),  # red 3x3 object
    object=(2, 3, 3),   # blue 3x3 object
    metadata={'distance': 5.0},
    direction_vector=np.array([1.0, 0.0]),
    confidence=0.9
)

# Converts to MeTTa:
# (and (spatial-transform obj_1_3_3 obj_2_3_3 0.9)
#      (direction obj_1_3_3 obj_2_3_3 right))
```

### 6. Temporal Relations

**Behavioral Meaning**: Before/after, sequence
**MeTTa Predicate**: `before`
**Inference Rules**:
- **Transitivity**: `(before A B) ∧ (before B C) → (before A C)`

### 7. Causal Relations

**Behavioral Meaning**: If-then, causation
**MeTTa Predicate**: `causes`
**Example**:

```python
frame = RelationalFrame(
    relation_type=RelationType.CAUSAL,
    source="event_A",
    target="event_B",
    strength=0.7,  # Uncertain causation
    context=["potential_causation"]
)
```

## Usage Examples

### Example 1: Basic Conversion

```python
from puma.hyperon_subagents.rft_bridge import RFTHyperonBridge
from puma.rft.reasoning import RelationalFrame, RelationType

# Create bridge
bridge = RFTHyperonBridge()

# Create RFT frame
frame = RelationalFrame(
    relation_type=RelationType.COORDINATION,
    source="pattern_A",
    target="pattern_B",
    strength=0.85,
    context=["similar_structure"]
)

# Convert to MeTTa
metta_expr = bridge.rft_frame_to_metta(frame)
print(f"MeTTa: {metta_expr}")
# Output: (with-context ((same-as pattern_A pattern_B 0.85)) ("similar_structure"))

# Convert back to RFT
reconstructed = bridge.metta_to_rft_frame(metta_expr.split("(with-context")[0].strip())
print(f"Reconstructed: {reconstructed}")
```

### Example 2: Frame Composition (Transitivity)

```python
# Create chain of relations
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

# Compose: A→B, B→C ⟹ A→C
composed = bridge.compose_frames(frame1, frame2)

print(f"Derived: {composed.source} → {composed.target}")
print(f"Strength: {composed.strength}")  # Decayed: min(0.9, 0.8) * 0.8 = 0.64
print(f"Is derived: {composed.derived}")  # True
```

### Example 3: Frequency Ledger Integration

```python
from arc_solver.frequency_ledger import FrequencyLedger, FrequencySignature

# Create frequency ledger from ARC task
ledger = FrequencyLedger()
ledger.color_frequencies = {1: 10, 2: 5, 3: 3}
ledger.size_frequencies = {9: 8, 4: 6, 1: 2}

# Create frequency signatures
sig1 = FrequencySignature(color=1, size=9, occurrence_count=8)
sig2 = FrequencySignature(color=1, size=9, occurrence_count=7)
sig3 = FrequencySignature(color=2, size=4, occurrence_count=5)

ledger.object_signatures = [sig1, sig2, sig3]
ledger.relational_groupings = {
    'group_0': [sig1, sig2],  # Similar objects
    'group_1': [sig3]
}

# Convert to MeTTa knowledge base
metta_exprs = bridge.frequency_ledger_to_metta(ledger)
for expr in metta_exprs[:5]:
    print(expr)

# Derive relational frames from frequency patterns
derived_frames = bridge.derive_frequency_relations(ledger)
print(f"\nDerived {len(derived_frames)} frames from frequency patterns")
for frame in derived_frames:
    print(f"  {frame.source} --[{frame.relation_type.value}]--> {frame.target}")
```

### Example 4: Derived Relation Inference

```python
# Set of known relations
known_frames = [
    RelationalFrame(RelationType.COORDINATION, "A", "B", 0.9),
    RelationalFrame(RelationType.COORDINATION, "B", "C", 0.8),
    RelationalFrame(RelationType.HIERARCHY, "X", "Y", 1.0),
    RelationalFrame(RelationType.HIERARCHY, "Y", "Z", 1.0),
    RelationalFrame(RelationType.COMPARATIVE, "small", "medium", 1.0),
    RelationalFrame(RelationType.COMPARATIVE, "medium", "large", 1.0),
]

# Infer new relations using Hyperon
derived = bridge.infer_derived_relations(known_frames, max_depth=2)

print(f"Inferred {len(derived)} new relations:")
for frame in derived:
    print(f"  {frame.source} --[{frame.relation_type.value}]--> {frame.target}")
    print(f"    Context: {frame.context}, Derived: {frame.derived}")
```

### Example 5: ARC Solver Integration

```python
from arc_solver.rft import RelationalFrameAnalyzer, RelationalFact
import numpy as np

# Analyze ARC task
analyzer = RelationalFrameAnalyzer()
facts = analyzer.analyze(train_pairs)

# Convert spatial facts to MeTTa
for fact in facts['spatial'][:5]:
    metta_expr = bridge.rft_fact_to_metta(fact)
    print(metta_expr)

# Convert transformation facts
for fact in facts['transformation'][:5]:
    metta_expr = bridge.rft_fact_to_metta(fact)
    print(metta_expr)
```

## MeTTa Programs for RFT Operations

The bridge initializes MeTTa with comprehensive RFT reasoning programs:

### Coordination (Similarity)

```scheme
; Transitivity
(= (derive-coordination $A $B $C)
   (if (and (same-as $A $B) (same-as $B $C))
       (same-as $A $C)))

; Symmetry
(= (coordination-symmetric $A $B)
   (if (same-as $A $B)
       (same-as $B $A)))
```

### Opposition (Contrast)

```scheme
; Symmetry
(= (opposition-symmetric $A $B)
   (if (opposite-of $A $B)
       (opposite-of $B $A)))
```

### Hierarchy (Part-Of)

```scheme
; Transitivity
(= (derive-hierarchy $A $B $C)
   (if (and (part-of $A $B) (part-of $B $C))
       (part-of $A $C)))
```

### Comparison (More/Less)

```scheme
; Transitivity
(= (derive-comparison $A $B $C)
   (if (and (more-than $A $B) (more-than $B $C))
       (more-than $A $C)))

; Inverse
(= (comparison-inverse $A $B)
   (if (more-than $A $B)
       (less-than $B $A)))
```

### Temporal (Before)

```scheme
; Transitivity
(= (derive-temporal $A $B $C)
   (if (and (before $A $B) (before $B $C))
       (before $A $C)))
```

### Frequency-Based Similarity

```scheme
; If two signatures belong to same frequency group, they are similar
(= (frequency-similar $A $B)
   (if (and (belongs-to-group $A $group)
           (belongs-to-group $B $group))
       (same-as $A $B)))
```

## Integration Approach

### 1. Behavioral Foundation (RFT)

PUMA's RFT system provides the behavioral foundation:

- **Learned Relational Responding**: Relations emerge from behavioral contingencies
- **Derivational Relations**: Models derive new relations without explicit training
- **Contextual Control**: Relational responding is context-dependent
- **Equivalence Classes**: Objects with similar properties form behavioral equivalence classes

### 2. Symbolic Reasoning (Hyperon)

Hyperon provides symbolic reasoning capabilities:

- **Logical Inference**: Apply formal logical rules to derive new knowledge
- **Pattern Matching**: Match complex patterns in knowledge base
- **Knowledge Representation**: Represent knowledge in structured atomspace
- **Query Execution**: Execute complex queries over knowledge

### 3. Hybrid Architecture

The bridge combines both approaches:

```
Behavioral Analysis (RFT) → Bridge → Symbolic Reasoning (Hyperon)
         ↓                              ↓
  Pattern Discovery              Logical Inference
  Frequency Analysis             Rule Application
  Similarity Detection           Knowledge Integration
         ↓                              ↓
         └──────────→ Hybrid Reasoning ←────────┘
                            ↓
                    Emergent Intelligence
```

### 4. Frequency-Guided Reasoning

The Frequency Ledger System enhances symbolic reasoning:

1. **Frequency Analysis**: Identify numerical patterns in data
2. **Abstract Groupings**: Cluster similar objects by frequency
3. **MeTTa Encoding**: Represent frequency knowledge symbolically
4. **Guided Inference**: Use frequency patterns to guide logical inference

### 5. Confidence Propagation

Relations have strength/confidence values that propagate through inference:

- **Direct relations**: Full confidence (e.g., 1.0)
- **Symmetric relations**: Same confidence as original
- **Transitive relations**: Decayed confidence (min * 0.8)
- **Frequency-based**: Computed from similarity score

## Benefits of Integration

### 1. Emergent Reasoning

Combining behavioral and symbolic approaches enables emergent capabilities:

- **Novel Derivations**: Discover relations never explicitly trained
- **Analogical Transfer**: Apply learned patterns to new situations
- **Abstract Generalization**: Form abstract concepts from concrete examples

### 2. Grounded Symbols

RFT grounds symbolic reasoning in behavioral analysis:

- **Behavioral Meaning**: Symbols have behavioral significance
- **Frequency-Based**: Symbols emerge from statistical patterns
- **Context-Dependent**: Symbol meaning depends on context

### 3. Scalable Inference

Hyperon's reasoning engine enables efficient large-scale inference:

- **Parallel Reasoning**: Execute multiple inference chains in parallel
- **Incremental Updates**: Update knowledge base incrementally
- **Query Optimization**: Optimize complex queries

### 4. Human-Like Reasoning

The hybrid architecture mirrors human cognitive processes:

- **Bottom-Up**: Pattern discovery from experience (RFT)
- **Top-Down**: Rule-based reasoning (Hyperon)
- **Interactive**: Bidirectional information flow

## Performance Characteristics

### Conversion Performance

- **RFT → MeTTa**: O(1) for single frame, O(n) for n frames
- **MeTTa → RFT**: O(1) for simple expressions
- **Caching**: Converted relations are cached for reuse

### Inference Performance

- **Symmetry**: O(n) for n known frames
- **Transitivity**: O(n²) for pairwise composition
- **Max Depth**: Configurable to limit inference depth
- **Pruning**: Confidence threshold for pruning low-quality inferences

### Memory Usage

- **Relation Cache**: O(n) for n cached relations
- **MeTTa Space**: Depends on Hyperon's atomspace implementation
- **Frequency Ledger**: O(m) for m object signatures

## Future Enhancements

1. **Advanced Inference**
   - Multi-step causal chains with confidence decay
   - Analogical mapping between problem domains
   - Concept blending and synthesis

2. **Learning Integration**
   - Update relation strengths based on outcomes
   - Learn new relation types from experience
   - Meta-learning over relational patterns

3. **Distributed Reasoning**
   - Parallel inference across multiple Hyperon instances
   - Distributed knowledge base
   - Federated learning of relational frames

4. **Visualization**
   - Visual representation of relational networks
   - Interactive exploration of derived relations
   - Explanation generation for inferences

## Testing

Run the test suite:

```bash
cd /home/user/PUMA-Program-Understanding-Meta-learning-Architecture
python -m pytest puma/hyperon_subagents/test_rft_bridge.py -v
```

Or run the example demonstrations:

```bash
python puma/hyperon_subagents/rft_bridge.py
```

## References

- **RFT Implementation**: `arc_solver/rft.py`, `puma/rft/reasoning.py`
- **Frequency Ledger**: `arc_solver/frequency_ledger.py`
- **Hyperon Documentation**: https://github.com/trueagi-io/hyperon-experimental
- **MeTTa Language**: https://metta-lang.dev/

## Summary

The RFT-Hyperon Bridge creates a powerful hybrid cognitive architecture that:

1. ✅ Converts RFT frames to MeTTa expressions (bidirectional)
2. ✅ Implements MeTTa programs for all RFT operation types
3. ✅ Integrates Frequency Ledger for MeTTa-based frequency analysis
4. ✅ Supports relational frame composition via transitivity
5. ✅ Enables derived relation inference using Hyperon's reasoning engine
6. ✅ Provides comprehensive examples and tests

This integration combines the strengths of behavioral analysis (RFT) with symbolic reasoning (Hyperon), enabling emergent intelligent capabilities that mirror human relational framing and logical reasoning.
