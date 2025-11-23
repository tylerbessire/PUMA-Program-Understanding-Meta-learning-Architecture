# Hyperon Atomspace Integration

## Overview

This document describes the integration of native Hyperon Atomspace into PUMA's cognitive architecture. The integration provides:

1. **Dual persistence**: Both JSON (backward compatible) and Hyperon native storage
2. **Type mapping**: PUMA schema types mapped to Hyperon atom types
3. **Pattern matching**: Leverage Hyperon's MeTTa query capabilities
4. **Backward compatibility**: Existing code continues to work unchanged

## Architecture

### Class Hierarchy

```
Atomspace (JSON-based)
    ├── Basic atom storage (Dict)
    ├── JSON persistence
    └── Simple queries

HyperonAtomspaceAdapter
    ├── Inherits Atomspace API
    ├── Dual storage (JSON + Hyperon)
    ├── MeTTa runtime integration
    ├── Pattern matching queries
    └── Bidirectional type conversion
```

### Key Components

1. **HyperonAtomspaceAdapter**: Main integration class
2. **Type Converters**: `_puma_atom_to_hyperon()` and `_hyperon_atom_to_puma()`
3. **Persistence Layer**: Dual JSON + MeTTa file storage
4. **Query Engine**: Hyperon pattern matching via MeTTa

## Type Mappings

### PUMA → Hyperon Atom Types

| PUMA Type | Hyperon Representation | MeTTa Syntax |
|-----------|----------------------|--------------|
| `EpisodicMemoryNode` | EpisodicMemory | `(EpisodicMemory id content timestamp tv conf)` |
| `ConceptNode` | Concept | `(Concept id name properties tv conf)` |
| `RelationalFrameNode` | RelationalFrame | `(RelationalFrame id type relations tv conf)` |
| `CodeNode` | Code (MeTTa executable) | `(Code id metta-expr timestamp)` |
| `SelfModelNode` | SelfModel | `(SelfModel id properties timestamp tv conf)` |
| `GoalNode` | Goal | `(Goal id content timestamp tv conf)` |
| `PerceptionNode` | Perception | `(Perception id content timestamp tv conf)` |
| `EmotionalStateNode` | EmotionalState | `(EmotionalState id content timestamp tv conf)` |

### Links

PUMA Links are represented as Hyperon expressions:

```metta
(Link source_id link_type target_id strength)
```

Example:
```metta
(Link episode_001 "relates_to" concept_learning 0.9)
```

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from atomspace-db.core import bootstrap_atomspace, Atom, AtomType

# Create atomspace with Hyperon integration
atomspace = bootstrap_atomspace(
    persistence_path=Path('./data/atomspace'),
    use_hyperon=True  # Use Hyperon if available
)

# Add atoms (same API as before)
atom = Atom(
    id="concept_001",
    type=AtomType.CONCEPT,
    content={'name': 'learning', 'domain': 'cognitive'},
    timestamp=datetime.now(timezone.utc)
)
atomspace.add_atom(atom)

# Save to dual storage (JSON + Hyperon)
atomspace.save()
```

### Pattern Matching Queries

```python
from atomspace-db.core import HyperonAtomspaceAdapter

adapter = HyperonAtomspaceAdapter(Path('./data'))

# Query all concepts
results = adapter.query_hyperon("(Concept $id $name $props $tv $conf)")

# Query specific episodic memories
results = adapter.query_hyperon("(EpisodicMemory $id $content $ts $tv $conf)")

# Query links of specific type
results = adapter.query_hyperon('(Link $source "semantic" $target $strength)')
```

### Type-Specific Examples

#### Episodic Memory
```python
episode = Atom(
    id="ep_001",
    type=AtomType.EPISODIC_MEMORY,
    content={
        'event': 'learned new concept',
        'context': 'study session',
        'outcome': 'success'
    },
    timestamp=datetime.now(timezone.utc),
    truth_value=0.9,
    confidence=0.85
)
adapter.add_atom(episode)
# Hyperon: (EpisodicMemory ep_001 "{...}" "2025-11-23T..." "0.9" "0.85")
```

#### Concept Node
```python
concept = Atom(
    id="concept_learning",
    type=AtomType.CONCEPT,
    content={'name': 'learning', 'properties': {'abstract': True}},
    timestamp=datetime.now(timezone.utc)
)
adapter.add_atom(concept)
# Hyperon: (Concept concept_learning "learning" "{...}" "1.0" "1.0")
```

#### Code Node (MeTTa)
```python
code = Atom(
    id="factorial",
    type=AtomType.CODE,
    content="""
    (= (factorial 0) 1)
    (= (factorial $n) (* $n (factorial (- $n 1))))
    """,
    timestamp=datetime.now(timezone.utc)
)
adapter.add_atom(code)
# Hyperon: (Code factorial <parsed-metta-expr> "2025-11-23T...")
```

## Persistence

### Dual Storage Format

When you call `atomspace.save()`, the system creates:

1. **JSON Files** (backward compatible):
   - `atoms.json`: All atoms with metadata
   - `links.json`: All links between atoms
   - `transaction_log.json`: Transaction history

2. **Hyperon Native** (if enabled):
   - `atomspace.metta`: MeTTa format with all atoms and type definitions

### File Structure

```
persistence_path/
├── atoms.json              # JSON atom storage
├── links.json              # JSON link storage
├── atomspace.metta         # Hyperon native storage
├── transaction_log.json    # Transaction log
└── snapshots/              # Versioned snapshots
    ├── 20251123_143022/
    │   ├── atoms.json
    │   ├── links.json
    │   └── atomspace.metta
    └── 20251123_150130/
        └── ...
```

### Snapshots

Create versioned snapshots for rollback:

```python
# Create snapshot
snapshot_id = atomspace.create_snapshot()
print(f"Snapshot: {snapshot_id}")  # "20251123_143022"

# Restore from snapshot
atomspace.restore_snapshot(snapshot_id)
```

## Pattern Matching

### Query Patterns

Hyperon uses MeTTa pattern matching with variables (`$var`):

```python
# Match all concepts
pattern = "(Concept $id $name $props $tv $conf)"

# Match specific link type
pattern = '(Link $source "semantic" $target $strength)'

# Match episodic memories with high truth value
pattern = "(EpisodicMemory $id $content $ts $tv $conf)"
# (then filter results by tv > 0.8)
```

### Example Queries

```python
# All episodic memories
atomspace.query_hyperon("(EpisodicMemory $id $content $ts $tv $conf)")

# All concepts
atomspace.query_hyperon("(Concept $id $name $props $tv $conf)")

# All goals
atomspace.query_hyperon("(Goal $id $content $ts $tv $conf)")

# Links from specific atom
atomspace.query_hyperon("(Link atom_123 $type $target $strength)")

# Relational frames of specific type
atomspace.query_hyperon("(RelationalFrame $id cause-effect $relations $tv $conf)")
```

## Backward Compatibility

### Existing Code Works Unchanged

```python
# Old code using Atomspace
from atomspace-db.core import Atomspace

atomspace = Atomspace(Path('./data'))
atomspace.add_atom(atom)
atomspace.save()
```

### New Code Can Use Hyperon

```python
# New code can leverage Hyperon
from atomspace-db.core import HyperonAtomspaceAdapter

atomspace = HyperonAtomspaceAdapter(Path('./data'), use_hyperon=True)
atomspace.add_atom(atom)  # Stored in both JSON and Hyperon
results = atomspace.query_hyperon("(Concept $id $name $props $tv $conf)")
```

### Migration Path

1. Replace `Atomspace` with `HyperonAtomspaceAdapter`
2. Set `use_hyperon=True`
3. Existing JSON data loads automatically
4. Start using pattern matching queries

## Configuration

### Check Hyperon Availability

```python
from atomspace-db.core import get_atomspace_info

info = get_atomspace_info()
print(f"Hyperon available: {info['hyperon_available']}")
print(f"Default backend: {info['default_backend']}")
print(f"Supported backends: {info['supported_backends']}")
```

### Choose Backend

```python
# Use Hyperon if available
atomspace = bootstrap_atomspace(use_hyperon=True)

# Force JSON-only (no Hyperon)
atomspace = bootstrap_atomspace(use_hyperon=False)

# Explicitly choose implementation
from atomspace-db.core import HyperonAtomspaceAdapter, Atomspace

# Hyperon-enabled
atomspace = HyperonAtomspaceAdapter(Path('./data'), use_hyperon=True)

# JSON-only
atomspace = Atomspace(Path('./data'))
```

## Error Handling

The integration includes graceful fallback:

```python
# If Hyperon import fails
HYPERON_AVAILABLE = False

# Adapter falls back to JSON-only mode
adapter = HyperonAtomspaceAdapter(Path('./data'), use_hyperon=True)
print(adapter.use_hyperon)  # False if Hyperon unavailable
```

Warnings are printed but execution continues:
- `"Warning: Hyperon initialization failed. Falling back to JSON."`
- `"Warning: Failed to convert PUMA atom to Hyperon"`
- `"Warning: Hyperon query failed"`

## Testing

Run the integration examples:

```bash
cd atomspace-db
python hyperon_integration_example.py
```

This runs:
1. Basic usage demo
2. Type mapping examples
3. Pattern matching queries
4. Dual persistence demo
5. Snapshot/restore demo
6. Backward compatibility check

## API Reference

### HyperonAtomspaceAdapter

#### Constructor
```python
HyperonAtomspaceAdapter(
    persistence_path: Optional[Path] = None,
    use_hyperon: bool = True
)
```

#### Methods

**Atom Operations**
- `add_atom(atom: Atom) -> str`: Add atom to both JSON and Hyperon stores
- `get_atom(atom_id: str) -> Optional[Atom]`: Retrieve atom by ID
- `query_by_type(atom_type: AtomType) -> List[Atom]`: Query atoms by type

**Link Operations**
- `add_link(link: Link)`: Add link to both stores
- `get_linked_atoms(atom_id: str, link_type: Optional[str]) -> List[Atom]`: Get linked atoms

**Hyperon-Specific**
- `query_hyperon(pattern: str) -> List[Dict[str, Any]]`: Pattern matching query
- `_puma_atom_to_hyperon(atom: Atom) -> Optional[HyperonAtom]`: Convert to Hyperon
- `_hyperon_atom_to_puma(hatom: HyperonAtom) -> Optional[Atom]`: Convert from Hyperon

**Persistence**
- `save()`: Save to dual storage (JSON + Hyperon)
- `load()`: Load from dual storage
- `create_snapshot() -> str`: Create versioned snapshot
- `restore_snapshot(snapshot_id: str)`: Restore from snapshot

**Utilities**
- `count_atoms() -> int`: Count total atoms
- `count_concepts() -> int`: Count concept nodes

## Performance Considerations

### Dual Persistence Overhead

- **Write operations**: ~2x slower (writes to both JSON and Hyperon)
- **Read operations**: Same speed (reads from JSON)
- **Queries**: Much faster with Hyperon pattern matching for complex queries

### When to Use Hyperon

**Use Hyperon when:**
- Complex pattern matching needed
- MeTTa reasoning integration required
- Advanced graph queries
- Large-scale knowledge graphs

**Use JSON-only when:**
- Simple CRUD operations
- Small datasets
- No pattern matching needed
- Minimal dependencies preferred

## Future Enhancements

Planned improvements:
1. Native Hyperon persistence (eliminate JSON dependency)
2. Advanced pattern matching templates
3. MeTTa reasoning integration with PUMA agents
4. Distributed atomspace support
5. Real-time synchronization between JSON and Hyperon

## Troubleshooting

### Hyperon Not Available

```python
info = get_atomspace_info()
if not info['hyperon_available']:
    print("Install Hyperon: pip install hyperon>=0.3.0")
```

### Import Errors

If you see import errors:
```bash
pip install hyperon>=0.3.0
```

### Persistence Issues

Check file permissions and paths:
```python
persistence_path = Path('./data/atomspace')
persistence_path.mkdir(parents=True, exist_ok=True)
```

## References

- [Hyperon Documentation](https://github.com/trueagi-io/hyperon-experimental)
- [MeTTa Language Guide](https://github.com/trueagi-io/hyperon-experimental/blob/main/docs/METTA.md)
- [PUMA Architecture](../README.md)

## License

Same as PUMA project license.
