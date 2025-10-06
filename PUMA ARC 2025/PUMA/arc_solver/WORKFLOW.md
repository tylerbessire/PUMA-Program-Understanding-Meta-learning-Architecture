# PUMA ARC Solver: End-to-End Workflow

This document describes the complete workflow from object inventory through pliance rules, tracking, LLM reasoning, and search fusion.

## Architecture Overview

```
Training Pairs
    ↓
Object Inventory ──→ Persistent Objects + Deltas
    ↓
RFT Analysis ──→ Relational Facts + Patterns
    ↓
Tracking ──→ Conflicts + Repair Queue
    ↓
Pliance Engine ──→ Rules (Selector + Action)
    ↓
LLM Reasoning ──→ Priorities + Suggestions
    ↓
Search Fusion ──→ Weighted Candidates
    ↓
Enhanced Search ──→ Final Programs
```

## Pipeline Stages

### 1. Object Inventory (`object_inventory.py`)

**Purpose**: Build persistent object dictionary with IDs and attribute deltas across training pairs.

**Key Classes**:
- `ObjectInventory`: Main inventory manager
- `ObjectEntry`: Persistent object with ID, occurrences, attributes, variations
- `ObjectDelta`: Tracks changes between input/output

**Process**:
1. Extract objects from each training pair (input + output)
2. Match objects across pairs using signatures (color, shape, size, position)
3. Compute deltas for matched objects (moved, recolored, resized, etc.)
4. Track attribute variations to identify stable vs. changing properties

**Output**:
```python
{
    'objects': {
        'obj_0': {
            'id': 'obj_0',
            'attributes': {'color': 3, 'shape_type': 'rectangle', 'size': 12},
            'variations': {'color': [3], 'size': [12]},  # Stable
            'tags': ['placeholder', 'interior_object']
        },
        ...
    },
    'patterns': [
        {'type': 'consistent_transformation', 'transformation': 'recolored', 'confidence': 0.8}
    ],
    'transformation_rules': [...]
}
```

**Configuration**:
- No flags; always runs on training pairs

---

### 2. RFT Tracking (`rft_tracking.py`)

**Purpose**: Ingest object history, detect conflicts, queue rule repairs.

**Key Classes**:
- `RFTTracker`: Orchestrates tracking state
- `RuleConflict`: Represents conflicting rules with evidence
- `RuleRepairTask`: Queued repair with priority and suggested fixes
- `TaskTrackingCache`: Persists state across reruns

**Process**:
1. Build object inventory from training pairs
2. Run RFT relational analysis
3. Cross-reference inventory deltas with RFT facts to detect conflicts:
   - Transformation conflicts (same object → different outcomes)
   - Spatial conflicts (inconsistent spatial relations)
   - Multiple transformation types for same object
4. Queue repair tasks sorted by priority
5. Export state for LLM consumption

**Output**:
```python
{
    'task_id': 'task_abc123',
    'conflicts': [
        {
            'id': 'conflict_0',
            'rules': ['rule_1', 'rule_2'],
            'severity': 0.7,
            'status': 'pending'
        }
    ],
    'repair_tasks': [
        {
            'id': 'repair_0',
            'rule': 'rule_1',
            'issue': 'conflict',
            'priority': 0.7,
            'suggested_fixes': [...]
        }
    ]
}
```

**Configuration**:
- `cache_dir` (default: `.arc_tracking_cache`): Where to persist tracking state

---

### 3. Pliance Rule Engine (`pliance_engine.py`)

**Purpose**: Define, apply, and validate transformation rules.

**Key Classes**:
- `ObjectSelector`: Selects objects by color, shape, size, position, tags
- `SpatialRelation`: Defines spatial constraints between objects
- `RuleAction`: Transformation to apply (recolor, move, copy, delete, fill_region)
- `PlianceRule`: Complete rule (selector + relations + action + confidence)
- `PlianceEngine`: Manages rule collection, application, validation

**Process**:
1. Emit provisional rules from analysis (automated, LLM, or manual)
2. Apply rules to grids, logging violations
3. Validate rules against training pairs (accuracy measurement)
4. Repair rules based on violations and conflicts
5. Export rule metrics (accuracy, provenance, confidence distribution)

**Output**:
```python
{
    'rules': [
        {
            'rule_id': 'rule_0',
            'name': 'recolor_rectangles',
            'selector': {'shape_type': 'rectangle', 'color': 8},
            'action': {'action_type': 'recolor', 'parameters': {'color': 3}},
            'confidence': 0.85,
            'provenance': 'automated'
        }
    ],
    'violation_log': [...],
    'metrics': {
        'total_rules': 15,
        'enabled_rules': 12,
        'by_confidence': {'high': 5, 'medium': 7, 'low': 3}
    }
}
```

**Configuration**:
- No runtime flags; rules managed programmatically

---

### 4. LLM Integration (`llm_*.py`)

**Purpose**: Provide LLM-powered meta-reasoning with fallback support.

**Key Modules**:
- `llm_interface.py`: Interface to local small LLMs (Phi-3, Qwen, Llama)
- `llm_adapters.py`: Convert internal data → LLM-friendly JSON
- `llm_prompts.py`: Specialized prompt templates
- `llm_cache.py`: Cache LLM outputs by task signature + state
- `llm_fallbacks.py`: Heuristic fallbacks when LLM unavailable

**Process**:
1. Adapt inventory, conflicts, repairs → LLM context
2. Generate specialized prompts (inventory analysis, rule synthesis, conflict resolution, search prioritization)
3. Call LLM (or fallback to heuristics)
4. Cache results keyed by task signature + tracking state
5. Parse JSON responses into actionable results

**Key Prompts**:
- **Inventory Analysis**: Identify patterns, suggest transformation rules
- **Rule Synthesis**: Generate pliance rules from patterns
- **Conflict Resolution**: Choose resolution strategy (choose_best, conditional, merge, disable)
- **Search Prioritization**: Weight search methods, suggest operations
- **Repair Tasks**: Recommend fixes for failing rules

**Output Example** (Search Prioritization):
```python
{
    'primary_strategy': 'pliance_rules',
    'method_priorities': {
        'episodic_memory': 0.6,
        'pliance_rules': 0.9,
        'beam_search': 0.7,
        'neural_guidance': 0.5
    },
    'suggested_operations': ['recolor', 'extract', 'fill_region'],
    'confidence': 0.85
}
```

**Configuration**:
```python
llm_config = {
    'use_llm_reasoning': True,  # Enable LLM meta-reasoner
    'llm_model_name': 'microsoft/Phi-3-mini-4k-instruct',
    'llm_temperature': 0.3,
    'llm_max_tokens': 512,
    'cache_dir': '.llm_cache'
}
```

**Fallback Behavior**:
- If LLM fails or disabled: `FallbackReasoner` uses heuristics
  - Most common transformation → highest priority
  - Evidence count → conflict resolution
  - Default priorities for search methods

---

### 5. Search Fusion (`search_fusion.py`)

**Purpose**: Integrate LLM meta-reasoning with EnhancedSearch.

**Key Classes**:
- `SearchPriorityWeighter`: Applies LLM priorities to weight candidates
- `OperationSeeder`: Seeds beam/heuristic search with LLM-suggested operations
- `PlianceRuleFusion`: Converts pliance rules → programs, injects into episodic memory
- `AdaptiveResearcher`: Triggers re-search after rule amendments

**Process**:
1. **Priority Weighting**: Weight candidate sources by LLM priorities
   - Episodic memory: 0.6 → top 60% priority
   - Pliance rules: 0.9 → top 90% priority
   - Beam search: 0.7 → top 70% priority
   - Sample candidates proportional to weights

2. **Operation Seeding**: Generate seed programs from LLM suggestions
   - Single-step programs from top 10 suggested operations
   - Two-step compositions from top 5
   - Apply parameter hints where available

3. **Pliance Fusion**: Feed validated rules into search
   - Convert high-confidence rules (≥0.7) to DSL programs
   - Add to candidate pool
   - Inject into episodic memory for future retrieval

4. **Adaptive Re-search**: Decide when to re-search
   - Conflicts resolved since last search
   - New objects discovered (inventory grew)
   - Last search yielded no candidates

**Output**:
- Weighted and merged candidate programs ready for EnhancedSearch

**Configuration**:
```python
fusion_config = {
    'use_priorities': True,  # Apply LLM priorities
    'use_operation_seeding': True,  # Seed with LLM ops
    'use_pliance_fusion': True,  # Inject pliance rules
    'use_adaptive_research': True  # Enable re-search logic
}
```

---

### 6. Enhanced Search (`enhanced_search.py`)

**Purpose**: Final program synthesis with all techniques integrated.

**Process** (with fusion):
1. **Human-Grade Reasoning**: Spatial/template reasoning (highest priority)
2. **Memory Candidates**: From comprehensive memory + episodic retrieval
3. **Pliance Rules** (if fusion enabled): Converted to programs, weighted by confidence
4. **Facts-Guided Search**: RFT-based heuristics
5. **Heuristic Candidates**: Single-step programs
6. **Beam Search** (if seeded): Initialized with LLM-suggested operations
7. **MCTS**: If still limited candidates
8. **Neural Guidance**: If still limited
9. **Sketch-Based Search**: If still limited
10. **Priority Weighting** (if enabled): Re-rank all candidates by LLM priorities
11. **Test-Time Adaptation**: Adapt candidates to training pairs
12. **Selection**: Score, deduplicate, select top programs

**Integration Point**:
```python
# In synthesize_enhanced():
if meta_reasoning_result:
    # Apply priority weighting
    weighter = SearchPriorityWeighter()
    base_candidates = {
        'human_reasoning': human_candidates,
        'episodic_memory': memory_candidates,
        'pliance_rules': pliance_candidates,
        'beam_search': beam_programs,
        ...
    }
    weighted = weighter.apply_priorities(base_candidates, meta_reasoning_result.search_priorities)
    all_candidates = weighter.sample_by_priority(weighted, max_programs)

    # Seed beam search
    seeder = OperationSeeder()
    beam_seeds = seeder.seed_beam_search(
        meta_reasoning_result.suggested_operations,
        meta_reasoning_result.operation_parameters,
        train_pairs
    )
```

---

## Complete Example Workflow

### Input
```python
train_pairs = [
    (input_grid_0, output_grid_0),
    (input_grid_1, output_grid_1),
    (input_grid_2, output_grid_2)
]

llm_config = {
    'use_llm_reasoning': True,
    'llm_model_name': 'microsoft/Phi-3-mini-4k-instruct'
}

runtime_flags = {
    'rft_first': True,
    'use_tracking': True,
    'use_pliance_fusion': True
}
```

### Execution

```python
from arc_solver import ARCSolver
from arc_solver.rft_tracking import RFTTracker, TaskTrackingCache
from arc_solver.pliance_engine import PlianceEngine
from arc_solver.search_fusion import integrate_search_fusion

# 1. Initialize solver with LLM config
solver = ARCSolver(llm_config=llm_config, runtime_flags=runtime_flags)

# 2. Initialize tracking
task_id = "task_abc123"
cache = TaskTrackingCache()

# Try to load cached state
tracker = cache.load_tracking_state(task_id)
if tracker is None:
    # Build new tracking state
    tracker = RFTTracker(task_id)
    tracker.ingest_training_pairs(train_pairs)
    cache.save_tracking_state(tracker)

# 3. Initialize pliance engine
engine = PlianceEngine()

# Emit provisional rules from tracking
for pattern in tracker.inventory.get_rule_friendly_schema()['patterns']:
    if pattern['type'] == 'consistent_transformation':
        # Auto-generate rule
        rule = engine.emit_provisional_rule(
            name=f"auto_{pattern['transformation']}",
            selector=ObjectSelector(tags={'auto_detected'}),
            action=RuleAction(action_type=pattern['transformation']),
            confidence=0.7,
            provenance='automated'
        )

# 4. LLM meta-reasoning (integrated in solver.solve())
# Happens inside EnhancedSearch.synthesize_enhanced()

# 5. Solve with fusion
programs = solver.solve(train_pairs, test_input=test_grid)

# 6. Apply best program
if programs:
    prediction = apply_program(programs[0], test_grid)
```

### Output Flow

1. **Object Inventory**: 15 objects tracked, 3 patterns detected
2. **Tracking**: 2 conflicts found, 3 repair tasks queued (priority: 0.7, 0.6, 0.4)
3. **Pliance Rules**: 5 provisional rules emitted (3 automated, 2 from LLM)
4. **LLM Reasoning**:
   - Strategy: `pliance_rules`
   - Priorities: `{pliance_rules: 0.9, beam_search: 0.7, episodic: 0.6}`
   - Suggested ops: `['recolor', 'extract', 'fill_region']`
5. **Search Fusion**:
   - 12 pliance candidates (weighted 0.9)
   - 8 beam seeds from LLM ops
   - 25 episodic candidates (weighted 0.6)
   - Total: 45 candidates → sampled to top 30 by priority
6. **Enhanced Search**: Final 256 programs, best score: 1.000

---

## Configuration Flags Reference

### Runtime Flags (`runtime_flags`)

```python
runtime_flags = {
    'rft_first': True,          # Prioritize RFT/pliance over heuristics
    'use_tracking': True,       # Enable conflict detection + repair queue
    'use_pliance_fusion': True, # Inject pliance rules into search
    'fallback_policy': 'standard'  # 'standard' | 'aggressive' | 'conservative'
}
```

### LLM Config (`llm_config`)

```python
llm_config = {
    'enabled': True,
    'model_path': '/path/to/phi-3-mini.gguf',  # Optional: local GGUF model
    'use_llm_reasoning': True,   # Enable meta-reasoner
    'llm_model_name': 'microsoft/Phi-3-mini-4k-instruct',
    'llm_temperature': 0.3,      # Lower = more deterministic
    'llm_max_tokens': 512,
    'cache_dir': '.llm_cache'
}
```

### Solver Initialization

```python
solver = ARCSolver(
    llm_config=llm_config,
    runtime_flags=runtime_flags,
    use_rft=True,           # Enable RFT engine
    enable_beam_search=True # Enable beam search in EnhancedSearch
)
```

---

## Cache & Persistence

### Tracking Cache
- **Location**: `.arc_tracking_cache/`
- **Files**:
  - `{task_id}.json`: Tracking state (conflicts, repairs)
  - `{task_id}_inventory.json`: Object inventory
- **Access**: `TaskTrackingCache.load_tracking_state(task_id)`

### LLM Cache
- **Location**: `.llm_cache/`
- **Files**:
  - `cache_index.json`: Cache metadata
  - `{cache_key}.pkl`: Cached LLM outputs
- **Keys**: `{prompt_type}:task:{task_sig}:state:{state_sig}`
- **Eviction**: LRU when `max_entries` reached (default: 1000)

### Episodic Memory
- **Location**: `episodes.json`
- **Content**: Successful solutions with metadata (including pliance rules)

---

## Debugging & Inspection

### View Tracking State
```python
tracker = RFTTracker('task_abc123')
tracker.ingest_training_pairs(train_pairs)

# Export for inspection
state = tracker.export_for_llm()
print(json.dumps(state, indent=2))
```

### View Pliance Rules
```python
engine = PlianceEngine()
# ... add rules ...

metrics = engine.get_rule_metrics()
print(f"Total rules: {metrics['total_rules']}")
print(f"High confidence: {metrics['by_confidence']['high']}")

# Validate against training pairs
results = engine.validate_rules(train_pairs)
for rule_id, result in results.items():
    print(f"{rule_id}: {result['accuracy']:.2f} accuracy")
```

### View LLM Cache Stats
```python
from arc_solver.llm_cache import LLMCache

cache = LLMCache()
stats = cache.get_stats()
print(f"Cache entries: {stats['total_entries']}")
print(f"Hit rate: {stats['hit_rate']:.2f}")
print(f"Cache size: {stats['cache_size_mb']:.2f} MB")
```

---

## Performance Tuning

### For Speed
```python
llm_config = {
    'enabled': False  # Disable LLM, use fallbacks only
}

runtime_flags = {
    'rft_first': False,  # Heuristics first
    'use_tracking': False
}

solver = ARCSolver(
    llm_config=llm_config,
    runtime_flags=runtime_flags,
    enable_beam_search=False  # Skip beam search
)
```

### For Accuracy
```python
llm_config = {
    'enabled': True,
    'llm_temperature': 0.1,  # More deterministic
    'llm_max_tokens': 1024   # Longer reasoning
}

runtime_flags = {
    'rft_first': True,
    'use_tracking': True,
    'use_pliance_fusion': True
}

solver = ARCSolver(
    llm_config=llm_config,
    runtime_flags=runtime_flags,
    enable_beam_search=True
)

# More candidates
programs = solver.solve(train_pairs, max_programs=512)
```

---

## Module Ownership

| Module | Owner | Purpose |
|--------|-------|---------|
| `object_inventory.py` | Object Tracking | Persistent object IDs + deltas |
| `rft_tracking.py` | Tracking | Conflict detection + repair queue |
| `pliance_engine.py` | Rules | Rule definition + application |
| `llm_*.py` | LLM Integration | Meta-reasoning + fallbacks |
| `search_fusion.py` | Search Fusion | Priority weighting + seeding |
| `enhanced_search.py` | Search | Final program synthesis |

---

## Next Steps

1. **Testing**: See `tests/` for unit/integration tests
2. **Validation**: Run `scripts/validate_arc.py` on curated subset
3. **Contribution**: See `CONTRIBUTING.md` for guidelines
