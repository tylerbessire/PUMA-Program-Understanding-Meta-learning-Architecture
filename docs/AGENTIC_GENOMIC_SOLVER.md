# PUMA Agentic + Genomic Solvers (Design & Drop-In Guide)

Add this file as docs/AGENTIC_GENOMIC_SOLVER.md (or drop into README.md if you want it front-and-center). It specifies two fully offline solvers you can plug into PUMA: Object-Agentic and Genomic (DNA). Both share utilities, expose a clean API, and run on CPU.

⸻

## Goals
- Win on compositional ARC tasks with interpretable, symbolic pipelines.
- Stay offline: pure Python + NumPy, no internet, no LLM calls at inference.
- Be pluggable: a minimal interface compatible with arc_solver.solver.solve_task.
- Falsify quickly: strong invariants + MDL (Minimum Description Length) bias to prune bad hypotheses fast.

⸻

## Repo layout (proposed)

```
PUMA/
├─ arc_solver/
│  ├─ solver.py                        # existing entrypoints
│  ├─ common/
│  │  ├─ grid.py                       # grid I/O, color palettes, hashing
│  │  ├─ objects.py                    # connected components, symmetry features
│  │  ├─ invariants.py                 # histograms, object-count deltas, symmetry flags
│  │  ├─ mdL.py                        # description-length utilities
│  │  └─ eval_utils.py                 # exact match, partial scores, logging
│  ├─ agents/                          # object-agentic solver
│  │  ├─ agentic_solver.py             # beam+blackboard referee (drop-in code goes here)
│  │  └─ ops.py                        # Op DSL: translate/reflect/recolor/duplicate/align...
│  ├─ genomic/                         # DNA-style solver
│  │  ├─ hilbert.py                    # 2D→1D space-filling order
│  │  ├─ tokenize.py                   # token + neighborhood signature + RLE
│  │  ├─ align.py                      # segment-aware alignment (NW/SW variant)
│  │  ├─ script.py                     # extract mutations, consensus, grid projection
│  │  └─ solver.py                     # GenomicSolver entrypoint
│  └─ registry.py                      # maps solver names to callables
├─ scripts/
│  ├─ eval_public.sh                   # runs the public eval with a chosen solver
│  └─ arc_submit.py                    # creates submission JSON (existing)
├─ tests/
│  ├─ test_agentic_small.py            # smoke tests & golden cases
│  └─ test_genomic_small.py
└─ docs/
   └─ AGENTIC_GENOMIC_SOLVER.md        # this file
```

⸻

## Unified solver interface

Both solvers should implement:

```python
# arc_solver/solver.py expects this signature (or adapt the registry).
def solve_task(task, solver_name: str = "baseline"):
    """
    task: dict with 'train': [{'input': grid, 'output': grid}, ...], 'test': [{'input': grid}, ...]
    Returns: list of predicted test grids
    """
```

Register them in arc_solver/registry.py:

```python
SOLVERS = {
    "agentic": "arc_solver.agents.agentic_solver:solve_task_agentic",
    "genomic": "arc_solver.genomic.solver:solve_task_genomic",
    # existing solvers...
}
```

⸻

## 1) Object-Agentic Solver (overview)

Idea: One agent per object. Agents propose local ops from a shared, interpretable DSL. A referee composes per-object proposals into a global program using a blackboard and a small beam search, pruned by invariants + MDL.

### Core pieces
- **Parsing**: connected_components(grid) -> List[Object]
- **Op DSL** (ops.py):
  - identity, translate(dy,dx), reflect(axis), recolor(a→b)
  - (next wave) duplicate(k, pattern), palette_permute(pi), align(row/col/edge/centroid)
- **Proposals**: propose_ops_for_object(obj, context) -> List[Op]
  Heuristic shortlist; later add a tiny learned prior (still offline).
- **Referee/Blackboard**:
  - Compose per-object ops, enforce no-overlap, canvas bounds, color budgets.
  - Invariants first: palette equality up to permutation, object-count deltas ∈ {−1,0,+1}, symmetry flags.
  - Score a candidate program over all train pairs: (exact_hits, -MDL, constraints_ok).

### Entry point

```python
# arc_solver/agents/agentic_solver.py
def solve_task_agentic(task, beam_width=64, max_depth=4):
    train_pairs = [(np.array(e["input"]), np.array(e["output"])) for e in task["train"]]
    best_prog, _score = beam_search_agentic(train_pairs, beam_width=beam_width)
    preds = []
    for t in task["test"]:
        gin = np.array(t["input"])
        pred = execute_program_on_grid(best_prog, gin)
        preds.append(pred if pred is not None else gin)  # conservative fallback
    return preds
```

**Why it works**: ARC is largely object-relational. Local proposals + a global referee nail "reflect each object, recolor uniformly, translate by rule, duplicate smallest along a row," etc., without neural nets.

⸻

## 2) Genomic (DNA-Style) Solver (overview)

Idea: Convert grids to sequences via a Hilbert curve, compress with RLE, then do segment-aware alignment between input/output to extract a mutational recipe (SUB/INS/DEL/DUP/INV/TRANSPOSE). Map mutations back to 2-D as concrete ops and apply the consensus script to the test grid.

### Pipeline
1. **Encode** (hilbert.py, tokenize.py):
   - hilbert_order(H,W) -> [(y,x), ...]
   - tokenize(grid) -> List[token_id] where each token encodes (color, 3x3 neighborhood signature or edge code)
   - rle(tokens) -> List[(token_id, run_len)]

2. **Align** (align.py):
   - Smith–Waterman / Needleman–Wunsch variant with run-aware scoring:
     - matches reward whole-run alignment,
     - substitutions cheap if color-only,
     - gaps penalized per run (encourages motif edits).

3. **Extract mutations** (script.py):
   - Merge adjacent edits into motifs.
   - Recognize patterns: DUP (regular repeats), INV (reversals), TRANSPOSE (big index jumps).

4. **Consensus** across train pairs:
   - Keep edits that apply to all examples; generalize color substitutions to palette permutations when consistent.

5. **Project back to grid**:
   - Map Hilbert spans → (y,x) regions; intersect with objects to move whole shapes.
   - Emit concrete ops: Recolor, Translate, Reflect, Duplicate, AlignRow/Col, PalettePermute.

6. **Score & pick**:
   - Choose the shortest script (MDL) that solves all trains exactly and respects invariants.

### Entry point

```python
# arc_solver/genomic/solver.py
def solve_task_genomic(task):
    train_pairs = [(np.array(e["input"]), np.array(e["output"])) for e in task["train"]]
    scripts = [infer_script(gin, gout) for gin, gout in train_pairs]
    recipe = consensus_script(scripts)
    preds = []
    for t in task["test"]:
        preds.append(apply_recipe(recipe, np.array(t["input"])))
    return preds
```

**Why it works**: Many ARC tasks are repeat/invert/move motifs. Sequence alignment finds the delta succinctly; back-projection turns it into interpretable grid ops.

⸻

## Shared invariants & MDL
- **Palette check**: multiset of colors equal up to permutation? If yes, prefer palette_permute over pixel edits.
- **Object-count delta**: ∈ {−1,0,+1} unless script explicitly includes DUP/DEL.
- **Symmetry**: detect vertical/horizontal symmetry to bias toward reflect.
- **MDL**: len(program) or weighted by op complexity (e.g., duplicate > recolor).

⸻

## Make targets & CLI

Add to Makefile:

```makefile
# Evaluate public ARC subset using the chosen solver
eval_agentic:
	python -m scripts.arc_submit --solver agentic --out submission/agentic_submission.json

eval_genomic:
	python -m scripts.arc_submit --solver genomic --out submission/genomic_submission.json
```

scripts/eval_public.sh gains:

```bash
#!/usr/bin/env bash
set -e
SOLVER="${1:-agentic}"   # or 'genomic'
OUT="submission/${SOLVER}_submission.json"
python -m scripts.arc_submit --solver "$SOLVER" --out "$OUT"
echo "Saved: $OUT"
```

⸻

## Minimal code contracts (snippets)

### Op DSL (agents/ops.py):

```python
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Op:
    kind: str                # 'identity','translate','reflect','recolor','duplicate','align','palette_permute'
    params: Tuple            # e.g., ('h',) or (dy,dx) or (from,to) or (k, pattern)

COST = {
    "identity": 0,
    "recolor": 1,
    "translate": 1,
    "reflect": 1,
    "align": 2,
    "duplicate": 3,
    "palette_permute": 1,
}
```

### Invariant gate (common/invariants.py):

```python
def palette_equiv(a, b):
    from collections import Counter
    return Counter(a.flatten()) == Counter(b.flatten())

def object_count_delta(a, b):
    from .objects import connected_components
    return len(connected_components(b)) - len(connected_components(a))
```

### MDL (common/mdL.py):

```python
def program_length(ops):
    from arc_solver.agents.ops import COST
    return sum(COST.get(op.kind, 2) for op in ops)
```

⸻

## Evaluation protocol
- **Dataset**: data/arc-agi_evaluation_challenges.json (+ ..._solutions.json for scoring).
- **Metric**: exact match per task; report also MDL and solve time.
- **A/B Plan**:
  1. Baseline (current).
  2. Agentic only.
  3. Genomic only.
  4. Hybrid: run both; if one fails trains or has higher MDL, choose the other.

**Log per-task**:

```
task_id, solver, train_exact, test_exact, mdl, time_ms, notes
```

⸻

## Roadmap (fast to slower)

### Day 1–2
- **Agentic**: identity/translate(±3)/reflect(h,v)/recolor, blackboard referee, invariants.
- **Genomic**: Hilbert, tokenization, RLE, simple NW alignment, SUB/DEL/INS script, palette-permute fast path.

### Week 1
- **Agentic**: duplicate + align row/col; symmetry grouping; beam priors.
- **Genomic**: segment merger → DUP/INV/TRANSPOSE; consensus across examples; grid projection with object snapping.

### Week 2
- **Shared**: small synthetic generator suite for unit tests.
- Add MDL-weighted beam and a hybrid "choose-best" meta-solver.

⸻

## Troubleshooting
- **Program executes to None (agentic)**: an object paste overlapped or went OOB → increase align options, or relax overlap if background.
- **Genomic INV vs reflect confusion**: disambiguate via 2-D IoU of candidate transforms; prefer lower-cost MDL.
- **Too many proposals (agentic)**: cap per-object ops (≤32) and translations (±3) until a prepass detects large moves.

⸻

## Why this combo is powerful
- **Agentic** nails object-wise relational rules with crisp constraints.
- **Genomic** discovers delta motifs and long-range repeats/inversions with minimal search.
- **Hybrid** lets the two cover each other's blind spots—and both stay offline, deterministic, and interpretable.

⸻

## Quick start

```bash
# choose one
make eval_agentic
# or
make eval_genomic
```

Then inspect submission/{agentic|genomic}_submission.json and per-task logs.

⸻

**Questions or edge cases you want encoded as golden tests? Toss a few task IDs and expected behaviors, and we'll lock them in under tests/.**