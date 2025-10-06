#!/usr/bin/env bash
# Evaluation script using a specified solver from the registry
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
PY="${PY:-python3}"
BATCH="${BATCH:-50}"
SOLVER="${SOLVER:-default}"
OUT="${OUT:-submission/${SOLVER}_submission.json}"
LOGDIR="$ROOT/runlogs"
mkdir -p "$LOGDIR" "$(dirname "$OUT")"

echo "Evaluating with solver: $SOLVER"
echo "Output file: $OUT"

# Memory-friendly defaults
export PYTHONUNBUFFERED=1 PYTHONMALLOC=malloc MALLOC_ARENA_MAX=2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Ensure sitecustomize.py exists (float32 + trim)
if [[ ! -f "$ROOT/sitecustomize.py" ]]; then
  cat > "$ROOT/sitecustomize.py" <<'PY'
import os, atexit, ctypes, numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
_orig = np.random.Generator.standard_normal
def _stdnorm(self, size=None, dtype=np.float32, out=None):  # default float32
    return _orig(self, size=size, dtype=dtype, out=out)
np.random.Generator.standard_normal = _stdnorm
try:
    libc = ctypes.CDLL("libc.so.6")
    atexit.register(lambda: libc.malloc_trim(0))
except Exception:
    pass
PY
fi

# Chunked submission using solver registry
"$PY" - "$BATCH" "$OUT" "$SOLVER" <<'PY'
import json, os, sys, time, traceback
from pathlib import Path

BATCH = int(sys.argv[1])
OUT = sys.argv[2]
SOLVER = sys.argv[3] if len(sys.argv) > 3 else "default"
ROOT = Path(os.getcwd())
sys.path.append(str(ROOT))

# Import solver registry
from arc_solver.registry import get_solver

def loadj(p): 
    with open(p,"r") as f: return json.load(f)

eval_ch = loadj(ROOT/"data/arc-agi_evaluation_challenges.json")

# Build {task_id: task_obj}
E = {}
if isinstance(eval_ch, list):
    for it in eval_ch:
        tid = it.get("task_id") or it.get("id")
        if tid is not None: E[str(tid)] = it
elif isinstance(eval_ch, dict):
    for k,v in eval_ch.items():
        E[str(k)] = v

print(f"Loaded {len(E)} tasks")

# Get solver function
try:
    solve_func = get_solver(SOLVER)
    print(f"Using solver: {SOLVER}")
except Exception as e:
    print(f"Error loading solver '{SOLVER}': {e}")
    print("Falling back to default solver")
    from arc_solver.solver import solve_task
    solve_func = solve_task

ids = sorted(E.keys())
chunks = [ids[i:i+BATCH] for i in range(0, len(ids), BATCH)]
all_preds = []
start = time.time()

for ci, chunk in enumerate(chunks, 1):
    t0 = time.time()
    ok = 0
    for tid in chunk:
        task = E[tid]
        try:
            result = solve_func(task)
            
            # Handle different return formats
            if isinstance(result, dict):
                # Registry solvers return {"attempt_1": [...], "attempt_2": [...]}
                pred = result.get("attempt_1", [])
            elif isinstance(result, list):
                # Legacy solvers return list directly
                pred = result
            else:
                pred = []
            
            # Ensure proper format for submission
            if pred and isinstance(pred, list):
                all_preds.append({"task_id": tid, "outputs": pred})
                ok += 1
            else:
                all_preds.append({"task_id": tid, "outputs": []})
                
        except Exception as e:
            print(f"Error solving task {tid}: {e}")
            traceback.print_exc()
            # record empty prediction on error to keep submission shape stable
            all_preds.append({"task_id": tid, "outputs": []})
    
    dt = time.time()-t0
    print(f"[chunk {ci}/{len(chunks)}] solved {ok}/{len(chunk)} in {dt:.1f}s", flush=True)

# Write final submission
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(all_preds, f)
print(f"Wrote {OUT} with {len(all_preds)} items in {time.time()-start:.1f}s", flush=True)
PY

# Score against public eval solutions (if present)
if [[ -f data/arc-agi_evaluation_solutions.json ]]; then
  "$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path

OUT = sys.argv[1]
sub = json.load(open(OUT))
sol = json.load(open("data/arc-agi_evaluation_solutions.json"))

def norm(grids):
    if grids and isinstance(grids[0], (list,tuple)) and grids and isinstance(grids[0][0], (list,tuple)):
        if all(isinstance(r,(list,tuple)) and r and isinstance(r[0],(int,float)) for r in grids):
            grids = [grids]
    return grids

pred = {}
if isinstance(sub, list):
    for it in sub:
        tid = str(it.get("task_id") or it.get("id"))
        out = it.get("outputs") or it.get("output")
        if tid is not None and out is not None:
            pred[tid] = norm(out)

gt = {}
if isinstance(sol, list):
    for it in sol:
        tid = str(it.get("task_id") or it.get("id"))
        out = it.get("solutions") or it.get("outputs") or it.get("solution")
        if tid is not None and out is not None:
            gt[tid] = norm(out)
elif isinstance(sol, dict):
    for k,v in sol.items():
        gt[str(k)] = norm(v)

ids = sorted(set(pred) & set(gt))
ok = 0
for tid in ids:
    p, g = pred[tid], gt[tid]
    if len(p)==len(g) and all(pp==gg for pp,gg in zip(p,g)):
        ok += 1
total = len(ids)
pct = (ok/total*100.0) if total else 0.0
print(f"EVAL SCORE ({Path(OUT).stem}): {ok}/{total} = {pct:.2f}%")
PY
else
  echo "Note: public solutions not found at data/arc-agi_evaluation_solutions.json; skipping score."
fi

echo "Submission completed: $OUT"