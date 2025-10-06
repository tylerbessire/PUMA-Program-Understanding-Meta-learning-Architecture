# tools/verify_layout.py
from pathlib import Path

REQUIRED = [
  "arc_solver/__init__.py",
  "arc_solver/dsl.py", "arc_solver/grid.py", "arc_solver/heuristics.py", "arc_solver/objects.py",
  "arc_solver/search.py", "arc_solver/solver.py", "arc_solver/io_utils.py",
  "arc_solver/features.py", "arc_solver/neural/guidance.py",
  "arc_solver/neural/sketches.py", "arc_solver/neural/episodic.py",
  "arc_solver/ttt.py",
  "tools/train_guidance.py", "tools/mine_sketches.py", "tools/build_memory.py", "tools/benchmark.py",
  "models", "data", "notebooks", "scripts", "submission",
  "arc_submit.py", "requirements.txt",
  "README.md", "docs/architecture.md", "docs/contributing.md",
  "tests/test_submission_schema.py", "tests/test_solver_end2end.py", "tests/test_guidance.py",
  "tests/test_ttt.py", "tests/test_memory.py", "tests/test_dsl_ops.py", "tests/test_features.py",
  ".github/workflows/ci.yml"
]

missing = [p for p in REQUIRED if not Path(p).exists()]
if missing:
    print("❌ Missing paths:\n  " + "\n  ".join(missing))
    raise SystemExit(1)
print("✅ PUMA layout matches spec.")
