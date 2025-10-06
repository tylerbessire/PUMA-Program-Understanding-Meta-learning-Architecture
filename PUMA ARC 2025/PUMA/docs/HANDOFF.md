Public Eval Runner — Handoff

Overview

Adds a one-command public evaluation runner that executes the solver across the ARC public evaluation set in chunks, writes a valid submission JSON, and (optionally) scores it against public solutions if available.
•Primary entrypoint: scripts/eval_public.sh
•Optional convenience: make eval_public
•Memory-safe defaults (thread caps, float32 guidance via sitecustomize.py, process-friendly allocator env)
•Logs and artifacts are created deterministically

Assumptions
•data/arc-agi_evaluation_challenges.json exists (required to run).
•data/arc-agi_evaluation_solutions.json exists only if local scoring is desired.
•Repository root is the working dir when running the script/Makefile.

Files Added
•scripts/eval_public.sh – chunked runner + optional scoring.
•(On first run) sitecustomize.py is created at repo root to enforce float32 guidance init and thread caps.

Note: No changes to solver code are required for this runner.

How to Run

Direct (bash)

bash scripts/eval_public.sh

With knobs

BATCH=40 OUT=submission/eval_public.json bash scripts/eval_public.sh

Makefile (optional)

make eval_public                 # uses defaults
make eval_public BATCH=40 OUT=submission/eval_public.json

Outputs & Logs
•Submission JSON: ${OUT} (default: submission/full_submission.json)
•Console prints per-chunk progress:

[chunk 1/8] solved 50/50 in 27.3s
...
Wrote submission/full_submission.json with 400 items in 214.8s


•If data/arc-agi_evaluation_solutions.json is present, prints:

EVAL SCORE (public): N/M = P%


Feature Flags / Config
•BATCH (default 50): number of tasks per chunk; lower if memory is tight.
•OUT (default submission/full_submission.json): path to final submission.
•Env set by the script for stability:
•PYTHONUNBUFFERED=1, PYTHONMALLOC=malloc, MALLOC_ARENA_MAX=2
•OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1
•sitecustomize.py (auto-created if missing) sets NumPy random normals to float32 and trims malloc arenas on exit.

Rollback
•Remove scripts/eval_public.sh and any Makefile target that references it.
•Delete sitecustomize.py if you don’t want repo-local Python customization.
•No other files are touched.

Acceptance Criteria
•AC-1: scripts/eval_public.sh completes on Colab/T4 or A100 with default BATCH=50 without OOM.
•AC-2: Produces a submission with one entry per eval task at ${OUT}.
•AC-3: If public solutions exist, prints a final score line EVAL SCORE (public): N/M = P%.
•AC-4: Per-chunk progress is visible in stdout.

Troubleshooting

SymptomLikely CauseFix
MemoryError / process killed mid-runSearch/guidance RAM spikeLower BATCH (e.g., 30), ensure sitecustomize.py exists (script auto-creates), keep thread envs at 1
Hanging with no outputBuffered child outputRunner uses unbuffered flags; re-run the script (don’t call solver directly)
SystemError: ufunc equalHuge boolean temp in equality checkHarmless as a one-off; usually disappears when BATCH is lowered; longer-term fix: bytewise equality in solver
Submission has fewer than expected itemsException while solving a taskRunner records empty outputs on error to keep shape; inspect solver logs for specific failures

Security / Cleanroom Notes
•The runner does not train or tune on evaluation solutions.
•Local scoring (if solutions are present) is for a single post-run sanity check; avoid iterative tuning on eval to preserve generalization.

Next Steps (separate PRs)
•P0: guidance reuse per process, op pre-validation (translate, pad) to avoid exceptions.
•P1: symmetry-canonical hashing & invariant filters.
•P2: hard-negative mining for guidance retrain.

⸻

Quick Sanity Commands

# default run
bash scripts/eval_public.sh

# tighter memory profile
BATCH=30 bash scripts/eval_public.sh

# custom output path
OUT=submission/public_eval_YYYYMMDD.json bash scripts/eval_public.sh

That’s it—this gives Codex and the Colab teammate a reliable “press go” for public eval, plus a clean rollback and clear acceptance criteria.
