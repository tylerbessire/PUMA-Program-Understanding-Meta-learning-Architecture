# AGENTS.md - PUMA Development Roadmap

## Mission
Develop a competition-ready ARC solver with measurable progress at each milestone.

## Ethos
1. Work sequentially by milestone.
2. After completing a task, record a progress marker.
3. Run tests and document results.
4. Keep implementations production-grade; no partial stubs.

---

## Phase Plan

### M0 — Repo health (1 day)
Goal: Verify local environment and baseline runnability.
Checklist:
- Makefile targets run locally.
- CI green on mock data; `pytest` passes.
- `arc_submit.py` produces a submission JSON.

### M1 — Data foundation (1–2 days)
Goal: Build canonicalised training dataset.
Checklist:
- `prep_build_dataset.py` fills `train_X.npy` and `train_Y.npy`.
- Canonicalisation verified by `test_canonical.py` (rotations/reflections hash-equal).

### M2 — Baseline guidance (1–2 days)
Goal: Train neural guidance to cut search.
Checklist:
- `NeuralGuidance` reaches ≥ micro-F1 0.55@top-k.
- `integrate_stack.py` reduces node expansions ≥30% vs unguided.

### M3 — Facts & relational sketches (2–3 days)
Goal: Mine facts and program sketches.
Checklist:
- `facts.jsonl` coverage ≥95%; schema frozen.
- `sketches.json` mined; top-20 macros explain ≥60% programs.

### M4 — Episodic memory online (1 day)
Goal: Retrieval speeds up solving.
Checklist:
- `episodes.json` built; retrieval hit-rate ≥40%; solve time ↓ ≥20%.

### M5 — Full stack solve (2 days)
Goal: Enhanced solver > baseline by 8–12% absolute.
Checklist:
- Diversity-2 attempts comply with ARC rules and improve pass@2.

### M6 — Test-time adaptation (2 days)
Goal: TTT improves borderline tasks.
Checklist:
- `adapt_test_time.py` improves mini eval ≥3% with runtime ≤30s median.

### M7 — Public eval harness (ongoing)
Goal: Nightly evaluation tooling.
Checklist:
- `scripts/eval_public.sh` and `tools/benchmark.py` produce reports with timing and failures.

---

## Progress Ledger
Record completion as:

```
[X] Milestone: short description
    Date: YYYY-MM-DD
    Test Result: command + outcome
    Notes: details
```

### Completed
```
[X] M1: D4 canonicalisation validated
    Date: 2025-09-14
    Test Result: pytest tests/test_canonical.py -q
    Notes: Property-based invariance checks for D4 symmetries and colour relabeling
```

---

## Working Protocol
1. Work on one milestone at a time.
2. Validate each checklist item with tests or benchmarks.
3. Update the progress ledger after validation.
4. If regressions occur, halt and resolve before proceeding.

**Success Criterion**: 80%+ accuracy on ARC evaluation set with clear reasoning traces and adaptive behaviour.

Start with M0.
