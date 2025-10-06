# PUMA RFT Module

[S:DOC v2] scope=puma.rft overview=complete pass

The Relational Frame Theory (RFT) module adds pliance-based rule following and
tracking-driven rule variation to the PUMA solver while remaining fully
feature-flagged behind `PUMA_RFT`.

## Architecture

```
+-------------------+      +----------------------+      +------------------+
| feature_flags     | ---> | pliance (determinism)| ---> | explain          |
+-------------------+      +----------------------+      +------------------+
           |                         |                               ^
           v                         v                               |
+-------------------+      +----------------------+      +------------------+
| demo/grid harness | ---> | tracking (variation) | ---> | orchestrator     |
+-------------------+      +----------------------+      +------------------+
```

* `types.py` exposes typed primitives (rules, context, hypotheses).
* `pliance.py` executes prioritized rules deterministically.
* `tracking.py` evaluates adaptive hypotheses and promotes successful ones.
* `orchestrator.py` alternates pliance and tracking with metrics and guards.
* `explain.py` records rich traces surfaced via `explain_last_run()`.

## Usage

1. Enable the feature flag: `export PUMA_RFT=1`.
2. Run the demo tests: `pytest tests/rft/test_rft_integration.py -q`.
3. Inspect the most recent trace:

   ```python
   from puma.rft.explain import explain_last_run
   print(explain_last_run())
   ```

To exercise the adapter inside the ARC solver:

```bash
PUMA_RFT=1 python -c 'from arc_solver import ARCSolver; solver = ARCSolver();
print(solver.solve_task({"train": [...], "test": [...], "metadata":
{"rft_demo": {"grid": {"grid": [[0,1,0]], "targets": [[0,0],[0,1],[0,2]],
"inject_output": true}}}}))'
```

### Demo promotion proof

Copy-paste the snippet below to verify a promotion and trace:

```bash
PUMA_RFT=1 python - <<'PY'
from puma.rft.demo.grid import build_context
from puma.rft.orchestrator import solve_with_rft
from puma.rft.tracking import HypothesisMemory
from puma.rft.feature_flags import build_limits
from puma.rft.explain import explain_last_run

grid = [
    [0, 1],
    [2, 1],
    [2, 1],
]
targets = [(r, 0) for r in range(len(grid))]
context, rules = build_context(grid, targets, build_limits())
memory = HypothesisMemory()
context, status = solve_with_rft(context, rules, memory, outer_budget=4)
print(status)
print(explain_last_run(context))
PY
```

Expected output (abridged):

```
DONE
Status: GOAL
Top hypotheses:
  - propagate 0 by (1,0) regardless of color (score=1.00)
Promoted rules:
  - track_1_0_broad (score=1.00)
Metrics: ... tracking_promotions=1 ...
```

## Observability

Structured debug logs are emitted under the `puma.rft.*` namespace. Metrics are
collected on the context (e.g., `pliance_rule_fires`, `tracking_hypotheses_evaluated`)
and surfaced via `explain_last_run`.

## Tracking score

Hypothesis tracking uses the explicit scoring rule described in the spec:

```
progress = satisfied_targets / total_targets
simplicity = len(active_conditions)
score = progress - 0.1 * simplicity
```

Scores greater than the configured threshold (`RFT_DEFAULT_LIMITS["THRESH"]`) are
promoted. Simplicity penalties ensure compact rules are favoured when progress is
equivalent.
