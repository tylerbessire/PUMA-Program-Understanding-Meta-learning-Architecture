"""CLI for remediating ARC failures via the neuro-symbolic pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc_solver.failure_replay import remediate_failures


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate programs for failed ARC tasks")
    parser.add_argument("failure_log", type=_path, help="Path to failure log JSONL")
    parser.add_argument("challenges", type=_path, help="Path to ARC challenges JSON")
    parser.add_argument("program_bank", type=_path, help="Directory for storing induced programs")
    parser.add_argument("--synthetic-count", type=int, default=200, help="Number of synthetic examples to generate")
    parser.add_argument("--synthetic-seed", type=int, default=0, help="Seed for synthetic data generator")
    parser.add_argument("--perception-epochs", type=int, default=3, help="Epochs for perception model training")

    args = parser.parse_args()

    stats = remediate_failures(
        failure_log_path=args.failure_log,
        challenges_path=args.challenges,
        program_bank_path=args.program_bank,
        synthetic_count=args.synthetic_count,
        synthetic_seed=args.synthetic_seed,
        perception_epochs=args.perception_epochs,
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
