"""
Command-line entry point for generating ARC Prize submissions.

When run, this script loads the test challenge tasks, solves each one using
`arc_solver.solver.solve_task`, and writes a `submission.json` file with the
required two attempts per test input. The Kaggle platform picks up this file
as the submission when executing within a notebook environment.
"""

import json
from arc_solver.io_utils import load_rerun_json, save_submission
from arc_solver.solver import solve_task


def main() -> None:
    # Load all test tasks from the JSON file injected by Kaggle
    data = load_rerun_json()  # { task_id: {train:..., test:...}, ... }
    solutions = {}
    for task_id, task in data.items():
        result = solve_task(task)
        # Kaggle requires both attempts for every task id
        solutions[task_id] = {
            "attempt_1": result["attempt_1"],
            "attempt_2": result["attempt_2"],
        }
    path = save_submission(solutions, "submission.json")
    print(f"Saved {path} with {len(solutions)} tasks.")


if __name__ == "__main__":
    main()