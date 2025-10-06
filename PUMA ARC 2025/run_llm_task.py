
import json
import sys
import numpy as np
from pathlib import Path

# Add the PUMA directory to the path to allow imports
puma_path = str(Path(__file__).parent / "PUMA")
if puma_path not in sys.path:
    sys.path.insert(0, puma_path)

from arc_solver.solver import ARCSolver
from arc_solver.grid import to_list

def main():
    """
    Loads the first evaluation challenge, runs the solver with the LLM enabled,
    and saves the output to a file to ensure a clean result.
    """
    output_file = Path(__file__).parent / "llm_output.json"

    try:
        # --- 1. Load the Challenge Data ---
        challenges_path = Path(__file__).parent / "PUMA" / "data" / "arc-agi_evaluation_challenges.json"
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)
        
        first_task_id = list(challenges.keys())[0]
        task_data = challenges[first_task_id]
        task_data['task_id'] = first_task_id

        # --- 2. Configure and Run the Solver ---
        # This is the configuration to enable the LLM
        llm_config = {
            'use_llm_reasoning': True
        }

        # Initialize the solver with LLM config and logging disabled
        solver = ARCSolver(
            use_enhancements=True, 
            enable_logging=False,
            llm_config=llm_config
        )
        
        result = solver.solve_task(task_data)

        # --- 3. Save the Output Grid to a File ---
        if result and "attempt_1" in result and result["attempt_1"]:
            output_grid = result["attempt_1"][0]
            with open(output_file, 'w') as f:
                json.dump(output_grid, f)
        else:
            error_message = {"error": "Solver did not produce a valid output."}
            with open(output_file, 'w') as f:
                json.dump(error_message, f)

    except Exception as e:
        import traceback
        error_info = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        with open(output_file, 'w') as f:
            json.dump(error_info, f)

if __name__ == "__main__":
    main()
