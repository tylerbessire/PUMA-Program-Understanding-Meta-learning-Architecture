import json
import numpy as np
import sys
from pathlib import Path

# Add the PUMA directory to the path to allow imports from arc_solver
sys.path.insert(0, str(Path(__file__).parent / 'PUMA'))
from arc_solver.arc_dsl_lib import (
    Task, Grid, Object, compose, map_objects, paint, reflect, copy, get_objects, find_objects, bounding_box,
    copy_grid, deepcopy, filter_color, paint_diagonal, fill_largest_object, paint_grid
)

def execute_solution(dsl_string, task_data):
    """Executes a dsl_program string on a task with extensive debugging.""" 
    print("--- Creating Task Object ---")
    task_obj = Task(task_data)
    print(f"task_obj created. Type: {type(task_obj)}")

    exec_globals = {
        "np": np,
        "task": task_obj,
        "Grid": Grid,
        "Object": Object,
        "compose": compose,
        "find_objects": find_objects,
        "get_objects": get_objects,
        "bounding_box": bounding_box,
        "copy_grid": copy_grid,
        "deepcopy": deepcopy,
        "filter_color": filter_color,
        "paint_diagonal": paint_diagonal,
        "fill_largest_object": fill_largest_object,
        "paint_grid": paint_grid,
        "reflect": reflect,
        "copy": copy,
        "map_objects": map_objects,
        "paint": paint,
    }
    
    print("--- Executing DSL Program ---")
    exec(dsl_string, exec_globals)
    solve_func = exec_globals.get('solve')
    
    if not solve_func:
        raise ValueError("No solve() function found in the dsl_program.")

    print("--- Calling solve_func(task_obj) ---")
    predicted_output = solve_func(task_obj)
    print("--- solve_func returned ---")
    
    return predicted_output

def debug_single_solution():
    task_id_to_debug = '007bbfb7' # Known to fail with a different error

    log_file = Path('PUMA ARC 2025/artifacts/master_solution_log.jsonl')
    challenges_file = Path('PUMA ARC 2025/PUMA/data/arc-agi_training_challenges.json')

    print(f"--- Debugging Task: {task_id_to_debug} ---")

    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    
    dsl_program = None
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('task_id') == task_id_to_debug:
                    dsl_program = data.get('dsl_program')
                    break
            except json.JSONDecodeError:
                continue

    if not dsl_program:
        print(f"Could not find DSL program for task {task_id_to_debug}")
        return

    task_data = challenges[task_id_to_debug]
    task_data['task_id'] = task_id_to_debug

    try:
        print("--- Starting Execution ---")
        predicted_output = execute_solution(dsl_program, task_data)
        print("--- Execution Successful ---")
        print("Output:", predicted_output)
    except Exception as e:
        print(f"\n--- !!! EXECUTION FAILED !!! ---")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_single_solution()
