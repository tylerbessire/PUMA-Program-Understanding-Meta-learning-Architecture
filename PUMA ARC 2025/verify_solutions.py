import json
import numpy as np
import sys
import re
from pathlib import Path

# Add the PUMA directory to the path to allow imports from arc_solver
sys.path.insert(0, str(Path(__file__).parent / 'PUMA'))
from arc_solver.arc_dsl_lib import (
    Task, Grid, Object, compose, map_objects, paint, reflect, copy, get_objects, find_objects, bounding_box,
    copy_grid, deepcopy, filter_color, paint_diagonal, fill_largest_object, paint_grid, translate, crop, row, extend_lines, fill_rectangle, paint_object
)

def execute_solution(dsl_string, task_data):
    """Executes a dsl_program string on a task."""
    task_obj = Task(task_data)
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
        "translate": translate,
        "crop": crop,
        "row": row,
        "extend_lines": extend_lines,
        "fill_rectangle": fill_rectangle,
        "paint_object": paint_object,
    }
    
    exec(dsl_string, exec_globals)
    solve_func = exec_globals.get('solve')
    
    if not solve_func:
        raise ValueError("No solve() function found in the dsl_program.")

    predicted_output = solve_func(task_obj)
    
    return predicted_output

def find_usage_examples(log_file, method_name, max_examples=3):
    examples = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                dsl_program = data.get("dsl_program")
                if dsl_program and (f".{method_name}(" in dsl_program or f"{method_name}(" in dsl_program):
                    examples.append(dsl_program)
                    if len(examples) >= max_examples:
                        break
            except json.JSONDecodeError:
                continue
    return examples

def verify_solutions():
    """Grades the generated solutions and creates the final Program Bank."""
    log_file = Path('PUMA ARC 2025/artifacts/master_solution_log.jsonl')
    challenges_file = Path('PUMA ARC 2025/PUMA/data/arc-agi_training_challenges.json')
    solutions_file = Path('PUMA ARC 2025/PUMA/data/arc-agi_training_solutions.json')
    program_bank_file = Path('program_bank.json')

    if not log_file.exists() or not solutions_file.exists() or not challenges_file.exists():
        print(f"Error: Make sure log, solutions, and challenges files are in the correct directories.")
        return

    print("Loading generated solutions, challenges, and official solutions...")
    with open(challenges_file, 'r') as f:
        challenges = json.load(f)
    with open(solutions_file, 'r') as f:
        official_solutions = json.load(f)

    generated_solutions = []
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get('status') == 'generated':
                    generated_solutions.append(data)
            except json.JSONDecodeError:
                continue

    print(f"Found {len(generated_solutions)} generated solutions to verify.")

    verified_program_bank = {}
    verified_success_count = 0
    verified_failure_count = 0
    execution_errors = 0

    for i, solution in enumerate(generated_solutions):
        task_id = solution['task_id']
        dsl_program = solution['dsl_program']

        if not dsl_program or not dsl_program.strip():
            continue

        if task_id not in challenges or task_id not in official_solutions:
            continue

        task_data = challenges[task_id]
        task_data['task_id'] = task_id

        try:
            predicted_output = execute_solution(dsl_program, task_data)
            predicted_grid = np.array(predicted_output)
            ground_truth_output_grid = np.array(official_solutions[task_id][0])

            if np.array_equal(predicted_grid, ground_truth_output_grid):
                verified_program_bank[task_id] = dsl_program
                verified_success_count += 1
            else:
                verified_failure_count += 1
        except NotImplementedError as e:
            error_message = str(e)
            method_name_match = re.search(r"(\w+)", error_message)
            if method_name_match:
                method_name = method_name_match.group(1)
                print(f"\n--- Encountered unimplemented DSL method: {method_name} ---")
                
                examples = find_usage_examples(log_file, method_name)
                
                print("\n## Meta-Prompt for DSL Implementation ##")
                print(f"You are an expert Python programmer. I need you to write the implementation for the DSL method/function called `{method_name}`.")
                print("\n**Context:** It may be a method of the `Grid` class or a global function.")
                print("\n**Intended Use:** Here are a few examples of how it was used in programs you generated earlier:")
                for ex in examples:
                    print(f"* ```python\n{ex}\n```")
                print(f"\nBased on these examples, write the Python code for the `{method_name}` method/function.")
                
                return # Stop execution
            else:
                print(f"Unrecognized NotImplementedError: {e}")
                execution_errors += 1
        except Exception as e:
            print(f"Error executing program for task {task_id}: {e}")
            execution_errors += 1
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(generated_solutions)} solutions...")

    print(f"\n--- Verification Complete ---")
    print(f"Verified Successes: {verified_success_count}")
    print(f"Verified Failures: {verified_failure_count}")
    print(f"Execution Errors: {execution_errors}")

    total_verified = verified_success_count + verified_failure_count
    verification_rate = (verified_success_count / total_verified) * 100 if total_verified > 0 else 0
    print(f"Verification Rate (of executed): {verification_rate:.2f}%")

    with open(program_bank_file, 'w') as f:
        json.dump(verified_program_bank, f, indent=2)
    print(f"\nSuccessfully saved {len(verified_program_bank)} verified programs to '{program_bank_file}'")


if __name__ == '__main__':
    verify_solutions()