#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "PUMA"))

import json
import numpy as np
from kaggle_setup import get_solver

# Load a simple task
with open('KAGGLE/ARC-challenges:solutions-kaggle/arc-agi_training_challenges.json') as f:
    challenges = json.load(f)

with open('KAGGLE/ARC-challenges:solutions-kaggle/arc-agi_training_solutions.json') as f:
    solutions = json.load(f)

task_id = '007bbfb7'
task_data = challenges[task_id]
solution = solutions[task_id]

print(f"=== Testing Task {task_id} ===\n")

# Show the task
for i, pair in enumerate(task_data['train']):
    inp = np.array(pair['input'])
    out = np.array(pair['output'])
    print(f"Example {i}: {inp.shape} -> {out.shape}")
    print("Input:")
    print(inp)
    print("Output:")
    print(out)
    print()

# Solve it
solver = get_solver()
result = solver.solve(task_data)

print("\n=== SOLUTION ===")
print(f"Generated {len(result['predictions'])} prediction(s)")

if result['predictions']:
    pred = np.array(result['predictions'][0])
    expected = np.array(solution[0])
    print(f"Prediction shape: {pred.shape}")
    print(f"Expected shape: {expected.shape}")
    print(f"Match: {np.array_equal(pred, expected)}")
    
    if not np.array_equal(pred, expected):
        print("\nPrediction:")
        print(pred)
        print("\nExpected:")
        print(expected)
