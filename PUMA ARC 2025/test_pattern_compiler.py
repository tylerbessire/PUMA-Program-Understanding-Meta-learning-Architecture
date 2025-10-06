#!/usr/bin/env python3
"""Test pattern compiler."""
import sys
sys.path.insert(0, 'PUMA')

import numpy as np
from arc_solver.pattern_compiler import PatternCompiler

# Simple test case
compiler = PatternCompiler()

# Create a test pattern from the learning log
pattern = {
    'type': 'extraction',
    'description': 'Transform: Reduce from (30, 30) to (9, 4) | Remove colors: [8, 5, 7] | Test: (30, 30) → (9, 3) | Extract region of size (9, 3)',
    'confidence': 0.5
}

# Create a simple test input
test_input = np.random.randint(0, 10, (30, 30))
expected_output = np.random.randint(0, 10, (9, 3))

train_pairs = [(test_input, expected_output)]

print("Testing pattern compilation...")
print(f"Pattern: {pattern['type']}")
print(f"Description: {pattern['description'][:100]}...")

program = compiler.compile_from_pattern(pattern, train_pairs)

if program:
    print("✓ Pattern compiled successfully!")
    try:
        result = program(test_input)
        print(f"✓ Program executed successfully!")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Expected shape: {expected_output.shape}")

        if result.shape == expected_output.shape:
            print("✓ Output shape matches expected!")
        else:
            print(f"✗ Shape mismatch: got {result.shape}, expected {expected_output.shape}")
    except Exception as e:
        print(f"✗ Program execution failed: {e}")
else:
    print("✗ Pattern failed to compile")
