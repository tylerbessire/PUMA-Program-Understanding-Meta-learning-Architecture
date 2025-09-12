#!/usr/bin/env python3
"""
Test to demonstrate that the P1 beam search scoring issue is fixed.

This test shows that valid programs are found even when they use operations
with low prior scores, proving that op_scores only affects ranking, not filtering.
"""

import numpy as np
from arc_solver.grid import to_array
from arc_solver.beam_search import beam_search
from arc_solver.dsl import apply_program


def test_p1_fix_low_prior_valid_programs_found():
    """
    Demonstrates the P1 fix: programs with correct transformations should be found
    even when using operations with very low or zero prior scores.
    """
    print("Testing P1 fix: Low-prior operations should still yield valid solutions")
    
    # Create a simple rotation task
    inp = to_array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    out = np.rot90(inp, -1)  # Rotate clockwise (k=1)
    
    print(f"Input:\n{inp}")
    print(f"Expected output:\n{out}")
    
    # Test 1: With very low prior for rotation (0.001)
    scores_low = {op: 1.0 for op in ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']}
    scores_low['rotate'] = 0.001  # Very low prior
    
    progs_low, stats_low = beam_search([(inp, out)], beam_width=5, depth=1, op_scores=scores_low)
    
    # Check that the correct rotation is found despite low prior
    rotation_found_low = any(p == [("rotate", {"k": 1})] for p in progs_low)
    print(f"\nTest 1 - Low prior (0.001) for rotate:")
    print(f"Programs found: {len(progs_low)}")
    print(f"Correct rotation found: {rotation_found_low}")
    for i, prog in enumerate(progs_low):
        result = apply_program(inp, prog)
        correct = np.array_equal(result, out)
        print(f"  Program {i+1}: {prog} -> Correct: {correct}")
    
    # Test 2: With zero prior for rotation
    scores_zero = {op: 1.0 for op in ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']}
    scores_zero['rotate'] = 0.0  # Zero prior
    
    progs_zero, stats_zero = beam_search([(inp, out)], beam_width=5, depth=1, op_scores=scores_zero)
    
    # Check that the correct rotation is still found with zero prior
    rotation_found_zero = any(p == [("rotate", {"k": 1})] for p in progs_zero)
    print(f"\nTest 2 - Zero prior (0.0) for rotate:")
    print(f"Programs found: {len(progs_zero)}")
    print(f"Correct rotation found: {rotation_found_zero}")
    for i, prog in enumerate(progs_zero):
        result = apply_program(inp, prog)
        correct = np.array_equal(result, out)
        print(f"  Program {i+1}: {prog} -> Correct: {correct}")
    
    # Test 3: Control - with normal prior for rotation
    scores_normal = {op: 1.0 for op in ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']}
    
    progs_normal, stats_normal = beam_search([(inp, out)], beam_width=5, depth=1, op_scores=scores_normal)
    
    rotation_found_normal = any(p == [("rotate", {"k": 1})] for p in progs_normal)
    print(f"\nTest 3 - Normal prior (1.0) for rotate:")
    print(f"Programs found: {len(progs_normal)}")
    print(f"Correct rotation found: {rotation_found_normal}")
    for i, prog in enumerate(progs_normal):
        result = apply_program(inp, prog)
        correct = np.array_equal(result, out)
        print(f"  Program {i+1}: {prog} -> Correct: {correct}")
    
    # Assertions
    assert rotation_found_low, "FAILED: Rotation should be found with low prior (0.001)"
    assert rotation_found_zero, "FAILED: Rotation should be found with zero prior (0.0)"
    assert rotation_found_normal, "FAILED: Rotation should be found with normal prior (1.0)"
    
    print(f"\n✅ SUCCESS: P1 fix verified! Valid programs found regardless of prior scores.")
    print(f"   - Low prior (0.001): {rotation_found_low}")
    print(f"   - Zero prior (0.0): {rotation_found_zero}")  
    print(f"   - Normal prior (1.0): {rotation_found_normal}")
    
    return True


if __name__ == "__main__":
    test_p1_fix_low_prior_valid_programs_found()
