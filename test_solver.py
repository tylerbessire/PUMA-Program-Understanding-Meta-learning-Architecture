"""
Test script for the enhanced ARC solver.

This script runs basic tests to ensure all components are working correctly
and demonstrates the enhanced capabilities.
"""

import numpy as np
import json
from typing import Dict, List, Any

from arc_solver.grid import to_array, to_list
from arc_solver.enhanced_solver import solve_task_enhanced, solve_task_baseline
from arc_solver.neural.features import extract_task_features, compute_task_signature
from arc_solver.neural.guidance import NeuralGuidance
from arc_solver.neural.episodic import EpisodicRetrieval
from arc_solver.neural.sketches import SketchMiner
from benchmark import setup_solver_environment


def create_test_task() -> Dict[str, Any]:
    """Create a simple test task for validation."""
    # Simple rotation task: rotate 90 degrees clockwise
    input_grid = [
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 0]
    ]
    
    output_grid = [
        [0, 1, 1],
        [0, 1, 0],
        [0, 0, 0]
    ]
    
    task = {
        "train": [
            {"input": input_grid, "output": output_grid}
        ],
        "test": [
            {"input": [
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 0]
            ]}
        ]
    }
    
    return task


def create_recolor_task() -> Dict[str, Any]:
    """Create a simple recoloring test task."""
    # Recoloring task: swap colors 1 and 2
    task = {
        "train": [
            {
                "input": [
                    [1, 2, 0],
                    [2, 1, 0],
                    [0, 0, 0]
                ],
                "output": [
                    [2, 1, 0],
                    [1, 2, 0],
                    [0, 0, 0]
                ]
            }
        ],
        "test": [
            {"input": [
                [1, 1, 2],
                [2, 0, 1],
                [0, 1, 2]
            ]}
        ]
    }
    
    return task


def test_basic_functionality():
    """Test basic solver functionality."""
    print("Testing basic functionality...")
    
    # Test simple task
    task = create_test_task()
    
    try:
        result = solve_task_baseline(task)
        print("✓ Baseline solver works")
        print(f"  Attempt 1 shape: {np.array(result['attempt_1'][0]).shape}")
        print(f"  Attempt 2 shape: {np.array(result['attempt_2'][0]).shape}")
    except Exception as e:
        print(f"✗ Baseline solver failed: {e}")
    
    try:
        result = solve_task_enhanced(task)
        print("✓ Enhanced solver works")
        print(f"  Attempt 1 shape: {np.array(result['attempt_1'][0]).shape}")
        print(f"  Attempt 2 shape: {np.array(result['attempt_2'][0]).shape}")
    except Exception as e:
        print(f"✗ Enhanced solver failed: {e}")


def test_feature_extraction():
    """Test feature extraction capabilities."""
    print("\nTesting feature extraction...")
    
    task = create_test_task()
    train_pairs = [(to_array(pair["input"]), to_array(pair["output"])) 
                   for pair in task["train"]]
    
    try:
        features = extract_task_features(train_pairs)
        print("✓ Feature extraction works")
        print(f"  Extracted {len(features)} features")
        print(f"  Shape preserved: {features.get('shape_preserved', 'N/A')}")
        print(f"  Likely rotation: {features.get('likely_rotation', 'N/A')}")
        
        signature = compute_task_signature(train_pairs)
        print(f"  Task signature: {signature}")
        
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")


def test_neural_guidance():
    """Test neural guidance component."""
    print("\nTesting neural guidance...")
    
    task = create_test_task()
    train_pairs = [(to_array(pair["input"]), to_array(pair["output"])) 
                   for pair in task["train"]]
    
    try:
        guidance = NeuralGuidance()
        predicted_ops = guidance.predict_operations(train_pairs)
        operation_scores = guidance.score_operations(train_pairs)
        
        print("✓ Neural guidance works")
        print(f"  Predicted operations: {predicted_ops}")
        print(f"  Operation scores: {operation_scores}")
        
    except Exception as e:
        print(f"✗ Neural guidance failed: {e}")


def test_episodic_retrieval():
    """Test episodic retrieval system."""
    print("\nTesting episodic retrieval...")
    
    try:
        episodic = EpisodicRetrieval("test_episodes.json")
        
        # Add a test episode
        task = create_test_task()
        train_pairs = [(to_array(pair["input"]), to_array(pair["output"])) 
                       for pair in task["train"]]
        test_program = [("rotate", {"k": 1})]
        
        episodic.add_successful_solution(train_pairs, [test_program], "test_task_1")
        
        # Query for similar tasks
        candidates = episodic.query_for_programs(train_pairs)
        
        print("✓ Episodic retrieval works")
        print(f"  Added episode successfully")
        print(f"  Retrieved {len(candidates)} candidate programs")
        
        stats = episodic.get_stats()
        print(f"  Database stats: {stats}")
        
    except Exception as e:
        print(f"✗ Episodic retrieval failed: {e}")


def test_sketch_mining():
    """Test program sketch mining."""
    print("\nTesting sketch mining...")
    
    try:
        sketch_miner = SketchMiner()
        
        # Add some test programs
        test_programs = [
            [("rotate", {"k": 1})],
            [("rotate", {"k": 2})],
            [("flip", {"axis": 0})],
            [("rotate", {"k": 1}), ("flip", {"axis": 0})],
        ]
        
        for program in test_programs:
            sketch_miner.add_successful_program(program)
        
        # Mine sketches
        sketches = sketch_miner.mine_sketches(min_frequency=1)
        
        print("✓ Sketch mining works")
        print(f"  Mined {len(sketches)} sketches")
        for sketch in sketches[:3]:
            print(f"    {sketch.operations} (frequency: {sketch.frequency})")
        
    except Exception as e:
        print(f"✗ Sketch mining failed: {e}")


def test_multiple_tasks():
    """Test solver on multiple different tasks."""
    print("\nTesting multiple task types...")
    
    tasks = [
        ("rotation", create_test_task()),
        ("recoloring", create_recolor_task()),
    ]
    
    for task_name, task in tasks:
        print(f"  Testing {task_name} task...")
        try:
            result = solve_task_enhanced(task)
            print(f"    ✓ {task_name} solved successfully")
        except Exception as e:
            print(f"    ✗ {task_name} failed: {e}")


def test_submission_format():
    """Test that output format matches Kaggle requirements."""
    print("\nTesting submission format...")
    
    task = create_test_task()
    
    try:
        result = solve_task_enhanced(task)
        
        # Check format requirements
        assert "attempt_1" in result, "Missing attempt_1"
        assert "attempt_2" in result, "Missing attempt_2"
        assert len(result["attempt_1"]) == len(task["test"]), "Wrong number of outputs in attempt_1"
        assert len(result["attempt_2"]) == len(task["test"]), "Wrong number of outputs in attempt_2"
        
        # Check that outputs are valid grids
        for attempt in [result["attempt_1"], result["attempt_2"]]:
            for output in attempt:
                assert isinstance(output, list), "Output not a list"
                assert all(isinstance(row, list) for row in output), "Output rows not lists"
                assert all(isinstance(cell, int) for row in output for cell in row), "Output cells not integers"
        
        print("✓ Submission format is correct")
        print(f"  attempt_1: {len(result['attempt_1'])} outputs")
        print(f"  attempt_2: {len(result['attempt_2'])} outputs")
        
    except Exception as e:
        print(f"✗ Submission format test failed: {e}")


def run_all_tests():
    """Run all tests."""
    print("=== Enhanced ARC Solver Test Suite ===\n")
    
    # Set up environment first
    print("Setting up test environment...")
    try:
        setup_solver_environment()
        print("✓ Environment setup complete\n")
    except Exception as e:
        print(f"✗ Environment setup failed: {e}\n")
    
    # Run individual tests
    test_basic_functionality()
    test_feature_extraction()
    test_neural_guidance()
    test_episodic_retrieval()
    test_sketch_mining()
    test_multiple_tasks()
    test_submission_format()
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    run_all_tests()
