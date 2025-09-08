"""
Test submission schema compliance for ARC Prize 2025.

This module validates that solver output matches the exact format required
by the Kaggle competition platform.
"""

import json
import pytest
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.solver import solve_task


def create_test_task() -> Dict[str, Any]:
    """Create a simple test task for validation."""
    return {
        "train": [
            {
                "input": [
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 0, 0]
                ],
                "output": [
                    [0, 1, 1],
                    [0, 1, 0],
                    [0, 0, 0]
                ]
            }
        ],
        "test": [
            {
                "input": [
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0]
                ]
            },
            {
                "input": [
                    [1, 1],
                    [0, 1]
                ]
            }
        ]
    }


class TestSubmissionSchema:
    """Test suite for submission format validation."""
    
    def test_output_structure(self):
        """Test that output has required top-level structure."""
        task = create_test_task()
        result = solve_task(task)
        
        assert isinstance(result, dict), "Output must be a dictionary"
        assert "attempt_1" in result, "Missing 'attempt_1' key"
        assert "attempt_2" in result, "Missing 'attempt_2' key"
        assert len(result) == 2, "Output should only contain attempt_1 and attempt_2"
    
    def test_attempt_structure(self):
        """Test that attempts have correct structure."""
        task = create_test_task()
        result = solve_task(task)
        
        for attempt_key in ["attempt_1", "attempt_2"]:
            attempt = result[attempt_key]
            assert isinstance(attempt, list), f"{attempt_key} must be a list"
            assert len(attempt) == len(task["test"]), f"{attempt_key} must have one output per test input"
    
    def test_grid_format(self):
        """Test that output grids are properly formatted."""
        task = create_test_task()
        result = solve_task(task)
        
        for attempt_key in ["attempt_1", "attempt_2"]:
            attempt = result[attempt_key]
            
            for i, output_grid in enumerate(attempt):
                assert isinstance(output_grid, list), f"{attempt_key}[{i}] must be a list"
                assert len(output_grid) > 0, f"{attempt_key}[{i}] cannot be empty"
                
                # Check that all rows are lists
                for j, row in enumerate(output_grid):
                    assert isinstance(row, list), f"{attempt_key}[{i}][{j}] must be a list"
                    assert len(row) > 0, f"{attempt_key}[{i}][{j}] cannot be empty"
                    
                    # Check that all cells are integers
                    for k, cell in enumerate(row):
                        assert isinstance(cell, int), f"{attempt_key}[{i}][{j}][{k}] must be an integer"
                        assert 0 <= cell <= 9, f"{attempt_key}[{i}][{j}][{k}] must be in range 0-9"
    
    def test_grid_consistency(self):
        """Test that grids have consistent row lengths."""
        task = create_test_task()
        result = solve_task(task)
        
        for attempt_key in ["attempt_1", "attempt_2"]:
            attempt = result[attempt_key]
            
            for i, output_grid in enumerate(attempt):
                if len(output_grid) > 0:
                    expected_width = len(output_grid[0])
                    for j, row in enumerate(output_grid):
                        assert len(row) == expected_width, \
                            f"{attempt_key}[{i}][{j}] has width {len(row)}, expected {expected_width}"
    
    def test_json_serializable(self):
        """Test that output is JSON serializable."""
        task = create_test_task()
        result = solve_task(task)
        
        try:
            json_str = json.dumps(result)
            reconstructed = json.loads(json_str)
            assert reconstructed == result, "JSON round-trip failed"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Output is not JSON serializable: {e}")
    
    def test_multiple_test_inputs(self):
        """Test handling of multiple test inputs."""
        task = create_test_task()
        result = solve_task(task)
        
        assert len(result["attempt_1"]) == 2, "Should have 2 outputs for 2 test inputs"
        assert len(result["attempt_2"]) == 2, "Should have 2 outputs for 2 test inputs"
    
    def test_deterministic_output(self):
        """Test that solver produces consistent output format."""
        task = create_test_task()
        
        # Run solver multiple times
        results = [solve_task(task) for _ in range(3)]
        
        # Check that all results have same structure
        for i, result in enumerate(results):
            assert "attempt_1" in result, f"Run {i} missing attempt_1"
            assert "attempt_2" in result, f"Run {i} missing attempt_2"
            assert len(result["attempt_1"]) == 2, f"Run {i} attempt_1 wrong length"
            assert len(result["attempt_2"]) == 2, f"Run {i} attempt_2 wrong length"
    
    def test_empty_task_handling(self):
        """Test handling of edge cases."""
        # Task with no training examples
        empty_train_task = {
            "train": [],
            "test": [{"input": [[1, 0], [0, 1]]}]
        }
        
        result = solve_task(empty_train_task)
        assert "attempt_1" in result
        assert "attempt_2" in result
        assert len(result["attempt_1"]) == 1
        assert len(result["attempt_2"]) == 1
    
    def test_large_grid_handling(self):
        """Test handling of larger grids."""
        large_task = {
            "train": [
                {
                    "input": [[i % 10 for i in range(30)] for _ in range(30)],
                    "output": [[i % 10 for i in range(30)] for _ in range(30)]
                }
            ],
            "test": [
                {
                    "input": [[i % 10 for i in range(25)] for _ in range(25)]
                }
            ]
        }
        
        result = solve_task(large_task)
        
        # Should still produce valid output
        assert isinstance(result["attempt_1"][0], list)
        assert isinstance(result["attempt_2"][0], list)
        assert len(result["attempt_1"][0]) > 0
        assert len(result["attempt_2"][0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
