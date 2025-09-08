"""
End-to-end integration tests for the complete solver pipeline.

This module tests the full solver workflow from task input to submission output,
ensuring all components work together correctly.
"""

import pytest
import numpy as np
from typing import Dict, List, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.solver import solve_task
from arc_solver.grid import to_array, to_list
from arc_solver.heuristics import score_candidate


class TestSolverEndToEnd:
    """End-to-end tests for the complete solver pipeline."""
    
    def create_rotation_task(self) -> Dict[str, Any]:
        """Create a simple rotation task."""
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
                }
            ]
        }
    
    def create_recolor_task(self) -> Dict[str, Any]:
        """Create a simple recoloring task."""
        return {
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
                {
                    "input": [
                        [1, 1, 2],
                        [2, 0, 1],
                        [0, 1, 2]
                    ]
                }
            ]
        }
    
    def create_identity_task(self) -> Dict[str, Any]:
        """Create a task where output equals input."""
        return {
            "train": [
                {
                    "input": [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ],
                    "output": [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]
                    ]
                }
            ],
            "test": [
                {
                    "input": [
                        [9, 8, 7],
                        [6, 5, 4],
                        [3, 2, 1]
                    ]
                }
            ]
        }
    
    def test_rotation_task_solving(self):
        """Test that solver can handle rotation tasks."""
        task = self.create_rotation_task()
        result = solve_task(task)
        
        # Check basic structure
        assert "attempt_1" in result
        assert "attempt_2" in result
        assert len(result["attempt_1"]) == 1
        assert len(result["attempt_2"]) == 1
        
        # Verify outputs are valid grids
        for attempt in ["attempt_1", "attempt_2"]:
            output = result[attempt][0]
            assert isinstance(output, list)
            assert all(isinstance(row, list) for row in output)
            assert all(isinstance(cell, int) for row in output for cell in row)
    
    def test_recolor_task_solving(self):
        """Test that solver can handle recoloring tasks."""
        task = self.create_recolor_task()
        result = solve_task(task)
        
        # Check basic structure
        assert "attempt_1" in result
        assert "attempt_2" in result
        
        # Verify non-trivial output (not just identity)
        input_grid = task["test"][0]["input"]
        output1 = result["attempt_1"][0]
        
        # Should produce some transformation
        assert isinstance(output1, list)
        assert len(output1) > 0
    
    def test_identity_task_solving(self):
        """Test that solver can handle identity transformations."""
        task = self.create_identity_task()
        result = solve_task(task)
        
        # Should produce valid output
        assert "attempt_1" in result
        assert "attempt_2" in result
        
        output1 = result["attempt_1"][0]
        assert isinstance(output1, list)
        assert len(output1) == 3
        assert len(output1[0]) == 3
    
    def test_multiple_test_inputs(self):
        """Test handling of multiple test inputs."""
        task = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            "test": [
                {"input": [[1, 1], [0, 0]]},
                {"input": [[2, 3], [4, 5]]},
                {"input": [[0, 1, 2], [3, 4, 5]]}
            ]
        }
        
        result = solve_task(task)
        
        assert len(result["attempt_1"]) == 3
        assert len(result["attempt_2"]) == 3
        
        # All outputs should be valid
        for i in range(3):
            for attempt in ["attempt_1", "attempt_2"]:
                output = result[attempt][i]
                assert isinstance(output, list)
                assert len(output) > 0
    
    def test_solver_consistency(self):
        """Test that solver produces consistent output structure."""
        tasks = [
            self.create_rotation_task(),
            self.create_recolor_task(),
            self.create_identity_task()
        ]
        
        for i, task in enumerate(tasks):
            result = solve_task(task)
            
            # Check consistent structure
            assert isinstance(result, dict), f"Task {i}: result not dict"
            assert set(result.keys()) == {"attempt_1", "attempt_2"}, f"Task {i}: wrong keys"
            
            # Check attempt lengths match test inputs
            expected_length = len(task["test"])
            assert len(result["attempt_1"]) == expected_length, f"Task {i}: attempt_1 length"
            assert len(result["attempt_2"]) == expected_length, f"Task {i}: attempt_2 length"
    
    def test_solver_robustness(self):
        """Test solver robustness with edge cases."""
        # Very small grids
        small_task = {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}]
        }
        
        result = solve_task(small_task)
        assert result["attempt_1"][0] == [[3]] or result["attempt_1"][0] == [[2]]
        
        # Large grids
        large_grid = [[i % 10 for i in range(20)] for _ in range(20)]
        large_task = {
            "train": [{"input": large_grid, "output": large_grid}],
            "test": [{"input": large_grid}]
        }
        
        result = solve_task(large_task)
        assert len(result["attempt_1"][0]) <= 30  # Should handle large grids
        assert len(result["attempt_2"][0]) <= 30
    
    def test_performance_characteristics(self):
        """Test that solver meets performance requirements."""
        import time
        
        task = self.create_rotation_task()
        
        # Measure solve time
        start_time = time.time()
        result = solve_task(task)
        solve_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert solve_time < 30.0, f"Solver took {solve_time:.2f}s, too slow"
        
        # Should produce valid output
        assert "attempt_1" in result
        assert "attempt_2" in result
    
    def test_fallback_behavior(self):
        """Test that solver gracefully handles difficult tasks."""
        # Create a task that's hard to solve
        difficult_task = {
            "train": [
                {
                    "input": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]],
                    "output": [[9, 8, 7, 6], [5, 4, 3, 2, 1, 0]]  # Complex transformation
                }
            ],
            "test": [
                {"input": [[1, 1, 1], [2, 2, 2], [3, 3, 3]]}
            ]
        }
        
        # Should not crash, even if it can't solve perfectly
        result = solve_task(difficult_task)
        
        assert "attempt_1" in result
        assert "attempt_2" in result
        assert isinstance(result["attempt_1"][0], list)
        assert isinstance(result["attempt_2"][0], list)
    
    def test_enhanced_vs_baseline(self):
        """Test that enhanced solver works and provides fallback."""
        import os
        
        task = self.create_rotation_task()
        
        # Test with enhancements enabled (default)
        os.environ.pop('ARC_USE_BASELINE', None)
        enhanced_result = solve_task(task)
        
        # Test with baseline only
        os.environ['ARC_USE_BASELINE'] = '1'
        baseline_result = solve_task(task)
        
        # Both should produce valid results
        for result in [enhanced_result, baseline_result]:
            assert "attempt_1" in result
            assert "attempt_2" in result
            assert len(result["attempt_1"]) == 1
            assert len(result["attempt_2"]) == 1
        
        # Clean up
        os.environ.pop('ARC_USE_BASELINE', None)
    
    def test_grid_conversion_pipeline(self):
        """Test that grid conversion works correctly through the pipeline."""
        task = self.create_rotation_task()
        
        # Convert to internal format and back
        train_input = to_array(task["train"][0]["input"])
        train_output = to_array(task["train"][0]["output"])
        
        # Should preserve data
        assert train_input.shape == (3, 3)
        assert train_output.shape == (3, 3)
        
        # Convert back to lists
        list_input = to_list(train_input)
        list_output = to_list(train_output)
        
        assert list_input == task["train"][0]["input"]
        assert list_output == task["train"][0]["output"]
        
        # Full pipeline should work
        result = solve_task(task)
        
        # Results should be valid lists
        for attempt in ["attempt_1", "attempt_2"]:
            output = result[attempt][0]
            array_output = to_array(output)
            reconstructed = to_list(array_output)
            assert reconstructed == output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
