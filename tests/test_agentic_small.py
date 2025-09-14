"""
Basic smoke tests for the agentic solver.

This module provides simple tests to verify that the agentic solver
can be imported and run on minimal examples without crashing.
"""

import unittest
import numpy as np
from arc_solver.agents.agentic_solver import solve_task_agentic, solve_task_agentic_dict
from arc_solver.agents.ops import Op, execute_operation, execute_program_on_grid
from arc_solver.grid import to_array


class TestAgenticOps(unittest.TestCase):
    """Test individual operations in the agentic solver."""
    
    def setUp(self):
        self.simple_grid = to_array([[1, 0], [0, 1]])
    
    def test_identity_op(self):
        """Test identity operation."""
        op = Op("identity", ())
        result = execute_operation(self.simple_grid, op)
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, self.simple_grid)
    
    def test_recolor_op(self):
        """Test recolor operation."""
        op = Op("recolor", (1, 2))
        result = execute_operation(self.simple_grid, op)
        self.assertIsNotNone(result)
        expected = to_array([[2, 0], [0, 2]])
        np.testing.assert_array_equal(result, expected)
    
    def test_translate_op(self):
        """Test translate operation."""
        op = Op("translate", (1, 0))  # Move down by 1
        result = execute_operation(self.simple_grid, op)
        self.assertIsNotNone(result)
        # Should shift content down
        self.assertEqual(result.shape, self.simple_grid.shape)
    
    def test_reflect_op(self):
        """Test reflect operation."""
        op = Op("reflect", ("h",))  # Horizontal reflection
        result = execute_operation(self.simple_grid, op)
        self.assertIsNotNone(result)
        expected = to_array([[0, 1], [1, 0]])  # Flipped left-right
        np.testing.assert_array_equal(result, expected)
    
    def test_invalid_op(self):
        """Test that invalid operations return None."""
        op = Op("nonexistent", ())
        result = execute_operation(self.simple_grid, op)
        self.assertIsNone(result)
    
    def test_program_execution(self):
        """Test executing a program (sequence of operations)."""
        program = [
            Op("recolor", (1, 2)),
            Op("reflect", ("h",))
        ]
        result = execute_program_on_grid(program, self.simple_grid)
        self.assertIsNotNone(result)
    
    def test_failed_program(self):
        """Test that programs with invalid operations fail gracefully."""
        program = [
            Op("recolor", (1, 2)),
            Op("nonexistent", ()),
            Op("reflect", ("h",))
        ]
        result = execute_program_on_grid(program, self.simple_grid)
        self.assertIsNone(result)


class TestAgenticSolver(unittest.TestCase):
    """Test the main agentic solver functionality."""
    
    def test_solver_import(self):
        """Test that solver can be imported."""
        from arc_solver.agents.agentic_solver import solve_task_agentic
        self.assertTrue(callable(solve_task_agentic))
    
    def test_empty_task(self):
        """Test solver behavior on empty task."""
        task = {"train": [], "test": []}
        result = solve_task_agentic(task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_identity_task(self):
        """Test solver on identity transformation task."""
        task = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[1, 0], [0, 1]]
                }
            ],
            "test": [
                {"input": [[2, 0], [0, 2]]}
            ]
        }
        result = solve_task_agentic(task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
    
    def test_simple_recolor_task(self):
        """Test solver on simple recoloring task."""
        task = {
            "train": [
                {
                    "input": [[1, 1], [1, 1]],
                    "output": [[2, 2], [2, 2]]
                }
            ],
            "test": [
                {"input": [[1, 1], [1, 1]]}
            ]
        }
        result = solve_task_agentic(task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        # Should ideally solve this simple recoloring
        output = result[0]
        self.assertEqual(output.shape, (2, 2))
    
    def test_dict_interface(self):
        """Test the dictionary interface for solver registry."""
        task = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            "test": [
                {"input": [[1, 0], [0, 1]]}
            ]
        }
        result = solve_task_agentic_dict(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)
        self.assertIsInstance(result["attempt_1"], list)
        self.assertIsInstance(result["attempt_2"], list)
    
    def test_malformed_input_handling(self):
        """Test handling of malformed input data."""
        task = {
            "train": [
                {
                    "input": "not a grid",
                    "output": [[1, 2]]
                }
            ],
            "test": [
                {"input": [[1, 2]]}
            ]
        }
        
        # Should not crash, should return some result
        result = solve_task_agentic(task)
        self.assertIsInstance(result, list)
    
    def test_beam_search_parameters(self):
        """Test that beam search parameters can be configured."""
        task = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            "test": [
                {"input": [[1, 0], [0, 1]]}
            ]
        }
        
        # Test with small beam width
        result1 = solve_task_agentic(task, beam_width=4, max_depth=2)
        self.assertIsInstance(result1, list)
        
        # Test with larger beam width
        result2 = solve_task_agentic(task, beam_width=16, max_depth=3)
        self.assertIsInstance(result2, list)


class TestAgenticIntegration(unittest.TestCase):
    """Integration tests for agentic solver components."""
    
    def test_registry_integration(self):
        """Test that solver works through registry."""
        from arc_solver.registry import get_solver
        
        solver = get_solver("agentic")
        self.assertTrue(callable(solver))
        
        task = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[2, 0], [0, 2]]
                }
            ],
            "test": [
                {"input": [[1, 0], [0, 1]]}
            ]
        }
        
        result = solver(task)
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)


if __name__ == "__main__":
    unittest.main()