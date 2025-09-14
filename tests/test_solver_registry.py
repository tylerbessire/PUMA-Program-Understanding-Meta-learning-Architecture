"""
Tests for the solver registry system.

This module tests the central solver registry that manages different
solver implementations and provides a unified interface.
"""

import unittest
from arc_solver.registry import (
    get_solver, register_solver, list_solvers, solve_with_solver,
    fallback_solve, validate_solver, get_solver_info
)


class TestSolverRegistry(unittest.TestCase):
    """Test basic solver registry functionality."""
    
    def test_list_solvers(self):
        """Test listing available solvers."""
        solvers = list_solvers()
        self.assertIsInstance(solvers, list)
        self.assertGreater(len(solvers), 0)
        
        # Should include our new solvers
        self.assertIn("agentic", solvers)
        self.assertIn("genomic", solvers)
        self.assertIn("baseline", solvers)
    
    def test_get_existing_solver(self):
        """Test getting an existing solver."""
        solver = get_solver("agentic")
        self.assertTrue(callable(solver))
        
        solver = get_solver("genomic")
        self.assertTrue(callable(solver))
    
    def test_get_nonexistent_solver(self):
        """Test getting a nonexistent solver raises error."""
        with self.assertRaises(ValueError):
            get_solver("nonexistent_solver")
    
    def test_register_new_solver(self):
        """Test registering a new solver."""
        original_count = len(list_solvers())
        
        def dummy_solver(task):
            return {"attempt_1": [], "attempt_2": []}
        
        # Register as a module path (this is a bit artificial for testing)
        register_solver("test_dummy", "arc_solver.registry:fallback_solve")
        
        new_count = len(list_solvers())
        self.assertEqual(new_count, original_count + 1)
        self.assertIn("test_dummy", list_solvers())
    
    def test_fallback_solve(self):
        """Test fallback solver behavior."""
        task = {
            "train": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]]
                }
            ],
            "test": [
                {"input": [[1, 0], [0, 1]]},
                {"input": [[2, 3], [4, 5]]}
            ]
        }
        
        result = fallback_solve(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)
        self.assertEqual(len(result["attempt_1"]), 2)
        self.assertEqual(len(result["attempt_2"]), 2)
        
        # Should return identity transformations
        self.assertEqual(result["attempt_1"][0], [[1, 0], [0, 1]])
        self.assertEqual(result["attempt_1"][1], [[2, 3], [4, 5]])


class TestSolverValidation(unittest.TestCase):
    """Test solver validation functionality."""
    
    def test_validate_existing_solver(self):
        """Test validation of existing solvers."""
        self.assertTrue(validate_solver("agentic"))
        self.assertTrue(validate_solver("genomic"))
        
        # Baseline might not validate if dependencies are missing
        # self.assertTrue(validate_solver("baseline"))
    
    def test_validate_nonexistent_solver(self):
        """Test validation of nonexistent solver."""
        self.assertFalse(validate_solver("nonexistent_solver"))
    
    def test_get_solver_info(self):
        """Test getting solver information."""
        info = get_solver_info("agentic")
        
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("entry_point", info)
        self.assertIn("valid", info)
        self.assertEqual(info["name"], "agentic")
        
        # Test nonexistent solver
        info = get_solver_info("nonexistent")
        self.assertIn("error", info)


class TestSolverExecution(unittest.TestCase):
    """Test solver execution through registry."""
    
    def setUp(self):
        self.simple_task = {
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
    
    def test_solve_with_agentic(self):
        """Test solving with agentic solver through registry."""
        result = solve_with_solver(self.simple_task, "agentic")
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)
        self.assertIsInstance(result["attempt_1"], list)
        self.assertEqual(len(result["attempt_1"]), 1)  # One test case
    
    def test_solve_with_genomic(self):
        """Test solving with genomic solver through registry."""
        result = solve_with_solver(self.simple_task, "genomic")
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)
        self.assertIsInstance(result["attempt_1"], list)
        self.assertEqual(len(result["attempt_1"]), 1)  # One test case
    
    def test_solve_with_default_solver(self):
        """Test using default solver."""
        result = solve_with_solver(self.simple_task)  # No solver specified
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)
    
    def test_solve_with_invalid_solver_fallback(self):
        """Test fallback when solver fails."""
        # This should fallback to identity transformation
        result = solve_with_solver(self.simple_task, "nonexistent_solver")
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)
    
    def test_malformed_task_handling(self):
        """Test handling of malformed tasks."""
        malformed_task = {
            "train": "not a list",
            "test": [{"input": [[1, 2]]}]
        }
        
        # Should not crash, should return some result
        result = solve_with_solver(malformed_task, "agentic")
        self.assertIsInstance(result, dict)


class TestEnsembleFunctionality(unittest.TestCase):
    """Test ensemble solver functionality."""
    
    def setUp(self):
        self.simple_task = {
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
    
    def test_ensemble_new_solvers(self):
        """Test ensemble of new solvers."""
        result = solve_with_solver(self.simple_task, "ensemble_new")
        
        self.assertIsInstance(result, dict)
        self.assertIn("attempt_1", result)
        self.assertIn("attempt_2", result)


if __name__ == "__main__":
    unittest.main()