"""
Basic smoke tests for the genomic solver.

This module provides simple tests to verify that the genomic solver
can be imported and run on minimal examples without crashing.
"""

import unittest
import numpy as np
from arc_solver.genomic.solver import solve_task_genomic, solve_task_genomic_dict, GenomicSolver
from arc_solver.genomic.hilbert import hilbert_order, grid_to_hilbert_sequence
from arc_solver.genomic.tokenize import tokenize_sequence, run_length_encode
from arc_solver.genomic.align import needleman_wunsch, AlignmentScorer
from arc_solver.genomic.script import infer_script, consensus_script
from arc_solver.grid import to_array


class TestHilbert(unittest.TestCase):
    """Test Hilbert curve functionality."""
    
    def test_hilbert_order_small(self):
        """Test Hilbert ordering for small grids."""
        coords = hilbert_order(2, 2)
        self.assertEqual(len(coords), 4)
        self.assertTrue(all(isinstance(c, tuple) and len(c) == 2 for c in coords))
        
        # All coordinates should be valid
        for y, x in coords:
            self.assertTrue(0 <= y < 2)
            self.assertTrue(0 <= x < 2)
    
    def test_hilbert_order_rectangular(self):
        """Test Hilbert ordering for rectangular grids."""
        coords = hilbert_order(2, 3)
        self.assertEqual(len(coords), 6)  # Should cover all cells
        
        # Check bounds
        for y, x in coords:
            self.assertTrue(0 <= y < 2)
            self.assertTrue(0 <= x < 3)
    
    def test_grid_to_hilbert_sequence(self):
        """Test conversion from grid to Hilbert sequence."""
        grid = to_array([[1, 2], [3, 4]])
        sequence = grid_to_hilbert_sequence(grid)
        
        self.assertIsInstance(sequence, list)
        self.assertEqual(len(sequence), 4)
        self.assertTrue(all(isinstance(x, int) for x in sequence))


class TestTokenization(unittest.TestCase):
    """Test tokenization functionality."""
    
    def setUp(self):
        self.simple_grid = to_array([[1, 0], [0, 1]])
        self.coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    def test_tokenize_sequence(self):
        """Test basic tokenization."""
        sequence = [1, 0, 0, 1]
        tokens = tokenize_sequence(sequence, self.simple_grid, self.coords)
        
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), len(sequence))
        self.assertTrue(all(isinstance(t, str) for t in tokens))
    
    def test_run_length_encode(self):
        """Test run-length encoding."""
        tokens = ["A", "A", "B", "A", "A", "A"]
        encoded = run_length_encode(tokens)
        
        expected = [("A", 2), ("B", 1), ("A", 3)]
        self.assertEqual(encoded, expected)
    
    def test_empty_sequence(self):
        """Test handling of empty sequences."""
        tokens = []
        encoded = run_length_encode(tokens)
        self.assertEqual(encoded, [])


class TestAlignment(unittest.TestCase):
    """Test sequence alignment functionality."""
    
    def test_needleman_wunsch_identical(self):
        """Test alignment of identical sequences."""
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "B", "C"]
        
        alignment = needleman_wunsch(seq1, seq2)
        
        self.assertIsNotNone(alignment)
        self.assertEqual(len(alignment.sequence1), len(seq1))
        self.assertEqual(len(alignment.sequence2), len(seq2))
        self.assertGreater(alignment.score, 0)
    
    def test_needleman_wunsch_different(self):
        """Test alignment of different sequences."""
        seq1 = ["A", "B", "C"]
        seq2 = ["A", "X", "C"]
        
        alignment = needleman_wunsch(seq1, seq2)
        
        self.assertIsNotNone(alignment)
        self.assertIsInstance(alignment.edits, list)
    
    def test_alignment_scorer(self):
        """Test custom alignment scoring."""
        scorer = AlignmentScorer(match_score=3.0, mismatch_penalty=-2.0)
        
        match_score = scorer.score_match("A", "A")
        mismatch_score = scorer.score_match("A", "B")
        gap_score = scorer.score_gap()
        
        self.assertEqual(match_score, 3.0)
        self.assertEqual(mismatch_score, -2.0)
        self.assertEqual(gap_score, -2.0)


class TestScript(unittest.TestCase):
    """Test script inference and consensus functionality."""
    
    def test_infer_script_identity(self):
        """Test script inference for identity transformation."""
        input_grid = to_array([[1, 0], [0, 1]])
        output_grid = to_array([[1, 0], [0, 1]])
        
        script = infer_script(input_grid, output_grid)
        
        self.assertIsNotNone(script)
        self.assertIsInstance(script.mutations, list)
        self.assertIsInstance(script.confidence, float)
        self.assertTrue(0 <= script.confidence <= 1)
    
    def test_infer_script_simple_change(self):
        """Test script inference for simple transformation."""
        input_grid = to_array([[1, 1], [1, 1]])
        output_grid = to_array([[2, 2], [2, 2]])
        
        script = infer_script(input_grid, output_grid)
        
        self.assertIsNotNone(script)
        self.assertIsInstance(script.mutations, list)
    
    def test_consensus_script_single(self):
        """Test consensus with single script."""
        input_grid = to_array([[1, 0], [0, 1]])
        output_grid = to_array([[2, 0], [0, 2]])
        
        script = infer_script(input_grid, output_grid)
        consensus = consensus_script([script])
        
        self.assertIsNotNone(consensus)
        self.assertIsInstance(consensus.mutations, list)
    
    def test_consensus_script_empty(self):
        """Test consensus with no scripts."""
        consensus = consensus_script([])
        
        self.assertIsNotNone(consensus)
        self.assertEqual(len(consensus.mutations), 0)
        self.assertEqual(consensus.confidence, 0.0)


class TestGenomicSolver(unittest.TestCase):
    """Test the main genomic solver functionality."""
    
    def test_solver_creation(self):
        """Test solver can be created."""
        solver = GenomicSolver()
        self.assertIsNotNone(solver)
    
    def test_solver_empty_task(self):
        """Test solver behavior on empty task."""
        solver = GenomicSolver()
        task = {"train": [], "test": []}
        
        result = solver.solve_task(task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_solver_identity_task(self):
        """Test solver on identity task."""
        solver = GenomicSolver()
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
        
        result = solver.solve_task(task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], np.ndarray)
    
    def test_solver_statistics(self):
        """Test solver statistics tracking."""
        solver = GenomicSolver()
        stats = solver.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('tasks_processed', stats)
        self.assertIn('successful_tasks', stats)
        self.assertIn('success_rate', stats)
    
    def test_solve_task_genomic_function(self):
        """Test the standalone genomic solver function."""
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
        
        result = solve_task_genomic(task)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
    
    def test_solve_task_genomic_dict(self):
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
        
        result = solve_task_genomic_dict(task)
        
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
        
        # Should not crash
        result = solve_task_genomic(task)
        self.assertIsInstance(result, list)
    
    def test_task_analysis(self):
        """Test detailed task analysis functionality."""
        solver = GenomicSolver()
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
        
        analysis = solver.analyze_task(task)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('train_pairs', analysis)
        self.assertIn('test_cases', analysis)
        self.assertIn('scripts', analysis)


class TestGenomicIntegration(unittest.TestCase):
    """Integration tests for genomic solver components."""
    
    def test_registry_integration(self):
        """Test that solver works through registry."""
        from arc_solver.registry import get_solver
        
        solver = get_solver("genomic")
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
    
    def test_alignment_method_configuration(self):
        """Test different alignment methods."""
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
        
        result1 = solve_task_genomic(task, alignment_method="needleman_wunsch")
        result2 = solve_task_genomic(task, alignment_method="smith_waterman")
        
        self.assertIsInstance(result1, list)
        self.assertIsInstance(result2, list)


if __name__ == "__main__":
    unittest.main()