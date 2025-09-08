"""
Tests for test-time training (TTT) component.

This module tests the test-time adaptation system that fine-tunes scoring
functions on individual ARC tasks using training demonstrations.
"""

import pytest
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.ttt import TestTimeTrainer, AdaptiveScorer, DataAugmentation


class TestTTT:
    """Test suite for test-time training components."""
    
    def create_test_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create test training pairs."""
        return [
            (to_array([[1, 0], [0, 1]]), to_array([[0, 1], [1, 0]])),
            (to_array([[2, 0], [0, 2]]), to_array([[0, 2], [2, 0]]))
        ]
    
    def create_test_programs(self) -> List[List[Tuple[str, dict]]]:
        """Create test candidate programs."""
        return [
            [("flip", {"axis": 0})],
            [("flip", {"axis": 1})],
            [("rotate", {"k": 2})],
            [("identity", {})]
        ]
    
    def test_adaptive_scorer_creation(self):
        """Test AdaptiveScorer initialization."""
        scorer = AdaptiveScorer(feature_dim=10)
        
        assert scorer.feature_dim == 10
        assert len(scorer.weights) == 10
        assert scorer.bias == 0.0
        assert scorer.learning_rate == 0.1
        assert np.allclose(scorer.weights, 1.0/10)  # Uniform initialization
    
    def test_program_feature_extraction(self):
        """Test feature extraction from programs."""
        scorer = AdaptiveScorer()
        train_pairs = self.create_test_pairs()
        programs = self.create_test_programs()
        
        for program in programs:
            features = scorer.extract_program_features(program, train_pairs)
            
            assert isinstance(features, np.ndarray)
            assert features.shape == (scorer.feature_dim,)
            assert np.all(features >= 0)  # Features should be non-negative
    
    def test_program_scoring(self):
        """Test program scoring functionality."""
        scorer = AdaptiveScorer()
        train_pairs = self.create_test_pairs()
        programs = self.create_test_programs()
        
        for program in programs:
            score = scorer.score_program(program, train_pairs)
            
            assert isinstance(score, (int, float))
            # Score can be any real number (no bounds enforced)
    
    def test_weight_updates(self):
        """Test adaptive weight updates."""
        scorer = AdaptiveScorer()
        train_pairs = self.create_test_pairs()
        
        # Create positive and negative program examples
        positive_programs = [
            [("flip", {"axis": 0})],  # Should work for the test task
        ]
        negative_programs = [
            [("identity", {})],  # Won't work for flip task
        ]
        
        # Store initial weights
        initial_weights = scorer.weights.copy()
        
        # Update weights
        scorer.update_weights(positive_programs, negative_programs, train_pairs)
        
        # Weights should have changed
        assert not np.array_equal(scorer.weights, initial_weights)
        
        # Weights should still be normalized and positive
        assert np.all(scorer.weights > 0)
        assert np.isclose(np.sum(scorer.weights), 1.0, atol=1e-6)
    
    def test_test_time_trainer_initialization(self):
        """Test TestTimeTrainer initialization."""
        trainer = TestTimeTrainer()
        
        assert trainer.base_scorer is not None
        assert trainer.adapted_scorer is None
        assert len(trainer.adaptation_history) == 0
    
    def test_task_adaptation(self):
        """Test adaptation to specific tasks."""
        trainer = TestTimeTrainer()
        train_pairs = self.create_test_pairs()
        candidate_programs = self.create_test_programs()
        
        # Perform adaptation
        adapted_scorer = trainer.adapt_to_task(train_pairs, candidate_programs, num_iterations=3)
        
        assert adapted_scorer is not None
        assert trainer.adapted_scorer is adapted_scorer
        assert len(trainer.adaptation_history) == 3
        
        # Check adaptation history structure
        for entry in trainer.adaptation_history:
            assert 'iteration' in entry
            assert 'positive_count' in entry
            assert 'negative_count' in entry
            assert 'weights' in entry
    
    def test_adapted_scoring(self):
        """Test scoring with adapted scorer."""
        trainer = TestTimeTrainer()
        train_pairs = self.create_test_pairs()
        candidate_programs = self.create_test_programs()
        
        # Adapt to task
        trainer.adapt_to_task(train_pairs, candidate_programs, num_iterations=2)
        
        # Test adapted scoring
        for program in candidate_programs:
            score = trainer.score_with_adaptation(program, train_pairs)
            assert isinstance(score, (int, float))
    
    def test_adaptation_statistics(self):
        """Test adaptation statistics collection."""
        trainer = TestTimeTrainer()
        train_pairs = self.create_test_pairs()
        candidate_programs = self.create_test_programs()
        
        # Perform adaptation
        trainer.adapt_to_task(train_pairs, candidate_programs, num_iterations=3)
        
        # Get statistics
        stats = trainer.get_adaptation_stats()
        
        assert isinstance(stats, dict)
        assert 'iterations' in stats
        assert 'total_positive_examples' in stats
        assert 'total_negative_examples' in stats
        assert 'final_weights' in stats
        assert 'weight_variance' in stats
        
        assert stats['iterations'] == 3
        assert isinstance(stats['final_weights'], list)
        assert len(stats['final_weights']) == trainer.adapted_scorer.feature_dim
    
    def test_data_augmentation(self):
        """Test data augmentation functionality."""
        train_pairs = self.create_test_pairs()
        
        # Test augmentation
        augmented = DataAugmentation.augment_training_pairs(train_pairs, max_augmentations=5)
        
        assert isinstance(augmented, list)
        assert len(augmented) >= len(train_pairs)  # Should have at least original pairs
        assert len(augmented) <= 5  # Should respect max limit
        
        # All pairs should be valid
        for inp, out in augmented:
            assert isinstance(inp, np.ndarray)
            assert isinstance(out, np.ndarray)
            assert inp.ndim == 2
            assert out.ndim == 2
    
    def test_negative_example_generation(self):
        """Test negative example generation."""
        train_pairs = self.create_test_pairs()
        
        # Generate negative examples
        negatives = DataAugmentation.generate_negative_examples(train_pairs, num_negatives=3)
        
        assert isinstance(negatives, list)
        assert len(negatives) <= 3
        
        # All examples should be valid pairs
        for inp, out in negatives:
            assert isinstance(inp, np.ndarray)
            assert isinstance(out, np.ndarray)
            assert inp.ndim == 2
            assert out.ndim == 2
    
    def test_empty_inputs(self):
        """Test TTT with empty inputs."""
        trainer = TestTimeTrainer()
        
        # Empty training pairs
        empty_pairs = []
        empty_programs = []
        
        # Should handle gracefully
        try:
            adapted_scorer = trainer.adapt_to_task(empty_pairs, empty_programs)
            # If it succeeds, check it returns something reasonable
            assert adapted_scorer is not None
        except Exception:
            # If it fails, that's also acceptable for empty inputs
            pass
    
    def test_single_program_adaptation(self):
        """Test adaptation with single program."""
        trainer = TestTimeTrainer()
        train_pairs = self.create_test_pairs()
        single_program = [self.create_test_programs()[0]]
        
        # Should work with single program
        adapted_scorer = trainer.adapt_to_task(train_pairs, single_program, num_iterations=2)
        assert adapted_scorer is not None
    
    def test_adaptation_consistency(self):
        """Test that adaptation is deterministic."""
        train_pairs = self.create_test_pairs()
        candidate_programs = self.create_test_programs()
        
        # Run adaptation twice with same inputs
        trainer1 = TestTimeTrainer()
        trainer1.adapt_to_task(train_pairs, candidate_programs, num_iterations=2)
        
        trainer2 = TestTimeTrainer()
        trainer2.adapt_to_task(train_pairs, candidate_programs, num_iterations=2)
        
        # Should produce same final weights (deterministic algorithm)
        weights1 = trainer1.adaptation_history[-1]['weights']
        weights2 = trainer2.adaptation_history[-1]['weights']
        
        assert np.allclose(weights1, weights2, atol=1e-6)
    
    def test_performance(self):
        """Test that TTT operations complete quickly."""
        import time
        
        trainer = TestTimeTrainer()
        train_pairs = self.create_test_pairs()
        candidate_programs = self.create_test_programs()
        
        # Time adaptation
        start_time = time.time()
        trainer.adapt_to_task(train_pairs, candidate_programs, num_iterations=5)
        adaptation_time = time.time() - start_time
        
        # Should be reasonably fast
        assert adaptation_time < 5.0, f"Adaptation took {adaptation_time:.3f}s, too slow"
        
        # Time individual scoring
        start_time = time.time()
        for program in candidate_programs:
            trainer.score_with_adaptation(program, train_pairs)
        scoring_time = time.time() - start_time
        
        assert scoring_time < 1.0, f"Scoring took {scoring_time:.3f}s, too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
