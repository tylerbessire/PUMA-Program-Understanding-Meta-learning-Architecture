"""
Tests for neural guidance component.

This module tests the neural guidance system that predicts which DSL operations
are likely relevant for solving a given ARC task.
"""

import pytest
import numpy as np
from typing import Dict, List, Any, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import extract_task_features
from arc_solver.guidance import NeuralGuidance, SimpleClassifier, HeuristicGuidance


class TestNeuralGuidance:
    """Test suite for neural guidance components."""
    
    def create_rotation_task_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training pairs for a rotation task."""
        input_grid = to_array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0]
        ])
        output_grid = to_array([
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0]
        ])
        return [(input_grid, output_grid)]
    
    def create_recolor_task_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training pairs for a recoloring task."""
        input_grid = to_array([
            [1, 2, 0],
            [2, 1, 0],
            [0, 0, 0]
        ])
        output_grid = to_array([
            [2, 1, 0],
            [1, 2, 0],
            [0, 0, 0]
        ])
        return [(input_grid, output_grid)]
    
    def test_feature_extraction(self):
        """Test that feature extraction works correctly."""
        train_pairs = self.create_rotation_task_pairs()
        features = extract_task_features(train_pairs)
        
        # Check that we get expected features
        assert isinstance(features, dict)
        assert 'num_train_pairs' in features
        assert 'shape_preserved' in features
        assert 'likely_rotation' in features
        assert 'input_colors_mean' in features
        
        # Check feature values make sense
        assert features['num_train_pairs'] == 1
        assert features['shape_preserved'] == True
        assert isinstance(features['likely_rotation'], (int, float))
    
    def test_simple_classifier_creation(self):
        """Test that SimpleClassifier can be created and used."""
        classifier = SimpleClassifier(input_dim=17)
        
        # Check basic properties
        assert classifier.input_dim == 17
        assert classifier.hidden_dim == 32
        assert len(classifier.operations) == 7
        assert 'rotate' in classifier.operations
        assert 'recolor' in classifier.operations
    
    def test_simple_classifier_forward_pass(self):
        """Test SimpleClassifier forward pass."""
        classifier = SimpleClassifier(input_dim=17)
        
        # Create dummy input
        dummy_input = np.random.rand(1, 17)
        output = classifier.forward(dummy_input)
        
        # Check output properties
        assert output.shape == (7,)  # 7 operations
        assert np.all(output >= 0) and np.all(output <= 1)  # Sigmoid bounds
    
    def test_simple_classifier_prediction(self):
        """Test SimpleClassifier operation prediction."""
        classifier = SimpleClassifier(input_dim=17)
        
        # Create features for rotation task
        train_pairs = self.create_rotation_task_pairs()
        features = extract_task_features(train_pairs)
        
        # Predict operations
        predicted_ops = classifier.predict_operations(features)
        
        # Should return list of operation names
        assert isinstance(predicted_ops, list)
        assert all(isinstance(op, str) for op in predicted_ops)
        assert all(op in classifier.operations or op == 'identity' for op in predicted_ops)
    
    def test_heuristic_guidance(self):
        """Test heuristic-based guidance."""
        guidance = HeuristicGuidance()
        
        # Test rotation task
        train_pairs = self.create_rotation_task_pairs()
        features = extract_task_features(train_pairs)
        predicted_ops = guidance.predict_operations(features)
        
        assert isinstance(predicted_ops, list)
        assert len(predicted_ops) > 0
        
        # For rotation task, should predict rotation-related operations
        # Note: This is heuristic-based, so we mainly test it doesn't crash
    
    def test_neural_guidance_interface(self):
        """Test main NeuralGuidance interface."""
        guidance = NeuralGuidance()
        
        # Test with rotation task
        train_pairs = self.create_rotation_task_pairs()
        predicted_ops = guidance.predict_operations(train_pairs)
        
        assert isinstance(predicted_ops, list)
        assert len(predicted_ops) > 0
        assert all(isinstance(op, str) for op in predicted_ops)
    
    def test_operation_scoring(self):
        """Test operation relevance scoring."""
        guidance = NeuralGuidance()
        
        # Test with rotation task
        train_pairs = self.create_rotation_task_pairs()
        scores = guidance.score_operations(train_pairs)
        
        assert isinstance(scores, dict)
        assert 'rotate' in scores
        assert 'flip' in scores
        assert 'recolor' in scores
        
        # Scores should be between 0 and 1
        for op, score in scores.items():
            assert 0 <= score <= 1, f"Score for {op} is {score}, should be in [0,1]"
    
    def test_different_task_types(self):
        """Test guidance on different task types."""
        guidance = NeuralGuidance()
        
        # Test rotation task
        rotation_pairs = self.create_rotation_task_pairs()
        rotation_ops = guidance.predict_operations(rotation_pairs)
        rotation_scores = guidance.score_operations(rotation_pairs)
        
        # Test recolor task
        recolor_pairs = self.create_recolor_task_pairs()
        recolor_ops = guidance.predict_operations(recolor_pairs)
        recolor_scores = guidance.score_operations(recolor_pairs)
        
        # Both should produce valid results
        assert isinstance(rotation_ops, list) and len(rotation_ops) > 0
        assert isinstance(recolor_ops, list) and len(recolor_ops) > 0
        assert isinstance(rotation_scores, dict) and len(rotation_scores) > 0
        assert isinstance(recolor_scores, dict) and len(recolor_scores) > 0
        
        # Scores might be different for different task types
        # (though with heuristics, the difference might be small)
    
    def test_edge_cases(self):
        """Test guidance with edge cases."""
        guidance = NeuralGuidance()
        
        # Empty train pairs
        empty_pairs = []
        try:
            result = guidance.predict_operations(empty_pairs)
            # Should handle gracefully, possibly returning default operations
            assert isinstance(result, list)
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
        
        # Single pixel grids
        tiny_pairs = [(to_array([[1]]), to_array([[2]]))]
        result = guidance.predict_operations(tiny_pairs)
        assert isinstance(result, list)
    
    def test_guidance_consistency(self):
        """Test that guidance produces consistent results."""
        guidance = NeuralGuidance()
        train_pairs = self.create_rotation_task_pairs()
        
        # Run prediction multiple times
        results = [guidance.predict_operations(train_pairs) for _ in range(3)]
        
        # Results should be consistent (same algorithm, same input)
        for i in range(1, len(results)):
            assert results[i] == results[0], f"Inconsistent results on run {i}"
    
    def test_feature_robustness(self):
        """Test that feature extraction is robust."""
        # Test with various grid sizes
        test_pairs = [
            # Small grid
            [(to_array([[1]]), to_array([[2]]))],
            # Rectangular grid
            [(to_array([[1, 2, 3]]), to_array([[3, 2, 1]]))],
            # Larger grid
            [(to_array([[i % 3 for i in range(5)] for _ in range(4)]), 
              to_array([[i % 3 for i in range(5)] for _ in range(4)]))],
        ]
        
        for pairs in test_pairs:
            features = extract_task_features(pairs)
            assert isinstance(features, dict)
            assert 'num_train_pairs' in features
            assert features['num_train_pairs'] == 1
    
    def test_performance(self):
        """Test that guidance operations complete quickly."""
        import time
        
        guidance = NeuralGuidance()
        train_pairs = self.create_rotation_task_pairs()
        
        # Time operation prediction
        start_time = time.time()
        predicted_ops = guidance.predict_operations(train_pairs)
        prediction_time = time.time() - start_time
        
        # Time scoring
        start_time = time.time()
        scores = guidance.score_operations(train_pairs)
        scoring_time = time.time() - start_time
        
        # Should be fast (< 1 second each)
        assert prediction_time < 1.0, f"Prediction took {prediction_time:.3f}s"
        assert scoring_time < 1.0, f"Scoring took {scoring_time:.3f}s"
        
        # Results should be valid
        assert isinstance(predicted_ops, list)
        assert isinstance(scores, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
