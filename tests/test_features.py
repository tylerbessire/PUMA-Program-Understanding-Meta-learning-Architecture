"""
Tests for feature extraction components.

This module tests the task feature extraction system that computes
numerical and categorical features from ARC training pairs.
"""

import pytest
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import extract_task_features, compute_task_signature, compute_numerical_features


class TestFeatures:
    """Test suite for feature extraction."""
    
    def create_rotation_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training pairs for a rotation task."""
        return [
            (to_array([[1, 0], [1, 1]]), to_array([[1, 1], [0, 1]])),
            (to_array([[2, 0], [2, 2]]), to_array([[2, 2], [0, 2]]))
        ]
    
    def create_recolor_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training pairs for a recoloring task."""
        return [
            (to_array([[1, 2], [2, 1]]), to_array([[2, 1], [1, 2]])),
            (to_array([[3, 4], [4, 3]]), to_array([[4, 3], [3, 4]]))
        ]
    
    def create_identity_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create training pairs for an identity task."""
        return [
            (to_array([[1, 2], [3, 4]]), to_array([[1, 2], [3, 4]])),
            (to_array([[5, 6], [7, 8]]), to_array([[5, 6], [7, 8]]))
        ]
    
    def test_basic_feature_extraction(self):
        """Test basic feature extraction functionality."""
        train_pairs = self.create_rotation_pairs()
        features = extract_task_features(train_pairs)
        
        # Check that we get a dictionary with expected keys
        assert isinstance(features, dict)
        assert 'num_train_pairs' in features
        assert 'input_height_mean' in features
        assert 'input_width_mean' in features
        assert 'shape_preserved' in features
        assert 'input_colors_mean' in features
        assert 'output_colors_mean' in features
        
        # Check basic values
        assert features['num_train_pairs'] == 2
        assert features['input_height_mean'] == 2.0
        assert features['input_width_mean'] == 2.0
    
    def test_shape_preservation_detection(self):
        """Test detection of shape preservation."""
        # Rotation preserves shape
        rotation_pairs = self.create_rotation_pairs()
        rotation_features = extract_task_features(rotation_pairs)
        assert rotation_features['shape_preserved'] == True
        
        # Identity preserves shape
        identity_pairs = self.create_identity_pairs()
        identity_features = extract_task_features(identity_pairs)
        assert identity_features['shape_preserved'] == True
        
        # Create shape-changing pairs
        shape_change_pairs = [
            (to_array([[1, 2]]), to_array([[1], [2]]))  # 1x2 -> 2x1
        ]
        shape_features = extract_task_features(shape_change_pairs)
        assert shape_features['shape_preserved'] == False
    
    def test_color_analysis(self):
        """Test color-related feature extraction."""
        train_pairs = self.create_recolor_pairs()
        features = extract_task_features(train_pairs)
        
        # Should detect color properties
        assert 'input_colors_mean' in features
        assert 'output_colors_mean' in features
        assert 'background_color_consistent' in features
        assert 'has_color_mapping' in features
        
        # Colors should be detected
        assert features['input_colors_mean'] > 0
        assert features['output_colors_mean'] > 0
    
    def test_transformation_likelihood_detection(self):
        """Test detection of likely transformations."""
        # Rotation task should show rotation likelihood
        rotation_pairs = self.create_rotation_pairs()
        rotation_features = extract_task_features(rotation_pairs)
        
        assert 'likely_rotation' in rotation_features
        assert 'likely_reflection' in rotation_features
        assert 'likely_translation' in rotation_features
        assert 'likely_recolor' in rotation_features
        
        # For rotation task, rotation likelihood should be high
        assert rotation_features['likely_rotation'] > 0.5
        
        # Recolor task should show recolor likelihood
        recolor_pairs = self.create_recolor_pairs()
        recolor_features = extract_task_features(recolor_pairs)
        assert recolor_features['likely_recolor'] > 0.5
    
    def test_object_counting(self):
        """Test object-related features."""
        train_pairs = self.create_rotation_pairs()
        features = extract_task_features(train_pairs)
        
        assert 'input_objects_mean' in features
        assert 'output_objects_mean' in features
        assert 'object_count_preserved' in features
        
        # Object counts should be reasonable
        assert features['input_objects_mean'] >= 0
        assert features['output_objects_mean'] >= 0
        assert isinstance(features['object_count_preserved'], (bool, int))
    
    def test_task_signature_computation(self):
        """Test task signature computation."""
        train_pairs = self.create_rotation_pairs()
        signature = compute_task_signature(train_pairs)
        
        assert isinstance(signature, str)
        assert len(signature) > 0
        
        # Same pairs should produce same signature
        signature2 = compute_task_signature(train_pairs)
        assert signature == signature2
        
        # Different pairs should produce different signatures
        different_pairs = self.create_recolor_pairs()
        different_signature = compute_task_signature(different_pairs)
        assert signature != different_signature
    
    def test_numerical_features(self):
        """Test numerical feature extraction."""
        train_pairs = self.create_rotation_pairs()
        numerical_features = compute_numerical_features(train_pairs)
        
        assert isinstance(numerical_features, np.ndarray)
        assert len(numerical_features) > 0
        assert np.all(np.isfinite(numerical_features))  # No NaN or inf values
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        empty_pairs = []
        
        try:
            features = extract_task_features(empty_pairs)
            # If it succeeds, should return some default features
            assert isinstance(features, dict)
            assert features.get('num_train_pairs', 0) == 0
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
    
    def test_single_pair_handling(self):
        """Test handling of single training pair."""
        single_pair = [self.create_rotation_pairs()[0]]
        features = extract_task_features(single_pair)
        
        assert isinstance(features, dict)
        assert features['num_train_pairs'] == 1
        assert 'shape_preserved' in features
        assert 'likely_rotation' in features
    
    def test_large_grid_handling(self):
        """Test feature extraction on larger grids."""
        large_grid_in = to_array([[i % 10 for i in range(20)] for _ in range(15)])
        large_grid_out = to_array([[i % 10 for i in range(20)] for _ in range(15)])
        large_pairs = [(large_grid_in, large_grid_out)]
        
        features = extract_task_features(large_pairs)
        
        assert isinstance(features, dict)
        assert features['input_height_mean'] == 15
        assert features['input_width_mean'] == 20
        assert features['shape_preserved'] == True
    
    def test_feature_consistency(self):
        """Test that features are computed consistently."""
        train_pairs = self.create_rotation_pairs()
        
        # Compute features multiple times
        features1 = extract_task_features(train_pairs)
        features2 = extract_task_features(train_pairs)
        
        # Should be identical
        assert features1.keys() == features2.keys()
        for key in features1.keys():
            if isinstance(features1[key], float):
                assert abs(features1[key] - features2[key]) < 1e-10
            else:
                assert features1[key] == features2[key]
    
    def test_different_task_types(self):
        """Test feature extraction on different task types."""
        task_types = [
            ("rotation", self.create_rotation_pairs()),
            ("recolor", self.create_recolor_pairs()),
            ("identity", self.create_identity_pairs())
        ]
        
        for task_name, pairs in task_types:
            features = extract_task_features(pairs)
            
            # All tasks should produce valid features
            assert isinstance(features, dict)
            assert len(features) > 10  # Should have many features
            
            # Basic properties should be consistent
            assert features['num_train_pairs'] == len(pairs)
            assert features['input_height_mean'] > 0
            assert features['input_width_mean'] > 0
    
    def test_extreme_cases(self):
        """Test feature extraction on extreme cases."""
        # Single pixel grids
        tiny_pairs = [(to_array([[1]]), to_array([[2]]))]
        tiny_features = extract_task_features(tiny_pairs)
        
        assert tiny_features['input_height_mean'] == 1
        assert tiny_features['input_width_mean'] == 1
        assert tiny_features['shape_preserved'] == True
        
        # Very sparse grids (mostly zeros)
        sparse_grid = np.zeros((10, 10))
        sparse_grid[5, 5] = 1
        sparse_pairs = [(sparse_grid, sparse_grid)]
        sparse_features = extract_task_features(sparse_pairs)
        
        assert sparse_features['input_height_mean'] == 10
        assert sparse_features['input_width_mean'] == 10
    
    def test_feature_bounds(self):
        """Test that features have reasonable bounds."""
        train_pairs = self.create_rotation_pairs()
        features = extract_task_features(train_pairs)
        
        # Boolean features should be 0 or 1
        boolean_features = ['shape_preserved', 'background_color_consistent', 
                          'has_color_mapping', 'object_count_preserved']
        
        for feat in boolean_features:
            if feat in features:
                assert features[feat] in [0, 1, True, False]
        
        # Likelihood features should be between 0 and 1
        likelihood_features = ['likely_rotation', 'likely_reflection', 
                             'likely_translation', 'likely_recolor']
        
        for feat in likelihood_features:
            if feat in features:
                assert 0 <= features[feat] <= 1
        
        # Count features should be non-negative
        count_features = ['num_train_pairs', 'input_colors_mean', 
                         'output_colors_mean', 'input_objects_mean', 'output_objects_mean']
        
        for feat in count_features:
            if feat in features:
                assert features[feat] >= 0
    
    def test_performance(self):
        """Test that feature extraction is fast enough."""
        import time
        
        train_pairs = self.create_rotation_pairs()
        
        start_time = time.time()
        features = extract_task_features(train_pairs)
        extraction_time = time.time() - start_time
        
        # Should be very fast
        assert extraction_time < 1.0, f"Feature extraction took {extraction_time:.3f}s"
        
        # Result should be valid
        assert isinstance(features, dict)
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
