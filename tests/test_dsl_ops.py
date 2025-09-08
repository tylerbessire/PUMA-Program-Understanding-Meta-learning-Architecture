"""
Tests for DSL operations.

This module tests all domain-specific language operations used by the ARC solver
to ensure they work correctly and handle edge cases appropriately.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.dsl import *


class TestDSLOperations:
    """Test suite for DSL operations."""
    
    def test_identity_operation(self):
        """Test identity operation."""
        grid = to_array([[1, 2], [3, 4]])
        result = identity(grid)
        
        assert np.array_equal(result, grid)
        assert result is not grid  # Should be a copy
    
    def test_rotate_operation(self):
        """Test rotation operations."""
        grid = to_array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        # Rotate 90 degrees clockwise
        result90 = rotate(grid, k=1)
        expected90 = to_array([
            [4, 1],
            [5, 2],
            [6, 3]
        ])
        assert np.array_equal(result90, expected90)
        
        # Rotate 180 degrees
        result180 = rotate(grid, k=2)
        expected180 = to_array([
            [6, 5, 4],
            [3, 2, 1]
        ])
        assert np.array_equal(result180, expected180)
        
        # Rotate 270 degrees
        result270 = rotate(grid, k=3)
        expected270 = to_array([
            [3, 6],
            [2, 5],
            [1, 4]
        ])
        assert np.array_equal(result270, expected270)
        
        # Rotate 360 degrees (should equal original)
        result360 = rotate(grid, k=4)
        assert np.array_equal(result360, grid)
    
    def test_flip_operations(self):
        """Test flip operations."""
        grid = to_array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        # Flip vertically (axis=0)
        result_v = flip(grid, axis=0)
        expected_v = to_array([
            [4, 5, 6],
            [1, 2, 3]
        ])
        assert np.array_equal(result_v, expected_v)
        
        # Flip horizontally (axis=1)
        result_h = flip(grid, axis=1)
        expected_h = to_array([
            [3, 2, 1],
            [6, 5, 4]
        ])
        assert np.array_equal(result_h, expected_h)
    
    def test_transpose_operation(self):
        """Test transpose operation."""
        grid = to_array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        result = transpose(grid)
        expected = to_array([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        assert np.array_equal(result, expected)
        
        # Double transpose should return to original
        double_transpose = transpose(transpose(grid))
        assert np.array_equal(double_transpose, grid)
    
    def test_translate_operation(self):
        """Test translation operations."""
        grid = to_array([
            [1, 2, 0],
            [3, 4, 0],
            [0, 0, 0]
        ])
        
        # Translate right
        result_right = translate(grid, dx=1, dy=0)
        expected_right = to_array([
            [0, 1, 2],
            [0, 3, 4],
            [0, 0, 0]
        ])
        assert np.array_equal(result_right, expected_right)
        
        # Translate down
        result_down = translate(grid, dx=0, dy=1)
        expected_down = to_array([
            [0, 0, 0],
            [1, 2, 0],
            [3, 4, 0]
        ])
        assert np.array_equal(result_down, expected_down)
        
        # Translate diagonally
        result_diag = translate(grid, dx=1, dy=1)
        expected_diag = to_array([
            [0, 0, 0],
            [0, 1, 2],
            [0, 3, 4]
        ])
        assert np.array_equal(result_diag, expected_diag)
    
    def test_recolor_operation(self):
        """Test recoloring operations."""
        grid = to_array([
            [1, 2, 1],
            [2, 1, 2],
            [0, 0, 0]
        ])
        
        # Simple color mapping
        result = recolor(grid, color_map={1: 3, 2: 4})
        expected = to_array([
            [3, 4, 3],
            [4, 3, 4],
            [0, 0, 0]
        ])
        assert np.array_equal(result, expected)
        
        # Partial mapping (unmapped colors unchanged)
        result_partial = recolor(grid, color_map={1: 5})
        expected_partial = to_array([
            [5, 2, 5],
            [2, 5, 2],
            [0, 0, 0]
        ])
        assert np.array_equal(result_partial, expected_partial)
    
    def test_crop_operation(self):
        """Test cropping operations."""
        grid = to_array([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ])
        
        # Crop to content
        result = crop(grid, top=1, bottom=3, left=1, right=3)
        expected = to_array([
            [1, 2],
            [3, 4]
        ])
        assert np.array_equal(result, expected)
        
        # Crop with different bounds
        result2 = crop(grid, top=0, bottom=2, left=0, right=2)
        expected2 = to_array([
            [0, 0],
            [0, 1]
        ])
        assert np.array_equal(result2, expected2)
    
    def test_pad_operation(self):
        """Test padding operations."""
        grid = to_array([
            [1, 2],
            [3, 4]
        ])
        
        # Pad with zeros
        result = pad(grid, top=1, bottom=1, left=1, right=1, fill_value=0)
        expected = to_array([
            [0, 0, 0, 0],
            [0, 1, 2, 0],
            [0, 3, 4, 0],
            [0, 0, 0, 0]
        ])
        assert np.array_equal(result, expected)
        
        # Pad with different value
        result2 = pad(grid, top=1, bottom=0, left=0, right=1, fill_value=9)
        expected2 = to_array([
            [9, 9, 9],
            [1, 2, 9],
            [3, 4, 9]
        ])
        assert np.array_equal(result2, expected2)
    
    def test_apply_program(self):
        """Test program application."""
        grid = to_array([
            [1, 2],
            [3, 4]
        ])
        
        # Single operation program
        program1 = [("flip", {"axis": 0})]
        result1 = apply_program(grid, program1)
        expected1 = to_array([
            [3, 4],
            [1, 2]
        ])
        assert np.array_equal(result1, expected1)
        
        # Multi-operation program
        program2 = [("flip", {"axis": 0}), ("flip", {"axis": 1})]
        result2 = apply_program(grid, program2)
        expected2 = to_array([
            [4, 3],
            [2, 1]
        ])
        assert np.array_equal(result2, expected2)
        
        # Empty program (identity)
        program3 = []
        result3 = apply_program(grid, program3)
        assert np.array_equal(result3, grid)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        
        # Single pixel grid
        single_pixel = to_array([[5]])
        
        assert np.array_equal(rotate(single_pixel, k=1), single_pixel)
        assert np.array_equal(flip(single_pixel, axis=0), single_pixel)
        assert np.array_equal(transpose(single_pixel), single_pixel)
        
        # Empty operations should not crash
        result = recolor(single_pixel, color_map={})
        assert np.array_equal(result, single_pixel)
        
        # Large rotations
        grid = to_array([[1, 2], [3, 4]])
        result_large = rotate(grid, k=8)  # 8*90 = 720 degrees = 2*360
        assert np.array_equal(result_large, grid)
        
        # Negative rotations
        result_neg = rotate(grid, k=-1)  # -90 degrees = 270 degrees
        expected_neg = rotate(grid, k=3)
        assert np.array_equal(result_neg, expected_neg)
    
    def test_out_of_bounds_translation(self):
        """Test translation that moves content out of bounds."""
        grid = to_array([
            [1, 2],
            [3, 4]
        ])
        
        # Translate completely out of bounds
        result = translate(grid, dx=5, dy=5)
        expected = to_array([
            [0, 0],
            [0, 0]
        ])
        assert np.array_equal(result, expected)
        
        # Negative translation
        result_neg = translate(grid, dx=-1, dy=-1)
        expected_neg = to_array([
            [4, 0],
            [0, 0]
        ])
        assert np.array_equal(result_neg, expected_neg)
    
    def test_crop_bounds_validation(self):
        """Test crop operation bounds validation."""
        grid = to_array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        # Valid crop
        result = crop(grid, top=0, bottom=2, left=1, right=3)
        expected = to_array([
            [2, 3],
            [5, 6]
        ])
        assert np.array_equal(result, expected)
        
        # Crop to single cell
        result_single = crop(grid, top=1, bottom=2, left=1, right=2)
        expected_single = to_array([[5]])
        assert np.array_equal(result_single, expected_single)
    
    def test_program_with_invalid_operations(self):
        """Test program application with invalid operations."""
        grid = to_array([[1, 2], [3, 4]])
        
        # Invalid operation should raise error or be ignored
        program = [("invalid_op", {})]
        
        try:
            result = apply_program(grid, program)
            # If it doesn't raise an error, it should return unchanged grid
            assert np.array_equal(result, grid)
        except (KeyError, ValueError, AttributeError):
            # Raising an error is also acceptable
            pass
    
    def test_operation_immutability(self):
        """Test that operations don't modify input grids."""
        original = to_array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        grid = original.copy()
        
        # Apply various operations
        rotate(grid, k=1)
        flip(grid, axis=0)
        transpose(grid)
        translate(grid, dx=1, dy=1)
        recolor(grid, color_map={1: 9})
        crop(grid, top=0, bottom=1, left=0, right=2)
        pad(grid, top=1, bottom=1, left=1, right=1)
        
        # Original grid should be unchanged
        assert np.array_equal(grid, original)
    
    def test_composition_properties(self):
        """Test mathematical properties of operation compositions."""
        grid = to_array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        
        # Rotation composition: 4 * 90° = 360° = identity
        result = grid
        for _ in range(4):
            result = rotate(result, k=1)
        assert np.array_equal(result, grid)
        
        # Flip composition: two flips = identity
        result = flip(flip(grid, axis=0), axis=0)
        assert np.array_equal(result, grid)
        
        result = flip(flip(grid, axis=1), axis=1)
        assert np.array_equal(result, grid)
        
        # Transpose composition: two transposes = identity
        result = transpose(transpose(grid))
        assert np.array_equal(result, grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
