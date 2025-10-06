"""
Hilbert curve utilities for 2D to 1D space-filling transformations.

This module provides functions to convert between 2D grid coordinates
and 1D Hilbert curve positions, enabling sequence-based analysis of
spatial patterns.
"""

from typing import List, Tuple, Dict
import numpy as np


def hilbert_order(height: int, width: int) -> List[Tuple[int, int]]:
    """
    Generate Hilbert curve ordering for a grid of given dimensions.
    
    For non-square grids, we use the larger dimension to determine
    the Hilbert curve size and map coordinates accordingly.
    
    Args:
        height: Grid height
        width: Grid width
    
    Returns:
        List of (y, x) coordinates in Hilbert curve order
    """
    # Find the smallest power of 2 that contains both dimensions
    max_dim = max(height, width)
    n = 1
    while n < max_dim:
        n *= 2
    
    # Generate Hilbert curve for n x n grid
    hilbert_coords = []
    for i in range(n * n):
        y, x = hilbert_index_to_xy(i, n)
        if y < height and x < width:
            hilbert_coords.append((y, x))
    
    return hilbert_coords


def hilbert_index_to_xy(index: int, n: int) -> Tuple[int, int]:
    """
    Convert Hilbert curve index to (y, x) coordinates.
    
    Args:
        index: Position along the Hilbert curve
        n: Size of the grid (must be power of 2)
    
    Returns:
        (y, x) coordinates
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    
    x = y = 0
    s = 1
    t = index
    
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    
    return (y, x)


def hilbert_xy_to_index(y: int, x: int, n: int) -> int:
    """
    Convert (y, x) coordinates to Hilbert curve index.
    
    Args:
        y: Y coordinate
        x: X coordinate  
        n: Size of the grid (must be power of 2)
    
    Returns:
        Position along the Hilbert curve
    """
    assert n > 0 and (n & (n - 1)) == 0, "n must be a power of 2"
    
    d = 0
    s = n // 2
    
    while s > 0:
        rx = 1 if x & s else 0
        ry = 1 if y & s else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _hilbert_rot(s, x, y, rx, ry)
        s //= 2
    
    return d


def _hilbert_rot(n: int, x: int, y: int, rx: int, ry: int) -> Tuple[int, int]:
    """Rotate/flip a quadrant appropriately for Hilbert curve."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        # Swap x and y
        x, y = y, x
    return (x, y)


def grid_to_hilbert_sequence(grid: np.ndarray) -> List[int]:
    """
    Convert a 2D grid to a 1D sequence following Hilbert curve order.
    
    Args:
        grid: 2D numpy array
    
    Returns:
        List of values in Hilbert curve order
    """
    height, width = grid.shape
    coords = hilbert_order(height, width)
    
    sequence = []
    for y, x in coords:
        sequence.append(int(grid[y, x]))
    
    return sequence


def hilbert_sequence_to_grid(sequence: List[int], height: int, width: int) -> np.ndarray:
    """
    Convert a 1D sequence back to a 2D grid using Hilbert curve order.
    
    Args:
        sequence: 1D sequence of values
        height: Target grid height
        width: Target grid width
    
    Returns:
        2D numpy array
    """
    coords = hilbert_order(height, width)
    grid = np.zeros((height, width), dtype=np.int16)
    
    for i, (y, x) in enumerate(coords):
        if i < len(sequence):
            grid[y, x] = sequence[i]
    
    return grid


def morton_order(height: int, width: int) -> List[Tuple[int, int]]:
    """
    Alternative space-filling curve (Morton/Z-order) for comparison.
    Can be used as a fallback when Hilbert curve doesn't work well.
    """
    coords = []
    max_dim = max(height, width)
    n = 1
    while n < max_dim:
        n *= 2
    
    for i in range(n * n):
        y, x = _morton_decode(i)
        if y < height and x < width:
            coords.append((y, x))
    
    return coords


def _morton_decode(z: int) -> Tuple[int, int]:
    """Decode Morton (Z-order) index to (y, x) coordinates."""
    x = y = 0
    for i in range(16):  # Sufficient for reasonable grid sizes
        x |= (z & (1 << (2 * i))) >> i
        y |= (z & (1 << (2 * i + 1))) >> (i + 1)
    return (y, x)


def compare_curves(grid: np.ndarray) -> Dict[str, float]:
    """
    Compare different space-filling curves on a grid to find the best one.
    Returns metrics for each curve type.
    """
    from collections import Counter
    
    # Get sequences for different curves
    hilbert_seq = grid_to_hilbert_sequence(grid)
    morton_coords = morton_order(*grid.shape)
    morton_seq = [int(grid[y, x]) for y, x in morton_coords]
    
    # Simple row-major order for comparison
    row_major_seq = grid.flatten().tolist()
    
    def sequence_smoothness(seq: List[int]) -> float:
        """Measure how smooth a sequence is (fewer transitions = smoother)."""
        if len(seq) <= 1:
            return 1.0
        transitions = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
        return 1.0 - (transitions / (len(seq) - 1))
    
    def sequence_entropy(seq: List[int]) -> float:
        """Measure entropy of the sequence."""
        if not seq:
            return 0.0
        counter = Counter(seq)
        total = len(seq)
        entropy = -sum((count / total) * np.log2(count / total) 
                      for count in counter.values())
        return entropy
    
    return {
        'hilbert_smoothness': sequence_smoothness(hilbert_seq),
        'morton_smoothness': sequence_smoothness(morton_seq),
        'row_major_smoothness': sequence_smoothness(row_major_seq),
        'hilbert_entropy': sequence_entropy(hilbert_seq),
        'morton_entropy': sequence_entropy(morton_seq),
        'row_major_entropy': sequence_entropy(row_major_seq),
    }