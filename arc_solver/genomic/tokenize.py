"""
Tokenization utilities for genomic sequence analysis.

This module converts grid sequences into tokens that capture both color
and local neighborhood information, then applies run-length encoding
for compression and pattern recognition.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from collections import Counter


def get_neighborhood_signature(grid: np.ndarray, y: int, x: int, radius: int = 1) -> str:
    """
    Get a neighborhood signature for a pixel.
    
    Args:
        grid: 2D numpy array
        y, x: Pixel coordinates
        radius: Neighborhood radius (1 = 3x3, 2 = 5x5, etc.)
    
    Returns:
        String signature representing the neighborhood pattern
    """
    h, w = grid.shape
    signature = []
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                signature.append(str(int(grid[ny, nx])))
            else:
                signature.append('X')  # Out of bounds marker
    
    return ''.join(signature)


def get_edge_signature(grid: np.ndarray, y: int, x: int) -> str:
    """
    Get an edge signature indicating which directions have boundaries.
    
    Args:
        grid: 2D numpy array
        y, x: Pixel coordinates
    
    Returns:
        String indicating edge directions (N, S, E, W, or combinations)
    """
    h, w = grid.shape
    edges = []
    
    if y == 0:
        edges.append('N')
    if y == h - 1:
        edges.append('S')
    if x == 0:
        edges.append('W')
    if x == w - 1:
        edges.append('E')
    
    return ''.join(edges) if edges else 'I'  # I = Interior


def tokenize_sequence(sequence: List[int], grid: np.ndarray, 
                     coords: List[Tuple[int, int]]) -> List[str]:
    """
    Convert a sequence of colors to tokens incorporating neighborhood information.
    
    Args:
        sequence: 1D sequence of color values
        grid: Original 2D grid
        coords: Corresponding (y, x) coordinates for each sequence element
    
    Returns:
        List of token strings
    """
    tokens = []
    
    for i, color in enumerate(sequence):
        if i < len(coords):
            y, x = coords[i]
            
            # Basic token: color value
            token_parts = [str(color)]
            
            # Add neighborhood signature (simplified 3x3)
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue  # Skip center pixel
                    ny, nx = y + dy, x + dx
                    h, w = grid.shape
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbors.append(str(int(grid[ny, nx])))
                    else:
                        neighbors.append('X')
            
            # Compress neighborhood to avoid explosion of unique tokens
            neighbor_sig = _compress_neighborhood(neighbors)
            token_parts.append(neighbor_sig)
            
            # Add edge information
            edge_sig = get_edge_signature(grid, y, x)
            token_parts.append(edge_sig)
            
            token = ':'.join(token_parts)
            tokens.append(token)
        else:
            # Fallback for out-of-bounds
            tokens.append(str(color))
    
    return tokens


def _compress_neighborhood(neighbors: List[str]) -> str:
    """
    Compress neighborhood signature to reduce token vocabulary size.
    
    Args:
        neighbors: List of neighbor values
    
    Returns:
        Compressed signature
    """
    # Count occurrences of each neighbor value
    counts = Counter(neighbors)
    
    # Create a signature based on pattern
    if len(counts) == 1:
        return 'UNIFORM'
    elif len(counts) == 2:
        return 'BINARY'
    elif 'X' in counts:
        return 'BOUNDARY'
    else:
        return 'MIXED'


def run_length_encode(tokens: List[str]) -> List[Tuple[str, int]]:
    """
    Apply run-length encoding to a token sequence.
    
    Args:
        tokens: List of tokens
    
    Returns:
        List of (token, run_length) pairs
    """
    if not tokens:
        return []
    
    encoded = []
    current_token = tokens[0]
    run_length = 1
    
    for token in tokens[1:]:
        if token == current_token:
            run_length += 1
        else:
            encoded.append((current_token, run_length))
            current_token = token
            run_length = 1
    
    # Don't forget the last run
    encoded.append((current_token, run_length))
    return encoded


def run_length_decode(encoded: List[Tuple[str, int]]) -> List[str]:
    """
    Decode a run-length encoded sequence.
    
    Args:
        encoded: List of (token, run_length) pairs
    
    Returns:
        Decoded token sequence
    """
    decoded = []
    for token, length in encoded:
        decoded.extend([token] * length)
    return decoded


def create_token_vocabulary(token_sequences: List[List[str]]) -> Dict[str, int]:
    """
    Create a vocabulary mapping from tokens to indices.
    
    Args:
        token_sequences: List of token sequences
    
    Returns:
        Dictionary mapping tokens to unique indices
    """
    all_tokens = set()
    for seq in token_sequences:
        all_tokens.update(seq)
    
    # Sort tokens for deterministic ordering
    sorted_tokens = sorted(all_tokens)
    return {token: i for i, token in enumerate(sorted_tokens)}


def tokens_to_indices(tokens: List[str], vocabulary: Dict[str, int]) -> List[int]:
    """
    Convert tokens to indices using a vocabulary.
    
    Args:
        tokens: List of token strings
        vocabulary: Token to index mapping
    
    Returns:
        List of token indices
    """
    return [vocabulary.get(token, 0) for token in tokens]  # 0 = unknown token


def indices_to_tokens(indices: List[int], vocabulary: Dict[str, int]) -> List[str]:
    """
    Convert indices back to tokens using a vocabulary.
    
    Args:
        indices: List of token indices
        vocabulary: Token to index mapping
    
    Returns:
        List of token strings
    """
    reverse_vocab = {idx: token for token, idx in vocabulary.items()}
    return [reverse_vocab.get(idx, 'UNK') for idx in indices]


def analyze_token_patterns(token_sequences: List[List[str]]) -> Dict[str, Any]:
    """
    Analyze patterns in token sequences for insights.
    
    Args:
        token_sequences: List of token sequences
    
    Returns:
        Dictionary with pattern analysis results
    """
    all_tokens = []
    for seq in token_sequences:
        all_tokens.extend(seq)
    
    token_counts = Counter(all_tokens)
    vocab_size = len(token_counts)
    
    # Analyze run lengths after RLE
    all_runs = []
    for seq in token_sequences:
        rle = run_length_encode(seq)
        all_runs.extend([length for _, length in rle])
    
    return {
        'vocab_size': vocab_size,
        'most_common_tokens': token_counts.most_common(10),
        'avg_sequence_length': np.mean([len(seq) for seq in token_sequences]),
        'avg_run_length': np.mean(all_runs) if all_runs else 0,
        'max_run_length': max(all_runs) if all_runs else 0,
        'compression_ratio': len(all_tokens) / len(all_runs) if all_runs else 1.0
    }


def extract_color_from_token(token: str) -> int:
    """
    Extract the base color value from a token.
    
    Args:
        token: Token string
    
    Returns:
        Color value as integer
    """
    try:
        # Token format is "color:neighborhood:edge"
        parts = token.split(':')
        return int(parts[0])
    except (ValueError, IndexError):
        return 0  # Default fallback


def simplify_tokens(tokens: List[str], keep_neighborhood: bool = True, 
                   keep_edges: bool = False) -> List[str]:
    """
    Simplify tokens by removing some components.
    
    Args:
        tokens: List of tokens
        keep_neighborhood: Whether to keep neighborhood information
        keep_edges: Whether to keep edge information
    
    Returns:
        Simplified token list
    """
    simplified = []
    
    for token in tokens:
        parts = token.split(':')
        new_parts = [parts[0]]  # Always keep color
        
        if keep_neighborhood and len(parts) > 1:
            new_parts.append(parts[1])
        
        if keep_edges and len(parts) > 2:
            new_parts.append(parts[2])
        
        simplified.append(':'.join(new_parts))
    
    return simplified