"""
Script extraction and consensus utilities for genomic analysis.

This module provides functions to extract mutational scripts from
sequence alignments and create consensus transformations across
multiple examples.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass

from .align import Alignment, Edit, EditType, detect_patterns_in_edits
from .tokenize import extract_color_from_token
from ..grid import Array


@dataclass
class MutationScript:
    """Represents a transformation script extracted from alignment."""
    mutations: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]
    
    def __str__(self):
        return f"MutationScript({len(self.mutations)} ops, conf={self.confidence:.2f})"


@dataclass
class GridTransformation:
    """Represents a high-level transformation operation on a grid."""
    op_type: str
    params: Dict[str, Any]
    affected_region: Optional[Tuple[int, int, int, int]]  # (top, left, height, width)
    confidence: float = 1.0
    
    def __str__(self):
        return f"{self.op_type}({self.params})"


def detect_global_transformations(input_grid: Array, output_grid: Array) -> List[Dict[str, Any]]:
    """Detect high-level transformations between grids before sequence analysis."""
    transformations = []
    
    # Size transformation detection
    h1, w1 = input_grid.shape
    h2, w2 = output_grid.shape
    
    if (h1, w1) != (h2, w2):
        # Significant size change - check for common patterns
        size_ratio = (h2 * w2) / (h1 * w1)
        
        if size_ratio < 0.3:  # Major compression
            # Check for pattern extraction
            if h2 < 10 and w2 < 10:
                transformations.append({
                    'type': 'compress_to_pattern',
                    'method': 'extract_core',
                    'confidence': 0.8
                })
            
            # Check for tiling pattern
            if h1 % h2 == 0 and w1 % w2 == 0:
                tile_h, tile_w = h1 // h2, w1 // w2
                if tile_h > 1 and tile_w > 1:
                    transformations.append({
                        'type': 'extract_tile',
                        'tile_size': (tile_h, tile_w),
                        'confidence': 0.9
                    })
        
        elif size_ratio > 3.0:  # Major expansion
            # Check if output is tiled version of input
            if h2 % h1 == 0 and w2 % w1 == 0:
                transformations.append({
                    'type': 'tile_expansion',
                    'factor': (h2 // h1, w2 // w1),
                    'confidence': 0.9
                })
    
    else:
        # Same size - check for geometric transformations
        if np.array_equal(input_grid, np.fliplr(output_grid)):
            transformations.append({
                'type': 'horizontal_flip',
                'confidence': 1.0
            })
        elif np.array_equal(input_grid, np.flipud(output_grid)):
            transformations.append({
                'type': 'vertical_flip',
                'confidence': 1.0
            })
        elif np.array_equal(input_grid, np.rot90(output_grid, 1)):
            transformations.append({
                'type': 'rotate_90',
                'confidence': 1.0
            })
        elif np.array_equal(input_grid, np.rot90(output_grid, 2)):
            transformations.append({
                'type': 'rotate_180',
                'confidence': 1.0
            })
        elif np.array_equal(input_grid, np.rot90(output_grid, 3)):
            transformations.append({
                'type': 'rotate_270',
                'confidence': 1.0
            })
    
    # Color transformation detection
    from collections import Counter
    input_colors = Counter(input_grid.flatten())
    output_colors = Counter(output_grid.flatten())
    
    if sorted(input_colors.values()) == sorted(output_colors.values()):
        transformations.append({
            'type': 'color_permutation',
            'confidence': 0.9
        })
    elif len(input_colors) != len(output_colors):
        transformations.append({
            'type': 'color_change',
            'confidence': 0.7
        })
    
    return transformations


def infer_script(input_grid: Array, output_grid: Array, 
                alignment_method: str = "needleman_wunsch") -> MutationScript:
    """
    Infer a transformation script from input to output grid.
    
    Args:
        input_grid: Source grid
        output_grid: Target grid  
        alignment_method: Alignment algorithm to use
    
    Returns:
        Extracted mutation script
    """
    # First try to detect high-level transformations
    global_transforms = detect_global_transformations(input_grid, output_grid)
    
    # If we found a high-confidence global transformation, use it
    for transform in global_transforms:
        if transform['confidence'] >= 0.95:
            return MutationScript(
                mutations=[transform],
                confidence=transform['confidence'],
                metadata={
                    'method': 'global_detection',
                    'input_shape': input_grid.shape,
                    'output_shape': output_grid.shape
                }
            )
    
    # Otherwise, proceed with sequence analysis
    from .hilbert import grid_to_hilbert_sequence, hilbert_order
    from .tokenize import tokenize_sequence, run_length_encode
    from .align import needleman_wunsch, smith_waterman
    
    # Convert grids to sequences
    h1, w1 = input_grid.shape
    h2, w2 = output_grid.shape
    
    coords1 = hilbert_order(h1, w1)
    coords2 = hilbert_order(h2, w2)
    
    seq1 = grid_to_hilbert_sequence(input_grid)
    seq2 = grid_to_hilbert_sequence(output_grid)
    
    # Tokenize sequences
    tokens1 = tokenize_sequence(seq1, input_grid, coords1)
    tokens2 = tokenize_sequence(seq2, output_grid, coords2)
    
    # Perform alignment
    if alignment_method == "smith_waterman":
        alignment = smith_waterman(tokens1, tokens2)
    else:
        alignment = needleman_wunsch(tokens1, tokens2)
    
    # Extract mutations from alignment
    mutations = _extract_mutations_from_alignment(alignment, input_grid, output_grid)
    
    # Merge with global transformations
    all_mutations = global_transforms + mutations
    
    # Analyze patterns
    patterns = detect_patterns_in_edits(alignment.get_edit_script())
    
    # Compute confidence based on alignment quality and global transforms
    edit_script = alignment.get_edit_script()
    total_ops = len(alignment.edits)
    error_ops = len([e for e in edit_script if e.edit_type != EditType.MATCH])
    alignment_confidence = 1.0 - (error_ops / max(1, total_ops))
    
    # Boost confidence if we have high-confidence global transforms
    global_confidence = max([t['confidence'] for t in global_transforms], default=0.0)
    combined_confidence = max(alignment_confidence, global_confidence * 0.8)
    
    metadata = {
        'alignment_score': alignment.score,
        'input_shape': input_grid.shape,
        'output_shape': output_grid.shape,
        'patterns': patterns,
        'total_edits': len(edit_script),
        'global_transforms': global_transforms
    }
    
    return MutationScript(all_mutations, combined_confidence, metadata)


def _extract_mutations_from_alignment(alignment: Alignment, input_grid: Array, 
                                    output_grid: Array) -> List[Dict[str, Any]]:
    """Extract structured mutations from sequence alignment."""
    mutations = []
    
    for edit in alignment.get_edit_script():
        mutation = {
            'type': edit.edit_type.value,
            'position': edit.pos1 if edit.pos1 >= 0 else edit.pos2,
        }
        
        if edit.edit_type == EditType.SUBSTITUTION:
            old_color = extract_color_from_token(edit.token1) if edit.token1 else 0
            new_color = extract_color_from_token(edit.token2) if edit.token2 else 0
            mutation.update({
                'old_value': old_color,
                'new_value': new_color,
                'old_token': edit.token1,
                'new_token': edit.token2
            })
        
        elif edit.edit_type == EditType.INSERTION:
            new_color = extract_color_from_token(edit.token2) if edit.token2 else 0
            mutation.update({
                'value': new_color,
                'token': edit.token2
            })
        
        elif edit.edit_type == EditType.DELETION:
            old_color = extract_color_from_token(edit.token1) if edit.token1 else 0
            mutation.update({
                'value': old_color,
                'token': edit.token1
            })
        
        mutations.append(mutation)
    
    return mutations


def consensus_script(scripts: List[MutationScript]) -> MutationScript:
    """
    Create a consensus transformation script from multiple examples.
    
    Args:
        scripts: List of mutation scripts from different examples
    
    Returns:
        Consensus script that generalizes across examples
    """
    if not scripts:
        return MutationScript([], 0.0, {})
    
    if len(scripts) == 1:
        return scripts[0]
    
    # Collect all mutations by type
    mutations_by_type = defaultdict(list)
    for script in scripts:
        for mutation in script.mutations:
            mutations_by_type[mutation['type']].append(mutation)
    
    # Find consensus mutations
    consensus_mutations = []
    
    # For substitutions, look for consistent color mappings
    if 'SUB' in mutations_by_type:
        color_mappings = Counter()
        for mut in mutations_by_type['SUB']:
            mapping = (mut.get('old_value', 0), mut.get('new_value', 0))
            color_mappings[mapping] += 1
        
        # Keep mappings that appear in most examples
        threshold = len(scripts) // 2 + 1
        for (old_color, new_color), count in color_mappings.items():
            if count >= threshold:
                consensus_mutations.append({
                    'type': 'RECOLOR',
                    'old_color': old_color,
                    'new_color': new_color,
                    'support': count / len(scripts)
                })
    
    # For insertions/deletions, look for systematic patterns
    if 'INS' in mutations_by_type or 'DEL' in mutations_by_type:
        # Analyze if there's a consistent size change
        size_changes = []
        for script in scripts:
            ins_count = len([m for m in script.mutations if m['type'] == 'INS'])
            del_count = len([m for m in script.mutations if m['type'] == 'DEL'])
            size_changes.append(ins_count - del_count)
        
        if len(set(size_changes)) == 1:  # Consistent size change
            change = size_changes[0]
            if change > 0:
                consensus_mutations.append({
                    'type': 'EXPAND',
                    'amount': change,
                    'support': 1.0
                })
            elif change < 0:
                consensus_mutations.append({
                    'type': 'CONTRACT',
                    'amount': -change,
                    'support': 1.0
                })
    
    # Compute overall confidence
    avg_confidence = np.mean([script.confidence for script in scripts])
    consensus_confidence = avg_confidence * (len(consensus_mutations) / max(1, len(scripts)))
    
    metadata = {
        'num_examples': len(scripts),
        'avg_input_confidence': avg_confidence,
        'mutation_types': list(mutations_by_type.keys())
    }
    
    return MutationScript(consensus_mutations, consensus_confidence, metadata)


def apply_recipe(recipe: MutationScript, input_grid: Array) -> Array:
    """
    Apply a mutation recipe to transform an input grid.
    
    Args:
        recipe: Transformation script to apply
        input_grid: Grid to transform
    
    Returns:
        Transformed grid
    """
    result = input_grid.copy()
    
    for mutation in recipe.mutations:
        result = _apply_single_mutation(result, mutation)
        if result is None:
            return input_grid  # Fallback on failure
    
    return result


def _apply_single_mutation(grid: Array, mutation: Dict[str, Any]) -> Optional[Array]:
    """Apply a single mutation to a grid."""
    try:
        mutation_type = mutation.get('type', mutation.get('kind', 'unknown'))
        
        if mutation_type == 'RECOLOR':
            old_color = mutation['old_color']
            new_color = mutation['new_color']
            result = grid.copy()
            result[grid == old_color] = new_color
            return result
        
        elif mutation_type == 'horizontal_flip':
            return np.fliplr(grid)
        
        elif mutation_type == 'vertical_flip':
            return np.flipud(grid)
        
        elif mutation_type == 'rotate_90':
            return np.rot90(grid, 1)
        
        elif mutation_type == 'rotate_180':
            return np.rot90(grid, 2)
        
        elif mutation_type == 'rotate_270':
            return np.rot90(grid, 3)
        
        elif mutation_type == 'extract_tile':
            # Extract a repeating tile from the grid
            tile_h, tile_w = mutation.get('tile_size', (1, 1))
            h, w = grid.shape
            if h >= tile_h and w >= tile_w:
                return grid[:tile_h, :tile_w]
            return grid
        
        elif mutation_type == 'tile_expansion':
            # Expand grid by tiling
            factor_h, factor_w = mutation.get('factor', (2, 2))
            h, w = grid.shape
            result = np.zeros((h * factor_h, w * factor_w), dtype=grid.dtype)
            for i in range(factor_h):
                for j in range(factor_w):
                    result[i*h:(i+1)*h, j*w:(j+1)*w] = grid
            return result
        
        elif mutation_type == 'compress_to_pattern':
            # Extract core pattern from large grid
            method = mutation.get('method', 'extract_core')
            h, w = grid.shape
            
            if method == 'extract_core':
                # Extract central region
                if h > 10 or w > 10:
                    center_h, center_w = max(h // 4, 3), max(w // 4, 3)
                    top = h // 2 - center_h // 2
                    left = w // 2 - center_w // 2
                    return grid[top:top + center_h, left:left + center_w]
                else:
                    return grid
            else:
                return grid
        
        elif mutation_type == 'color_permutation':
            # Apply intelligent color permutation
            from collections import Counter
            colors = list(set(grid.flatten()))
            if len(colors) == 2:
                # Simple binary swap
                result = grid.copy()
                result[grid == colors[0]] = colors[1] + 10  # Temp value
                result[grid == colors[1]] = colors[0]
                result[result == colors[1] + 10] = colors[1]
                return result
            else:
                # For now, just return original
                return grid
        
        elif mutation_type == 'EXPAND':
            # Simple expansion - duplicate the grid
            amount = mutation.get('amount', 1)
            if amount == 1:
                # Double grid size
                h, w = grid.shape
                result = np.zeros((h * 2, w * 2), dtype=grid.dtype)
                result[:h, :w] = grid
                result[h:, :w] = grid
                result[:h, w:] = grid
                result[h:, w:] = grid
                return result
            else:
                return grid  # More complex expansion not implemented
        
        elif mutation_type == 'CONTRACT':
            # Simple contraction - take top-left quadrant
            h, w = grid.shape
            if h >= 2 and w >= 2:
                return grid[:h//2, :w//2]
            else:
                return grid
        
        else:
            # Unknown mutation type - return original grid
            return grid
    
    except Exception:
        return None


def extract_grid_transformations(script: MutationScript, 
                                input_grid: Array, 
                                output_grid: Array) -> List[GridTransformation]:
    """
    Convert a mutation script into high-level grid transformations.
    
    Args:
        script: Low-level mutation script
        input_grid: Original grid
        output_grid: Target grid
    
    Returns:
        List of high-level transformations
    """
    transformations = []
    
    # Analyze shape changes
    if input_grid.shape != output_grid.shape:
        h1, w1 = input_grid.shape
        h2, w2 = output_grid.shape
        
        if h2 == h1 * 2 and w2 == w1 * 2:
            transformations.append(GridTransformation(
                "resize", {"scale": 2.0}, (0, 0, h2, w2), 0.8
            ))
        elif h2 == h1 // 2 and w2 == w1 // 2:
            transformations.append(GridTransformation(
                "resize", {"scale": 0.5}, (0, 0, h2, w2), 0.8
            ))
        else:
            transformations.append(GridTransformation(
                "resize", {"new_shape": (h2, w2)}, (0, 0, h2, w2), 0.6
            ))
    
    # Analyze color changes
    color_changes = defaultdict(int)
    for mutation in script.mutations:
        if mutation['type'] == 'RECOLOR':
            old_color = mutation['old_color']
            new_color = mutation['new_color']
            color_changes[(old_color, new_color)] += 1
    
    for (old_color, new_color), count in color_changes.items():
        confidence = min(1.0, count / 10.0)  # Arbitrary scaling
        transformations.append(GridTransformation(
            "recolor", 
            {"from": old_color, "to": new_color}, 
            None, 
            confidence
        ))
    
    # Analyze symmetry changes
    if _has_symmetry(input_grid) != _has_symmetry(output_grid):
        if _has_symmetry(output_grid):
            transformations.append(GridTransformation(
                "reflect", {"axis": "auto"}, None, 0.7
            ))
    
    return transformations


def _has_symmetry(grid: Array) -> bool:
    """Quick check if grid has obvious symmetry."""
    h, w = grid.shape
    
    # Check horizontal symmetry
    if np.array_equal(grid, np.fliplr(grid)):
        return True
    
    # Check vertical symmetry
    if np.array_equal(grid, np.flipud(grid)):
        return True
    
    return False


def script_to_description(script: MutationScript) -> str:
    """
    Convert a mutation script to a human-readable description.
    
    Args:
        script: Mutation script
    
    Returns:
        Text description of the transformations
    """
    if not script.mutations:
        return "Identity transformation (no changes)"
    
    descriptions = []
    
    # Group mutations by type
    by_type = defaultdict(list)
    for mutation in script.mutations:
        by_type[mutation['type']].append(mutation)
    
    for mut_type, muts in by_type.items():
        if mut_type == 'RECOLOR':
            color_mappings = [(m['old_color'], m['new_color']) for m in muts]
            mappings_str = ", ".join(f"{old}→{new}" for old, new in color_mappings)
            descriptions.append(f"Recolor: {mappings_str}")
        
        elif mut_type == 'EXPAND':
            amounts = [m.get('amount', 1) for m in muts]
            avg_amount = np.mean(amounts)
            descriptions.append(f"Expand by factor {avg_amount:.1f}")
        
        elif mut_type == 'CONTRACT':
            amounts = [m.get('amount', 1) for m in muts]
            avg_amount = np.mean(amounts)
            descriptions.append(f"Contract by factor {avg_amount:.1f}")
        
        else:
            descriptions.append(f"{mut_type}: {len(muts)} operations")
    
    result = "; ".join(descriptions)
    if script.confidence < 1.0:
        result += f" (confidence: {script.confidence:.2f})"
    
    return result