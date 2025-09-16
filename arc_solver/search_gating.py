"""
Search gating system that prioritizes hypothesis generation based on task signatures.
Reduces search space by focusing on relevant operations.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .grid import Array

class SearchGate:
    """Controls hypothesis generation based on task signature analysis."""
    
    def __init__(self):
        self.operation_budgets = {
            'recolor': {'move': 2, 'copy': 1, 'recolor': 10},
            'extraction': {'extract': 8, 'crop': 5, 'move': 2},
            'expansion': {'pad': 5, 'tile': 5, 'move': 3},
            'rotation': {'rotate': 8, 'flip': 4, 'transpose': 2},
            'reflection': {'flip': 8, 'rotate': 4, 'transpose': 2},
            'complex_same_size': {'move': 5, 'copy': 3, 'recolor': 3, 'pattern': 5}
        }
        
    def filter_hypotheses(self, hypotheses: List[Any], task_signature: Dict[str, Any]) -> List[Any]:
        """Filter and prioritize hypotheses based on task signature."""
        primary_pattern = task_signature.get('primary_pattern', 'unknown')
        color_change = task_signature.get('color_change', False)
        size_change = task_signature.get('size_change')
        
        # Get operation budget for this pattern type
        budget = self.operation_budgets.get(primary_pattern, {})
        
        # Categorize hypotheses by operation type
        categorized = self._categorize_hypotheses(hypotheses)
        
        # Apply gating rules
        filtered = []
        
        # Rule 1: Recolor tasks - heavily favor recolor operations
        if primary_pattern == 'recolor' and color_change:
            filtered.extend(categorized.get('recolor', [])[:budget.get('recolor', 10)])
            filtered.extend(categorized.get('move', [])[:budget.get('move', 2)])
            filtered.extend(categorized.get('copy', [])[:budget.get('copy', 1)])
            
        # Rule 2: Size-change tasks - require size-changing operations
        elif size_change:
            if 'to_' in size_change:  # Reduction/extraction
                filtered.extend(categorized.get('extract', [])[:budget.get('extract', 8)])
                filtered.extend(categorized.get('crop', [])[:budget.get('crop', 5)])
            else:  # Expansion
                filtered.extend(categorized.get('pad', [])[:budget.get('pad', 5)])
                filtered.extend(categorized.get('tile', [])[:budget.get('tile', 5)])
            
            # Always include some basic operations
            filtered.extend(categorized.get('move', [])[:budget.get('move', 2)])
            
        # Rule 3: Rotation/reflection tasks
        elif primary_pattern in ['rotation', 'reflection']:
            if primary_pattern == 'rotation':
                filtered.extend(categorized.get('rotate', [])[:budget.get('rotate', 8)])
                filtered.extend(categorized.get('flip', [])[:budget.get('flip', 4)])
            else:
                filtered.extend(categorized.get('flip', [])[:budget.get('flip', 8)])
                filtered.extend(categorized.get('rotate', [])[:budget.get('rotate', 4)])
            
            filtered.extend(categorized.get('transpose', [])[:budget.get('transpose', 2)])
            
        # Rule 4: Complex same-size - balanced approach
        else:
            for op_type, limit in budget.items():
                filtered.extend(categorized.get(op_type, [])[:limit])
        
        # Always include some top-scoring hypotheses regardless of type
        if len(filtered) < 5:
            remaining = [h for h in hypotheses if h not in filtered]
            remaining.sort(key=lambda h: getattr(h, 'verification_score', 0), reverse=True)
            filtered.extend(remaining[:5-len(filtered)])
        
        return filtered
    
    def _categorize_hypotheses(self, hypotheses: List[Any]) -> Dict[str, List[Any]]:
        """Categorize hypotheses by operation type."""
        categories = {
            'recolor': [],
            'move': [],
            'copy': [],
            'rotate': [],
            'flip': [],
            'transpose': [],
            'extract': [],
            'crop': [],
            'pad': [],
            'tile': [],
            'pattern': [],
            'other': []
        }
        
        for hyp in hypotheses:
            category = self._get_hypothesis_category(hyp)
            categories[category].append(hyp)
        
        return categories
    
    def _get_hypothesis_category(self, hypothesis: Any) -> str:
        """Determine the category of a hypothesis."""
        # Check hypothesis name/description for operation type
        name = getattr(hypothesis, 'name', '')
        description = getattr(hypothesis, 'description', '')
        
        combined = (name + ' ' + description).lower()
        
        if any(word in combined for word in ['recolor', 'color', 'mapping']):
            return 'recolor'
        elif any(word in combined for word in ['move', 'translate', 'shift']):
            return 'move'
        elif any(word in combined for word in ['copy', 'duplicate']):
            return 'copy'
        elif any(word in combined for word in ['rotate', 'rotation']):
            return 'rotate'
        elif any(word in combined for word in ['flip', 'mirror', 'reflect']):
            return 'flip'
        elif any(word in combined for word in ['transpose']):
            return 'transpose'
        elif any(word in combined for word in ['extract', 'region', 'marked']):
            return 'extract'
        elif any(word in combined for word in ['crop', 'trim']):
            return 'crop'
        elif any(word in combined for word in ['pad', 'expand']):
            return 'pad'
        elif any(word in combined for word in ['tile', 'repeat']):
            return 'tile'
        elif any(word in combined for word in ['pattern', 'formula', 'composition']):
            return 'pattern'
        else:
            return 'other'


class BlockSizeNegotiator:
    """Handles block-size parameter negotiation for extraction tasks."""
    
    def __init__(self):
        pass
    
    def negotiate_block_size(self, input_shape: Tuple[int, int], 
                           target_shape: Tuple[int, int]) -> List[Dict[str, int]]:
        """Find block sizes that can transform input_shape to target_shape."""
        input_h, input_w = input_shape
        target_h, target_w = target_shape
        
        valid_block_sizes = []
        
        # Try different block sizes
        for block_h in range(1, min(input_h, 10) + 1):
            for block_w in range(1, min(input_w, 10) + 1):
                # Check if this block size can produce the target
                if self._can_produce_target(input_h, input_w, block_h, block_w, target_h, target_w):
                    valid_block_sizes.append({
                        'block_size': min(block_h, block_w),  # Common param name
                        'block_height': block_h,
                        'block_width': block_w,
                        'grid_h': input_h // block_h,
                        'grid_w': input_w // block_w
                    })
        
        # Sort by preference (smaller blocks first, then by closeness to target)
        valid_block_sizes.sort(key=lambda x: (
            x['block_size'], 
            abs(x['grid_h'] - target_h) + abs(x['grid_w'] - target_w)
        ))
        
        return valid_block_sizes[:5]  # Return top 5 candidates
    
    def _can_produce_target(self, input_h: int, input_w: int, 
                           block_h: int, block_w: int,
                           target_h: int, target_w: int) -> bool:
        """Check if block extraction can produce target dimensions."""
        # Grid dimensions after blocking
        grid_h = input_h // block_h
        grid_w = input_w // block_w
        
        # Direct match
        if grid_h == target_h and grid_w == target_w:
            return True
        
        # Allowing for slight cropping/padding
        if abs(grid_h - target_h) <= 1 and abs(grid_w - target_w) <= 1:
            return True
        
        return False


class TaskSignatureAnalyzer:
    """Enhanced task signature analysis for better pattern detection."""
    
    def __init__(self):
        pass
    
    def analyze_enhanced(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
        """Enhanced task signature analysis with better handling of inconsistent shapes."""
        if not train_pairs:
            return {}
        
        # Basic size analysis
        input_shapes = [inp.shape for inp, _ in train_pairs]
        output_shapes = [out.shape for _, out in train_pairs]
        
        # Check for consistent shapes
        consistent_input = len(set(input_shapes)) == 1
        consistent_output = len(set(output_shapes)) == 1
        
        signature = {
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'consistent_input': consistent_input,
            'consistent_output': consistent_output,
        }
        
        # CRITICAL FIX: Handle inconsistent training data better
        if consistent_input and consistent_output:
            input_shape = input_shapes[0]
            output_shape = output_shapes[0]
        elif consistent_input:
            # Inconsistent outputs - choose the most common or largest
            input_shape = input_shapes[0]
            output_shape = self._choose_representative_output_shape(output_shapes)
            print(f"DEBUG: Inconsistent outputs, chose representative: {output_shape}")
        elif consistent_output:
            # Inconsistent inputs - use output shape as anchor
            output_shape = output_shapes[0]
            input_shape = self._choose_representative_input_shape(input_shapes)
        else:
            # Both inconsistent - use heuristics
            input_shape = self._choose_representative_input_shape(input_shapes)
            output_shape = self._choose_representative_output_shape(output_shapes)
            print(f"DEBUG: Both inconsistent, chose: input={input_shape}, output={output_shape}")
        
        signature.update({
            'input_size': input_shape,
            'output_size': output_shape,
            'size_change': self._analyze_size_change(input_shape, output_shape),
            'primary_pattern': self._detect_primary_pattern(train_pairs),
            'color_change': self._detect_color_changes(train_pairs),
            'spatial_relationships': self._analyze_spatial_relationships(train_pairs)
        })
        
        return signature
    
    def _choose_representative_output_shape(self, output_shapes: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Choose the most representative output shape from a list."""
        if not output_shapes:
            return (1, 1)
        
        # Strategy 1: Most common shape
        from collections import Counter
        shape_counts = Counter(output_shapes)
        most_common = shape_counts.most_common(1)[0][0]
        
        # Strategy 2: If all unique, choose largest by area
        if len(shape_counts) == len(output_shapes):
            return max(output_shapes, key=lambda s: s[0] * s[1])
        
        return most_common
    
    def _choose_representative_input_shape(self, input_shapes: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Choose the most representative input shape from a list."""
        if not input_shapes:
            return (1, 1)
        
        # Strategy: Choose most common, or largest if all unique
        from collections import Counter
        shape_counts = Counter(input_shapes)
        
        if len(shape_counts) == 1:
            return list(shape_counts.keys())[0]
        
        most_common = shape_counts.most_common(1)[0][0]
        
        # If all unique, choose largest
        if len(shape_counts) == len(input_shapes):
            return max(input_shapes, key=lambda s: s[0] * s[1])
        
        return most_common
    
    def _analyze_size_change(self, input_shape: Tuple[int, int], 
                           output_shape: Tuple[int, int]) -> Optional[str]:
        """Analyze type of size change."""
        if input_shape == output_shape:
            return None
        
        ih, iw = input_shape
        oh, ow = output_shape
        
        # Expansion
        if oh > ih or ow > iw:
            return f"{ih}x{iw}_to_{oh}x{ow}_expansion"
        
        # Reduction/extraction
        if oh < ih or ow < iw:
            return f"{ih}x{iw}_to_{oh}x{ow}_extraction"
        
        return f"{ih}x{iw}_to_{oh}x{ow}_transform"
    
    def _detect_primary_pattern(self, train_pairs: List[Tuple[Array, Array]]) -> str:
        """Detect the primary transformation pattern."""
        # Analyze first pair in detail
        inp, out = train_pairs[0]
        
        if inp.shape == out.shape:
            # Same size - could be recolor, rotation, or complex
            if self._is_likely_recolor(inp, out):
                return 'recolor'
            elif self._is_likely_rotation_reflection(inp, out):
                return 'rotation'
            else:
                return 'complex_same_size'
        else:
            # Size change
            if out.size < inp.size:
                return 'extraction'
            else:
                return 'expansion'
    
    def _is_likely_recolor(self, inp: Array, out: Array) -> bool:
        """Check if transformation is likely a recolor operation."""
        if inp.shape != out.shape:
            return False
        
        # Count position-wise differences
        position_changes = np.sum(inp != out)
        total_positions = inp.size
        
        # If many positions change but structure is preserved, likely recolor
        change_ratio = position_changes / total_positions
        
        # Check if it's a consistent color mapping
        unique_inp = set(inp.flatten())
        unique_out = set(out.flatten())
        
        # Similar number of colors suggests mapping
        color_ratio = len(unique_out) / max(1, len(unique_inp))
        
        return change_ratio > 0.3 and 0.5 <= color_ratio <= 2.0
    
    def _is_likely_rotation_reflection(self, inp: Array, out: Array) -> bool:
        """Check if transformation is likely rotation/reflection."""
        if inp.shape != out.shape:
            return False
        
        # Test standard transformations
        transformations = [
            np.rot90(inp, 1),
            np.rot90(inp, 2), 
            np.rot90(inp, 3),
            np.fliplr(inp),
            np.flipud(inp),
            inp.T
        ]
        
        for transformed in transformations:
            if transformed.shape == out.shape and np.array_equal(transformed, out):
                return True
        
        return False
    
    def _detect_color_changes(self, train_pairs: List[Tuple[Array, Array]]) -> bool:
        """Detect if colors change between input and output."""
        for inp, out in train_pairs:
            inp_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            
            # If new colors appear or old colors disappear
            if inp_colors != out_colors:
                return True
        
        return False
    
    def _analyze_spatial_relationships(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
        """Analyze spatial relationships in the transformation."""
        relationships = {
            'has_symmetry': False,
            'has_translation': False,
            'has_scaling': False,
            'preserves_topology': True
        }
        
        # Basic analysis on first pair
        if train_pairs:
            inp, out = train_pairs[0]
            
            # Check for translation patterns
            if inp.shape == out.shape:
                relationships['has_translation'] = self._detect_translation_pattern(inp, out)
            
            # Check for scaling
            if inp.shape != out.shape:
                relationships['has_scaling'] = True
        
        return relationships
    
    def _detect_translation_pattern(self, inp: Array, out: Array) -> bool:
        """Detect if there's a translation pattern."""
        # Simple check for shifted patterns
        if inp.shape != out.shape:
            return False
        
        # Look for correlation at different offsets
        best_correlation = 0
        h, w = inp.shape
        
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0:
                    continue
                
                # Create shifted version
                shifted = np.roll(np.roll(inp, dy, axis=0), dx, axis=1)
                correlation = np.sum(shifted == out) / out.size
                
                if correlation > best_correlation:
                    best_correlation = correlation
        
        return best_correlation > 0.8