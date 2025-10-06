"""
Shape constraint enforcement and anchor calibration for ARC tasks.
Implements hard size constraints and anchor sweep for near-miss fixes.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from .grid import Array, to_array
from .dsl import apply_program

class ShapeGuard:
    """Enforces shape constraints and provides anchor calibration."""
    
    def __init__(self):
        self.debug_traces = []
        
    def enforce_shape_constraint(self, program: List[Tuple[str, Dict[str, Any]]], 
                                input_grid: Array, 
                                target_shape: Tuple[int, int]) -> Optional[Array]:
        """Execute program with hard shape constraint enforcement."""
        trace_id = f"shape_{id(program)}_{target_shape}"
        
        try:
            # Execute base program
            result = apply_program(input_grid, program)
            actual_shape = result.shape
            
            self.debug_traces.append({
                'trace_id': trace_id,
                'program': str(program)[:100],
                'expected_shape': target_shape,
                'actual_shape': actual_shape,
                'status': 'executed'
            })
            
            # Hard constraint: must match target shape exactly
            if actual_shape == target_shape:
                return result
            
            # Try shape adaptation strategies
            adapted = self._adapt_to_target_shape(result, target_shape, trace_id)
            if adapted is not None:
                return adapted
                
            # Failed to meet shape constraint
            self.debug_traces.append({
                'trace_id': trace_id,
                'status': 'shape_violation',
                'message': f'Cannot adapt {actual_shape} to {target_shape}'
            })
            return None
            
        except Exception as e:
            self.debug_traces.append({
                'trace_id': trace_id,
                'status': 'execution_error',
                'error': str(e)
            })
            return None
    
    def _adapt_to_target_shape(self, result: Array, target_shape: Tuple[int, int], 
                              trace_id: str) -> Optional[Array]:
        """Try to adapt result to target shape using various strategies."""
        h_target, w_target = target_shape
        h_actual, w_actual = result.shape
        
        # Strategy 1: Exact tiling (for block-based extraction)
        if h_target % h_actual == 0 and w_target % w_actual == 0:
            tile_h = h_target // h_actual
            tile_w = w_target // w_actual
            adapted = np.tile(result, (tile_h, tile_w))
            
            self.debug_traces.append({
                'trace_id': trace_id,
                'adaptation': 'tiling',
                'tile_factors': (tile_h, tile_w)
            })
            return adapted
        
        # Strategy 2: Center placement with background fill
        if h_actual <= h_target and w_actual <= w_target:
            # Find background color (most common in result)
            bg_color = int(np.bincount(result.flatten()).argmax())
            
            adapted = np.full(target_shape, bg_color, dtype=result.dtype)
            
            # Center the result
            start_h = (h_target - h_actual) // 2
            start_w = (w_target - w_actual) // 2
            adapted[start_h:start_h+h_actual, start_w:start_w+w_actual] = result
            
            self.debug_traces.append({
                'trace_id': trace_id,
                'adaptation': 'center_placement',
                'placement': (start_h, start_w),
                'bg_color': bg_color
            })
            return adapted
        
        # Strategy 3: Smart cropping (if result is larger)
        if h_actual >= h_target and w_actual >= w_target:
            # Find the most diverse region
            best_crop = None
            best_diversity = -1
            
            for start_h in range(h_actual - h_target + 1):
                for start_w in range(w_actual - w_target + 1):
                    crop = result[start_h:start_h+h_target, start_w:start_w+w_target]
                    diversity = len(np.unique(crop))
                    
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_crop = crop
            
            if best_crop is not None:
                self.debug_traces.append({
                    'trace_id': trace_id,
                    'adaptation': 'smart_crop',
                    'diversity': best_diversity
                })
                return best_crop
        
        return None
    
    def anchor_sweep(self, program: List[Tuple[str, Dict[str, Any]]], 
                    input_grid: Array, 
                    target_output: Array,
                    score_fn: Callable[[Array, Array], float]) -> Tuple[Array, float, Dict]:
        """Perform anchor sweep for spatial formulas with near-perfect scores."""
        best_result = None
        best_score = -1
        best_anchor = None
        
        # Try different anchor points and parity adjustments
        anchor_variants = [
            (0, 0),   # Original
            (1, 0),   # Row offset
            (0, 1),   # Column offset  
            (1, 1),   # Both offset
        ]
        
        for anchor_r, anchor_c in anchor_variants:
            try:
                # Create anchored version of the program
                anchored_program = self._apply_anchor(program, anchor_r, anchor_c)
                
                # Execute with shape constraint
                result = self.enforce_shape_constraint(anchored_program, input_grid, target_output.shape)
                
                if result is not None:
                    score = score_fn(result, target_output)
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_anchor = (anchor_r, anchor_c)
                        
            except Exception:
                continue
        
        # Calculate improvement safely
        improvement = 0
        if best_result is not None:
            try:
                original_result = self.enforce_shape_constraint(program, input_grid, target_output.shape)
                if original_result is not None:
                    original_score = score_fn(original_result, target_output)
                    improvement = best_score - original_score
                else:
                    improvement = best_score
            except Exception:
                improvement = best_score
        
        anchor_info = {
            'best_anchor': best_anchor,
            'anchors_tried': anchor_variants,
            'improvement': improvement
        }
        
        return best_result if best_result is not None else input_grid, best_score, anchor_info
    
    def _apply_anchor(self, program: List[Tuple[str, Dict[str, Any]]], 
                     anchor_r: int, anchor_c: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Apply anchor offset to spatial operations in program."""
        anchored_program = []
        
        for op_name, params in program:
            new_params = params.copy()
            
            # Apply anchor offsets to spatial operations
            if op_name == "translate":
                new_params["dy"] = params.get("dy", 0) + anchor_r
                new_params["dx"] = params.get("dx", 0) + anchor_c
            elif op_name == "crop":
                new_params["top"] = params.get("top", 0) + anchor_r
                new_params["left"] = params.get("left", 0) + anchor_c
            
            anchored_program.append((op_name, new_params))
        
        return anchored_program
    
    def get_debug_summary(self) -> str:
        """Get summary of shape constraint violations and adaptations."""
        violations = [t for t in self.debug_traces if t.get('status') == 'shape_violation']
        adaptations = [t for t in self.debug_traces if 'adaptation' in t]
        
        summary = f"Shape Guard Debug Summary:\n"
        summary += f"Total traces: {len(self.debug_traces)}\n"
        summary += f"Shape violations: {len(violations)}\n"
        summary += f"Successful adaptations: {len(adaptations)}\n"
        
        if adaptations:
            summary += "\nAdaptation strategies used:\n"
            for adaptation in adaptations:
                summary += f"  - {adaptation['adaptation']}: {adaptation.get('message', 'success')}\n"
        
        return summary


class SmartRecolorMapper:
    """Improved recolor mapping for better color transformation detection."""
    
    def __init__(self):
        pass
    
    def discover_recolor_mapping(self, input_pairs: List[Tuple[Array, Array]]) -> Optional[Dict[int, int]]:
        """Discover recolor mapping from input-output pairs using co-occurrence analysis."""
        if not input_pairs:
            return None
        
        # Build color co-occurrence matrix
        color_votes = {}  # input_color -> {output_color: count}
        
        for inp, out in input_pairs:
            if inp.shape != out.shape:
                continue
                
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    inp_color = int(inp[i, j])
                    out_color = int(out[i, j])
                    
                    if inp_color not in color_votes:
                        color_votes[inp_color] = {}
                    
                    color_votes[inp_color][out_color] = color_votes[inp_color].get(out_color, 0) + 1
        
        # Extract argmax mapping
        mapping = {}
        for inp_color, out_counts in color_votes.items():
            if out_counts:
                best_out_color = max(out_counts.items(), key=lambda x: x[1])[0]
                mapping[inp_color] = best_out_color
        
        # Validate mapping consistency
        if self._validate_mapping(mapping, input_pairs):
            return mapping
        
        return None
    
    def _validate_mapping(self, mapping: Dict[int, int], 
                         input_pairs: List[Tuple[Array, Array]]) -> bool:
        """Validate that mapping works consistently across pairs."""
        for inp, out in input_pairs:
            if inp.shape != out.shape:
                continue
                
            mapped = self._apply_mapping(inp, mapping)
            accuracy = np.sum(mapped == out) / out.size
            
            if accuracy < 0.8:  # Require 80% consistency
                return False
        
        return True
    
    def _apply_mapping(self, inp: Array, mapping: Dict[int, int]) -> Array:
        """Apply color mapping to input."""
        result = inp.copy()
        for inp_color, out_color in mapping.items():
            result[inp == inp_color] = out_color
        return result