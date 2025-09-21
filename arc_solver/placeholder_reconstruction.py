"""
Phase 2: Placeholder Reconstruction Candidate Generator

This module implements the candidate generator for the placeholder reconstruction
system. It builds DSL programs that can fill placeholder regions based on
detected template patterns.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from .grid import Array
from .dsl import apply_op, apply_program


@dataclass
class PlaceholderTemplate:
    """Template describing a placeholder pattern detected in Phase 1."""
    uniform_region_color: int
    border_patterns: Dict[str, List[int]]  # 'top', 'bottom', 'left', 'right'
    bounds: Tuple[int, int, int, int]  # (top, left, bottom, right)
    symmetry_flags: Dict[str, bool]  # 'horizontal', 'vertical', 'rotational'
    position_type: str  # 'touching_border', 'internal', 'corner'
    metadata: Dict[str, Any]


@dataclass
class ReconstructionCandidate:
    """A candidate DSL program for filling a placeholder."""
    dsl_program: List[Tuple[str, Dict[str, Any]]]
    confidence: float
    strategy_name: str
    description: str
    complexity: float


class PlaceholderCandidateGenerator:
    """Generates candidate DSL programs to fill placeholder regions."""
    
    def __init__(self):
        self.cache = {}
        
    def generate_candidates(self, 
                          input_grid: Array, 
                          template: PlaceholderTemplate) -> List[ReconstructionCandidate]:
        """Generate candidate DSL programs for filling the placeholder region."""
        candidates = []
        
        # Strategy 1: Stripe pattern mirroring
        stripe_candidates = self._generate_stripe_mirror_candidates(input_grid, template)
        candidates.extend(stripe_candidates)
        
        # Strategy 2: Border pattern repetition
        repeat_candidates = self._generate_border_repeat_candidates(input_grid, template)
        candidates.extend(repeat_candidates)
        
        # Strategy 3: Symmetric reconstruction
        symmetric_candidates = self._generate_symmetric_candidates(input_grid, template)
        candidates.extend(symmetric_candidates)
        
        # Strategy 4: Contextual pattern extraction
        context_candidates = self._generate_context_candidates(input_grid, template)
        candidates.extend(context_candidates)
        
        # Sort by confidence and complexity
        candidates.sort(key=lambda c: (-c.confidence, c.complexity))
        
        return candidates
    
    def _generate_stripe_mirror_candidates(self, 
                                         input_grid: Array, 
                                         template: PlaceholderTemplate) -> List[ReconstructionCandidate]:
        """Generate candidates by extracting and mirroring stripe patterns from borders."""
        candidates = []
        top, left, bottom, right = template.bounds
        
        # Extract border patterns
        border_patterns = self._extract_border_patterns(input_grid, template)
        
        for direction, pattern in border_patterns.items():
            if not pattern:
                continue
                
            # Create mirror/repeat programs
            mirror_program = self._create_stripe_mirror_program(
                pattern, direction, template.bounds, input_grid.shape
            )
            
            if mirror_program:
                candidates.append(ReconstructionCandidate(
                    dsl_program=mirror_program,
                    confidence=0.8,
                    strategy_name=f"stripe_mirror_{direction}",
                    description=f"Mirror {direction} stripe pattern into placeholder",
                    complexity=2.0
                ))
        
        return candidates
    
    def _generate_border_repeat_candidates(self, 
                                         input_grid: Array, 
                                         template: PlaceholderTemplate) -> List[ReconstructionCandidate]:
        """Generate candidates by repeating border patterns."""
        candidates = []
        top, left, bottom, right = template.bounds
        
        border_patterns = template.border_patterns
        
        # Try different repetition strategies
        for direction, pattern in border_patterns.items():
            if len(pattern) < 2:
                continue
                
            # Strategy: Tile the border pattern to fill the region
            repeat_program = self._create_border_tile_program(
                pattern, direction, template.bounds, input_grid.shape
            )
            
            if repeat_program:
                candidates.append(ReconstructionCandidate(
                    dsl_program=repeat_program,
                    confidence=0.7,
                    strategy_name=f"border_repeat_{direction}",
                    description=f"Repeat {direction} border pattern to fill placeholder",
                    complexity=1.5
                ))
        
        return candidates
    
    def _generate_symmetric_candidates(self, 
                                     input_grid: Array, 
                                     template: PlaceholderTemplate) -> List[ReconstructionCandidate]:
        """Generate candidates using symmetry operations."""
        candidates = []
        
        if template.symmetry_flags.get('horizontal', False):
            # Try horizontal mirroring
            mirror_program = self._create_symmetric_mirror_program(
                'horizontal', template.bounds, input_grid.shape
            )
            
            if mirror_program:
                candidates.append(ReconstructionCandidate(
                    dsl_program=mirror_program,
                    confidence=0.9,
                    strategy_name="symmetric_horizontal",
                    description="Fill using horizontal symmetry",
                    complexity=1.0
                ))
        
        if template.symmetry_flags.get('vertical', False):
            # Try vertical mirroring
            mirror_program = self._create_symmetric_mirror_program(
                'vertical', template.bounds, input_grid.shape
            )
            
            if mirror_program:
                candidates.append(ReconstructionCandidate(
                    dsl_program=mirror_program,
                    confidence=0.9,
                    strategy_name="symmetric_vertical", 
                    description="Fill using vertical symmetry",
                    complexity=1.0
                ))
        
        return candidates
    
    def _generate_context_candidates(self, 
                                   input_grid: Array, 
                                   template: PlaceholderTemplate) -> List[ReconstructionCandidate]:
        """Generate candidates by analyzing surrounding context."""
        candidates = []
        top, left, bottom, right = template.bounds
        h, w = input_grid.shape
        
        # Strategy: Look for similar-sized regions in the grid
        region_h, region_w = bottom - top, right - left
        
        for scan_top in range(0, h - region_h + 1):
            for scan_left in range(0, w - region_w + 1):
                # Skip the placeholder region itself
                if (scan_top, scan_left) == (top, left):
                    continue
                    
                # Extract candidate region
                candidate_region = input_grid[scan_top:scan_top+region_h, 
                                            scan_left:scan_left+region_w]
                
                # Check if this region has similar border characteristics
                if self._is_compatible_region(candidate_region, template):
                    # Create copy program
                    copy_program = self._create_region_copy_program(
                        (scan_top, scan_left, scan_top+region_h, scan_left+region_w),
                        template.bounds
                    )
                    
                    candidates.append(ReconstructionCandidate(
                        dsl_program=copy_program,
                        confidence=0.6,
                        strategy_name="context_copy",
                        description=f"Copy similar region from ({scan_top},{scan_left})",
                        complexity=1.2
                    ))
                    
                    # Also create a recolored version
                    recolor_mapping = derive_recolor_mapping(input_grid, template, candidate_region)
                    if recolor_mapping:
                        recolor_program = copy_program + [
                            ('recolor', {'mapping': recolor_mapping})
                        ]
                        
                        candidates.append(ReconstructionCandidate(
                            dsl_program=recolor_program,
                            confidence=0.7,
                            strategy_name="context_copy_recolor",
                            description=f"Copy and recolor region from ({scan_top},{scan_left})",
                            complexity=1.5
                        ))
        
        # Strategy: Advanced mirroring with different techniques
        for strategy in ['symmetric', 'pattern_completion', 'contextual_fill']:
            mirrored_result = apply_advanced_mirroring(input_grid, template, strategy)
            
            # Create program that produces this result
            mirror_program = [
                ('apply_advanced_mirroring', {
                    'template': template,
                    'strategy': strategy
                })
            ]
            
            candidates.append(ReconstructionCandidate(
                dsl_program=mirror_program,
                confidence=0.8 if strategy == 'symmetric' else 0.6,
                strategy_name=f"advanced_mirror_{strategy}",
                description=f"Apply {strategy} mirroring strategy",
                complexity=1.8
            ))
        
        return candidates
    
    def _extract_border_patterns(self, 
                               input_grid: Array, 
                               template: PlaceholderTemplate) -> Dict[str, List[int]]:
        """Extract stripe patterns from the borders of the placeholder region."""
        top, left, bottom, right = template.bounds
        h, w = input_grid.shape
        patterns = {}
        
        # Top border (if not at edge)
        if top > 0:
            patterns['top'] = input_grid[top-1, left:right].tolist()
        
        # Bottom border (if not at edge)
        if bottom < h:
            patterns['bottom'] = input_grid[bottom, left:right].tolist()
        
        # Left border (if not at edge)
        if left > 0:
            patterns['left'] = input_grid[top:bottom, left-1].tolist()
        
        # Right border (if not at edge)
        if right < w:
            patterns['right'] = input_grid[top:bottom, right].tolist()
        
        return patterns
    
    def _create_stripe_mirror_program(self, 
                                    pattern: List[int], 
                                    direction: str, 
                                    bounds: Tuple[int, int, int, int],
                                    grid_shape: Tuple[int, int]) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
        """Create a DSL program that mirrors a stripe pattern into the placeholder."""
        top, left, bottom, right = bounds
        h, w = grid_shape
        
        if direction == 'top':
            # Create a grid filled with the top pattern, then crop to placeholder
            if len(pattern) != (right - left):
                return None
                
            program = [
                # Create the pattern by repeating vertically
                ('create_pattern_fill', {
                    'pattern': pattern,
                    'target_bounds': bounds,
                    'direction': 'vertical_repeat'
                })
            ]
            
        elif direction == 'left':
            # Create pattern by repeating horizontally
            if len(pattern) != (bottom - top):
                return None
                
            program = [
                ('create_pattern_fill', {
                    'pattern': pattern,
                    'target_bounds': bounds,
                    'direction': 'horizontal_repeat'
                })
            ]
            
        else:
            # Similar for bottom/right borders
            return None
        
        return program
    
    def _create_border_tile_program(self, 
                                  pattern: List[int], 
                                  direction: str, 
                                  bounds: Tuple[int, int, int, int],
                                  grid_shape: Tuple[int, int]) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
        """Create a DSL program that tiles a border pattern."""
        program = [
            ('tile_pattern', {
                'pattern': pattern,
                'target_bounds': bounds,
                'direction': direction
            })
        ]
        return program
    
    def _create_symmetric_mirror_program(self, 
                                       symmetry_type: str, 
                                       bounds: Tuple[int, int, int, int],
                                       grid_shape: Tuple[int, int]) -> Optional[List[Tuple[str, Dict[str, Any]]]]:
        """Create a DSL program for symmetric mirroring."""
        top, left, bottom, right = bounds
        h, w = grid_shape
        
        if symmetry_type == 'horizontal':
            # Mirror from the opposite side horizontally
            center_col = w // 2
            
            if left < center_col:
                # Placeholder on left, mirror from right
                mirror_left = center_col + (center_col - right)
                mirror_right = center_col + (center_col - left)
            else:
                # Placeholder on right, mirror from left
                mirror_right = center_col - (left - center_col)
                mirror_left = center_col - (right - center_col)
            
            program = [
                ('crop', {
                    'top': top,
                    'left': mirror_left,
                    'height': bottom - top,
                    'width': right - left
                }),
                ('flip', {'axis': 1}),  # Horizontal flip
                ('paste_at', {
                    'target_top': top,
                    'target_left': left
                })
            ]
            
        elif symmetry_type == 'vertical':
            # Similar for vertical symmetry
            center_row = h // 2
            
            if top < center_row:
                mirror_top = center_row + (center_row - bottom)
                mirror_bottom = center_row + (center_row - top)
            else:
                mirror_bottom = center_row - (top - center_row)
                mirror_top = center_row - (bottom - center_row)
            
            program = [
                ('crop', {
                    'top': mirror_top,
                    'left': left,
                    'height': bottom - top,
                    'width': right - left
                }),
                ('flip', {'axis': 0}),  # Vertical flip
                ('paste_at', {
                    'target_top': top,
                    'target_left': left
                })
            ]
        else:
            return None
        
        return program
    
    def _create_region_copy_program(self, 
                                  source_bounds: Tuple[int, int, int, int],
                                  target_bounds: Tuple[int, int, int, int]) -> List[Tuple[str, Dict[str, Any]]]:
        """Create a DSL program that copies one region to another."""
        src_top, src_left, src_bottom, src_right = source_bounds
        tgt_top, tgt_left, tgt_bottom, tgt_right = target_bounds
        
        program = [
            ('crop', {
                'top': src_top,
                'left': src_left,
                'height': src_bottom - src_top,
                'width': src_right - src_left
            }),
            ('paste_at', {
                'target_top': tgt_top,
                'target_left': tgt_left
            })
        ]
        
        return program
    
    def _is_compatible_region(self, 
                            candidate_region: Array, 
                            template: PlaceholderTemplate) -> bool:
        """Check if a candidate region is compatible with the template."""
        # Check if the region doesn't contain the placeholder color
        if (candidate_region == template.uniform_region_color).any():
            return False
        
        # Check color diversity (should have some pattern, not just uniform)
        unique_colors = len(np.unique(candidate_region))
        if unique_colors < 2:
            return False
        
        # Check if it has reasonable color distribution
        region_size = candidate_region.size
        most_frequent_count = np.max(np.bincount(candidate_region.flatten()))
        if most_frequent_count / region_size > 0.8:  # Too uniform
            return False
        
        return True


# Helper functions for custom DSL operations
def create_pattern_fill(input_grid: Array, 
                       pattern: List[int], 
                       target_bounds: Tuple[int, int, int, int],
                       direction: str) -> Array:
    """Create a grid with pattern filled in the target bounds."""
    result = input_grid.copy()
    top, left, bottom, right = target_bounds
    
    if direction == 'vertical_repeat':
        # Repeat pattern vertically
        pattern_array = np.array(pattern)
        for row in range(top, bottom):
            result[row, left:right] = pattern_array
    elif direction == 'horizontal_repeat':
        # Repeat pattern horizontally
        pattern_array = np.array(pattern)
        for col in range(left, right):
            result[top:bottom, col] = pattern_array
    
    return result


def tile_pattern(input_grid: Array,
                pattern: List[int], 
                target_bounds: Tuple[int, int, int, int],
                direction: str) -> Array:
    """Tile a pattern within the target bounds."""
    result = input_grid.copy()
    top, left, bottom, right = target_bounds
    
    if direction in ['top', 'bottom']:
        # Tile horizontally
        pattern_len = len(pattern)
        width = right - left
        
        for row in range(top, bottom):
            for col in range(left, right):
                pattern_idx = (col - left) % pattern_len
                result[row, col] = pattern[pattern_idx]
                
    elif direction in ['left', 'right']:
        # Tile vertically
        pattern_len = len(pattern)
        height = bottom - top
        
        for col in range(left, right):
            for row in range(top, bottom):
                pattern_idx = (row - top) % pattern_len
                result[row, col] = pattern[pattern_idx]
    
    return result


def derive_recolor_mapping(input_grid: Array, 
                          template: PlaceholderTemplate,
                          candidate_region: Array) -> Dict[int, int]:
    """Derive recolor mapping from border/neighbor palette analysis."""
    placeholder_color = template.uniform_region_color
    top, left, bottom, right = template.bounds
    
    # Extract palette from surrounding border regions
    border_palette = set()
    h, w = input_grid.shape
    
    # Add colors from immediate neighbors
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        check_r, check_c = top + dr, left + dc
        if 0 <= check_r < h and 0 <= check_c < w:
            border_palette.add(input_grid[check_r, check_c])
    
    # Add colors from border patterns
    for pattern in template.border_patterns.values():
        border_palette.update(pattern)
    
    # Remove placeholder color from palette
    border_palette.discard(placeholder_color)
    
    # Build mapping from candidate region colors to border palette
    candidate_colors = set(np.unique(candidate_region))
    border_colors = list(border_palette)
    
    mapping = {}
    for i, candidate_color in enumerate(candidate_colors):
        if candidate_color != placeholder_color and border_colors:
            # Map to corresponding border color (cyclic if needed)
            target_color = border_colors[i % len(border_colors)]
            mapping[candidate_color] = target_color
    
    return mapping


def apply_advanced_mirroring(input_grid: Array,
                           template: PlaceholderTemplate,
                           strategy: str = 'symmetric') -> Array:
    """Apply advanced mirroring strategies with pattern completion."""
    result = input_grid.copy()
    top, left, bottom, right = template.bounds
    ph, pw = bottom - top, right - left
    h, w = input_grid.shape
    
    if strategy == 'symmetric':
        # Find the symmetric position and extract pattern
        center_col = w // 2
        
        if left < center_col:
            # Placeholder on left, extract from right
            mirror_left = center_col + (center_col - right)
            mirror_right = center_col + (center_col - left)
        else:
            # Placeholder on right, extract from left
            mirror_right = center_col - (left - center_col)
            mirror_left = center_col - (right - center_col)
        
        # Extract mirrored region if valid
        if 0 <= mirror_left and mirror_right <= w:
            source_region = input_grid[top:bottom, mirror_left:mirror_right]
            # Apply horizontal flip for true mirroring
            mirrored_region = np.fliplr(source_region)
            result[top:bottom, left:right] = mirrored_region
            
    elif strategy == 'pattern_completion':
        # Complete patterns based on detected repetitions
        border_patterns = template.border_patterns
        
        # Use top/bottom borders to fill vertically
        if 'top' in border_patterns and len(border_patterns['top']) == pw:
            pattern = np.array(border_patterns['top'])
            for row in range(top, bottom):
                result[row, left:right] = pattern
                
        # Use left/right borders to fill horizontally  
        elif 'left' in border_patterns and len(border_patterns['left']) == ph:
            pattern = np.array(border_patterns['left'])
            for col in range(left, right):
                result[top:bottom, col] = pattern
    
    elif strategy == 'contextual_fill':
        # Fill based on surrounding context analysis
        surrounding_colors = []
        
        # Sample colors from surrounding 3x3 regions
        for sample_r in range(max(0, top-1), min(h, bottom+2)):
            for sample_c in range(max(0, left-1), min(w, right+2)):
                if not (top <= sample_r < bottom and left <= sample_c < right):
                    surrounding_colors.append(input_grid[sample_r, sample_c])
        
        if surrounding_colors:
            # Use most common surrounding color as base
            unique_colors, counts = np.unique(surrounding_colors, return_counts=True)
            most_common = unique_colors[np.argmax(counts)]
            
            # Create simple pattern based on most common color
            fill_pattern = np.full((ph, pw), most_common, dtype=input_grid.dtype)
            result[top:bottom, left:right] = fill_pattern
    
    return result


def extract_stripe_patterns(input_grid: Array, 
                          template: PlaceholderTemplate) -> Dict[str, np.ndarray]:
    """Extract detailed stripe patterns from borders with analysis."""
    top, left, bottom, right = template.bounds
    h, w = input_grid.shape
    patterns = {}
    
    # Extract border stripes with extended context
    if top > 0:
        # Include multiple rows for pattern detection
        context_rows = min(3, top)
        top_context = input_grid[top-context_rows:top, left:right]
        patterns['top'] = top_context
    
    if bottom < h:
        context_rows = min(3, h - bottom)
        bottom_context = input_grid[bottom:bottom+context_rows, left:right]
        patterns['bottom'] = bottom_context
    
    if left > 0:
        context_cols = min(3, left)
        left_context = input_grid[top:bottom, left-context_cols:left]
        patterns['left'] = left_context
    
    if right < w:
        context_cols = min(3, w - right)
        right_context = input_grid[top:bottom, right:right+context_cols]
        patterns['right'] = right_context
    
    return patterns


def paste_at(source_grid: Array, 
            target_grid: Array,
            target_top: int, 
            target_left: int) -> Array:
    """Paste source grid into target at specified position."""
    result = target_grid.copy()
    src_h, src_w = source_grid.shape
    tgt_h, tgt_w = target_grid.shape
    
    # Calculate valid paste region
    paste_h = min(src_h, tgt_h - target_top)
    paste_w = min(src_w, tgt_w - target_left)
    
    if paste_h > 0 and paste_w > 0:
        result[target_top:target_top+paste_h, 
               target_left:target_left+paste_w] = source_grid[:paste_h, :paste_w]
    
    return result