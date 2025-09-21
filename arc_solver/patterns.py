"""
Placeholder pattern detection and template management for Phase 1-3 of reconstruction plan.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import hashlib

from .grid import Array


@dataclass
class PlaceholderTemplate:
    """Template for detected placeholder patterns."""
    signature: str
    placeholder_shape: Tuple[int, int]
    fill_fn: Callable[[Array], Array]
    uniform_region_color: int = 8
    border_patterns: Dict[str, List[int]] = None
    bounds: Tuple[int, int, int, int] = None
    symmetry_flags: Dict[str, bool] = None
    position_type: str = 'internal'
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.border_patterns is None:
            self.border_patterns = {}
        if self.symmetry_flags is None:
            self.symmetry_flags = {}
        if self.metadata is None:
            self.metadata = {}


class PlaceholderTemplateEngine:
    """Engine for detecting and applying placeholder templates."""
    
    def __init__(self):
        self.template_cache = {}
        
    def detect_templates(self, train_pairs: List[Tuple[Array, Array]]) -> List[PlaceholderTemplate]:
        """Detect placeholder templates from training pairs (Phase 1)."""
        templates = []
        
        for i, (inp, out) in enumerate(train_pairs):
            # Look for uniform placeholder regions
            placeholder_regions = self._find_placeholder_regions(inp)
            
            for region in placeholder_regions:
                if region['shape'] == out.shape:
                    # This might be a target placeholder
                    template = self._create_template_from_region(inp, out, region)
                    if template:
                        templates.append(template)
        
        # Deduplicate templates by signature
        unique_templates = {}
        for template in templates:
            unique_templates[template.signature] = template
        
        return list(unique_templates.values())
    
    def apply_template(self, input_grid: Array, template: PlaceholderTemplate) -> Optional[Array]:
        """Apply a template to fill placeholder regions (Phase 2)."""
        try:
            return template.fill_fn(input_grid)
        except Exception:
            return None
    
    def _find_placeholder_regions(self, inp: Array) -> List[Dict[str, Any]]:
        """Find solid rectangular regions that might be placeholders."""
        regions = []
        h, w = inp.shape
        
        # Look for each unique color
        for color in np.unique(inp):
            # Find connected regions of this color
            positions = np.where(inp == color)
            if len(positions[0]) < 4:  # Too small to be interesting
                continue
                
            min_r, max_r = positions[0].min(), positions[0].max()
            min_c, max_c = positions[1].min(), positions[1].max()
            
            # Check if this forms a solid rectangle
            region = inp[min_r:max_r+1, min_c:max_c+1]
            if np.all(region == color):
                # Solid rectangle found
                area = region.size
                total_area = h * w
                
                # Ignore if it's too large (background) or too small
                if 0.01 < area / total_area < 0.5:
                    regions.append({
                        'color': int(color),
                        'bounds': (min_r, min_c, max_r+1, max_c+1),
                        'area': area,
                        'shape': (max_r - min_r + 1, max_c - min_c + 1)
                    })
        
        return regions
    
    def _create_template_from_region(self, inp: Array, out: Array, region: Dict[str, Any]) -> Optional[PlaceholderTemplate]:
        """Create a template from a detected region."""
        color = region['color']
        bounds = region['bounds']
        shape = region['shape']
        
        # Extract border patterns
        border_patterns = self._extract_border_patterns(inp, bounds)
        
        # Check for symmetries
        symmetry_flags = self._detect_symmetries(inp, bounds)
        
        # Create signature
        signature_data = {
            'color': color,
            'shape': shape,
            'borders': border_patterns,
            'symmetries': symmetry_flags
        }
        signature = hashlib.md5(str(signature_data).encode()).hexdigest()[:8]
        
        # Create fill function
        def fill_fn(input_grid: Array) -> Array:
            return self._reconstruct_placeholder(input_grid, color, shape, border_patterns, symmetry_flags)
        
        return PlaceholderTemplate(
            signature=signature,
            placeholder_shape=shape,
            fill_fn=fill_fn,
            uniform_region_color=color,
            border_patterns=border_patterns,
            bounds=bounds,
            symmetry_flags=symmetry_flags,
            metadata={'example_output': out.copy()}
        )
    
    def _extract_border_patterns(self, inp: Array, bounds: Tuple[int, int, int, int]) -> Dict[str, List[int]]:
        """Extract patterns from borders around the placeholder."""
        top, left, bottom, right = bounds
        h, w = inp.shape
        patterns = {}
        
        # Extract immediate border pixels
        if top > 0:
            patterns['top'] = inp[top-1, left:right].tolist()
        if bottom < h:
            patterns['bottom'] = inp[bottom, left:right].tolist()
        if left > 0:
            patterns['left'] = inp[top:bottom, left-1].tolist()
        if right < w:
            patterns['right'] = inp[top:bottom, right].tolist()
        
        return patterns
    
    def _detect_symmetries(self, inp: Array, bounds: Tuple[int, int, int, int]) -> Dict[str, bool]:
        """Detect symmetry patterns around the placeholder."""
        top, left, bottom, right = bounds
        h, w = inp.shape
        
        symmetries = {}
        
        # Check for horizontal symmetry
        center_col = w // 2
        if left < center_col and right <= w:
            # Check if there's a corresponding region on the right
            mirror_left = center_col + (center_col - right)
            mirror_right = center_col + (center_col - left)
            
            if 0 <= mirror_left and mirror_right <= w:
                left_region = inp[top:bottom, left:right]
                right_region = inp[top:bottom, mirror_left:mirror_right]
                
                # Check if they're mirrors (considering one might be placeholder)
                if left_region.shape == right_region.shape:
                    symmetries['horizontal'] = True
        
        return symmetries
    
    def _reconstruct_placeholder(self, input_grid: Array, color: int, shape: Tuple[int, int],
                                border_patterns: Dict[str, List[int]], 
                                symmetry_flags: Dict[str, bool]) -> Array:
        """Reconstruct the placeholder using detected patterns."""
        result = input_grid.copy()
        
        # Find the placeholder in the current input
        current_regions = self._find_placeholder_regions(input_grid)
        target_region = None
        
        for region in current_regions:
            if region['color'] == color and region['shape'] == shape:
                target_region = region
                break
        
        if not target_region:
            return input_grid  # No matching placeholder found
        
        top, left, bottom, right = target_region['bounds']
        
        # Strategy 1: Use horizontal symmetry if detected
        if symmetry_flags.get('horizontal', False):
            h, w = input_grid.shape
            center_col = w // 2
            
            if left < center_col:
                # Extract from right side
                mirror_left = center_col + (center_col - right)
                mirror_right = center_col + (center_col - left)
                
                if 0 <= mirror_left and mirror_right <= w:
                    source_region = input_grid[top:bottom, mirror_left:mirror_right]
                    # Apply horizontal flip for true mirroring
                    mirrored_region = np.fliplr(source_region)
                    result[top:bottom, left:right] = mirrored_region
                    return result
        
        # Strategy 2: Use border patterns to fill
        if 'top' in border_patterns and len(border_patterns['top']) == (right - left):
            pattern = np.array(border_patterns['top'])
            for row in range(top, bottom):
                result[row, left:right] = pattern
            return result
        
        # Fallback: return unchanged
        return result