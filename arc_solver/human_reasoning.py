"""
Human-Grade Spatial Reasoning for ARC Tasks.

This module implements the kind of hypothesis-driven, compositional reasoning 
that humans use to solve ARC tasks - spatial relationships, symmetries,
multi-source construction, and abstract pattern recognition.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import hashlib

from .grid import Array, to_array
from .object_reasoning import ObjectReasoner, ObjectHypothesisGenerator, ObjectTransformation
from .rft import RelationalFrameAnalyzer, RelationalFact
from .placeholders import PlaceholderTemplate, PlaceholderTemplateEngine


@dataclass
class SpatialHypothesis:
    """A hypothesis about how to construct the output."""
    name: str
    description: str
    confidence: float
    construction_rule: callable
    verification_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    complexity: float = 1.0


@dataclass  
class RegionRelation:
    """Describes spatial relationship between two regions."""
    region1: Tuple[int, int, int, int]  # (row1, col1, row2, col2)
    region2: Tuple[int, int, int, int]
    relation_type: str  # 'mirror_horizontal', 'mirror_vertical', 'rotation_90', etc.
    confidence: float


class HumanGradeReasoner:
    """Human-grade spatial reasoning engine for ARC tasks."""
    
    def __init__(self):
        self.hypotheses = []
        self.spatial_relations = []
        self.discovered_patterns = {}
        self.placeholder_engine = PlaceholderTemplateEngine()
        self.placeholder_templates: List[PlaceholderTemplate] = []
        self.object_reasoner = ObjectReasoner()
        self.object_hypothesis_generator = ObjectHypothesisGenerator()
        self.relational_facts: Optional[Dict[str, List[RelationalFact]]] = None
        
    def analyze_task(self, train_pairs: List[Tuple[Array, Array]]) -> List[SpatialHypothesis]:
        """Analyze task like a human would - form hypotheses about spatial relationships."""
        if not train_pairs:
            return []
            
        print("=== HUMAN-GRADE ANALYSIS ===")
        self.hypotheses = []
        
        # Step 1: Object-level analysis (NEW - RFT reasoning)
        object_transformations = self._generate_object_hypotheses(train_pairs)
        print(f"Object transformations found: {len(object_transformations)}")

        # Step 2: Identify key visual elements (like humans noticing 8-rectangles) 
        key_elements = self._identify_key_elements(train_pairs)
        print(f"Key elements identified: {len(key_elements)}")

        self.placeholder_templates = self.placeholder_engine.detect_templates(train_pairs)
        if self.placeholder_templates:
            print(f"Detected {len(self.placeholder_templates)} placeholder templates")

        # Step 2.5: Capture relational facts for downstream coordination
        analyzer = RelationalFrameAnalyzer()
        self.relational_facts = analyzer.analyze(train_pairs)

        # Step 3: Form hypotheses about spatial relationships
        self._generate_relational_hypotheses(train_pairs)
        self._generate_spatial_hypotheses(train_pairs, key_elements)
        
        # Step 4: Test hypotheses across all training examples
        self._verify_hypotheses(train_pairs)
        
        # Step 4: Rank simple-to-complex, then by confidence*verification
        self.hypotheses.sort(
            key=lambda h: (
                getattr(h, "complexity", 1.0),
                -(h.verification_score * h.confidence),
            )
        )
        
        print(f"Generated {len(self.hypotheses)} hypotheses")
        for i, h in enumerate(self.hypotheses[:3]):
            print(f"  {i+1}. {h.name}: {h.description} (confidence: {h.confidence:.2f}, verified: {h.verification_score:.2f})")
            
        return self.hypotheses
    
    def _generate_object_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> List[ObjectTransformation]:
        """Generate object-level transformation hypotheses using RFT reasoning."""
        # Generate object-level hypotheses
        object_transformations = self.object_hypothesis_generator.generate_hypotheses(train_pairs)
        
        # Convert object transformations to spatial hypotheses
        for i, obj_trans in enumerate(object_transformations):
            def object_transformation_rule(inp: Array) -> Array:
                return self.object_reasoner.apply_transformation(inp, obj_trans)
            
            # Test the transformation on training data
            verification_score = self.object_hypothesis_generator.test_hypothesis(obj_trans, train_pairs)
            
            # Create spatial hypothesis from object transformation
            hypothesis_name = f"object_{obj_trans.operation}_{i}"
            description = f"Object-level {obj_trans.operation}: {obj_trans.obj_selector} -> {obj_trans.parameters}"
            
            self.hypotheses.append(SpatialHypothesis(
                name=hypothesis_name,
                description=description,
                confidence=0.9,  # High confidence for object-level reasoning
                construction_rule=object_transformation_rule,
                verification_score=verification_score,
                complexity=1.0,
            ))
            
            print(f"  Object hypothesis: {hypothesis_name} (verified: {verification_score:.3f})")
        
        return object_transformations
    
    def _identify_key_elements(self, train_pairs: List[Tuple[Array, Array]]) -> List[Dict[str, Any]]:
        """Identify salient visual elements like humans do."""
        elements = []
        
        for i, (inp, out) in enumerate(train_pairs):
            # Look for placeholder regions (solid color rectangles that might be targets)
            placeholder_regions = self._find_placeholder_regions(inp)
            
            for region_info in placeholder_regions:
                # Check if this placeholder has same dimensions as output
                r1, c1, r2, c2 = region_info['bounds']
                region_shape = (r2 - r1, c2 - c1)
                
                if region_shape == out.shape:
                    elements.append({
                        'type': 'target_placeholder',
                        'example': i,
                        'bounds': region_info['bounds'],
                        'fill_color': region_info['color'],
                        'shape': region_shape,
                        'confidence': 0.9
                    })
                    print(f"  Found target placeholder in example {i}: {region_shape} filled with {region_info['color']}")
            
            # Look for symmetric regions
            symmetries = self._find_symmetries(inp)
            for sym in symmetries:
                elements.append({
                    'type': 'symmetry',
                    'example': i,
                    **sym
                })
                
            # Look for repeated patterns
            patterns = self._find_repeated_patterns(inp, out.shape)
            for pattern in patterns:
                elements.append({
                    'type': 'repeated_pattern',
                    'example': i,
                    **pattern
                })
        
        return elements
    
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
    
    def _find_symmetries(self, inp: Array) -> List[Dict[str, Any]]:
        """Find mirror symmetries and rotational patterns."""
        symmetries = []
        h, w = inp.shape
        
        # Test horizontal mirror symmetry
        if np.array_equal(inp, np.fliplr(inp)):
            symmetries.append({
                'type': 'horizontal_mirror',
                'confidence': 1.0
            })
        
        # Test vertical mirror symmetry  
        if np.array_equal(inp, np.flipud(inp)):
            symmetries.append({
                'type': 'vertical_mirror', 
                'confidence': 1.0
            })
            
        # Test partial symmetries (regions that mirror each other)
        center_col = w // 2
        center_row = h // 2
        
        # Test if left and right halves are related
        if center_col > 0:
            left_half = inp[:, :center_col]
            right_half = inp[:, center_col:center_col+left_half.shape[1]]
            
            if left_half.shape == right_half.shape:
                similarity = np.sum(left_half == np.fliplr(right_half)) / left_half.size
                if similarity > 0.7:
                    symmetries.append({
                        'type': 'partial_horizontal_mirror',
                        'confidence': similarity,
                        'left_region': (0, 0, h, center_col),
                        'right_region': (0, center_col, h, center_col + left_half.shape[1])
                    })
        
        return symmetries
    
    def _find_repeated_patterns(self, inp: Array, target_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Find patterns that repeat or could be sources for construction."""
        patterns = []
        h, w = inp.shape
        target_h, target_w = target_shape

        if target_h > h or target_w > w:
            return []

        # A dictionary to store regions by their canonical representation's hash
        # The value will be a list of (region, (r, c), transform_name)
        canonical_regions = defaultdict(list)

        # Pre-defined D4 transformations
        d4_transforms = {
            'identity': lambda g: g,
            'rot90': lambda g: np.rot90(g, 1),
            'rot180': lambda g: np.rot90(g, 2),
            'rot270': lambda g: np.rot90(g, 3),
            'fliph': lambda g: np.fliplr(g),
            'flipv': lambda g: np.flipud(g),
            'rot90_fliph': lambda g: np.rot90(np.fliplr(g), 1),
            'rot90_flipv': lambda g: np.rot90(np.flipud(g), 1),
        }

        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                region = inp[r:r+target_h, c:c+target_w]
                if len(np.unique(region)) < 2:
                    continue

                # Find the canonical representation of the region
                canonical_form = None
                canonical_key = None
                for transform_name, transform_func in d4_transforms.items():
                    try:
                        transformed_region = transform_func(region)
                        if transformed_region.shape != region.shape:
                            # This can happen for non-square regions with some rotations
                            continue
                        key = transformed_region.tobytes()
                        if canonical_key is None or key < canonical_key:
                            canonical_key = key
                            canonical_form = transformed_region
                    except: # some transformations might fail for non-square shapes
                        continue
                
                if canonical_form is not None:
                    canonical_hash = hashlib.sha256(canonical_form.tobytes()).digest()
                    canonical_regions[canonical_hash].append((region, (r, c)))

        # Now, find pairs of matching regions
        for canonical_hash, regions_info in canonical_regions.items():
            if len(regions_info) > 1:
                for i in range(len(regions_info)):
                    for j in range(i + 1, len(regions_info)):
                        region1, pos1 = regions_info[i]
                        region2, pos2 = regions_info[j]

                        # Find the transformation that maps region1 to region2
                        for trans_name, transform_func in d4_transforms.items():
                            try:
                                transformed_region = transform_func(region1)
                                if transformed_region.shape == region2.shape and np.array_equal(transformed_region, region2):
                                    patterns.append({
                                        'region1_pos': pos1,
                                        'region2_pos': pos2,
                                        'transformation': trans_name,
                                        'confidence': 1.0
                                    })
                                    # Found one transformation, no need to check others
                                    break
                            except:
                                continue
        return patterns
    
    def _generate_spatial_hypotheses(self, train_pairs: List[Tuple[Array, Array]], elements: List[Dict[str, Any]]):
        """Generate hypotheses about how to construct outputs."""
        
        if self.placeholder_templates:
            self._generate_template_hypotheses(train_pairs)

        # Hypothesis 1: Direct spatial relationship replacement
        target_placeholders = [e for e in elements if e['type'] == 'target_placeholder']
        if target_placeholders:
            self._generate_replacement_hypotheses(train_pairs, target_placeholders)
        
        # Hypothesis 2: Symmetry-based construction
        symmetries = [e for e in elements if e['type'] == 'symmetry']
        if symmetries:
            self._generate_symmetry_hypotheses(train_pairs, symmetries)
            
        # Hypothesis 3: Pattern-based construction  
        patterns = [e for e in elements if e['type'] == 'repeated_pattern']
        if patterns:
            self._generate_pattern_hypotheses(train_pairs, patterns)
            
        # Hypothesis 4: Multi-source composition (human creativity)
        self._generate_composition_hypotheses(train_pairs, elements)
    
    def _generate_template_hypotheses(self, train_pairs: List[Tuple[Array, Array]]) -> None:
        """Generate hypotheses from learned placeholder templates."""
        for idx, template in enumerate(self.placeholder_templates):
            def template_rule(inp: Array, *, _template=template) -> Array:
                result = self.placeholder_engine.apply_template(inp, _template)
                if result is None:
                    raise ValueError("placeholder template mismatch")
                return result

            fill_colors = np.unique(template.fill_pattern).tolist()
            target_shape = train_pairs[0][1].shape if train_pairs else template.fill_pattern.shape
            metadata = {
                'type': 'placeholder_template',
                'placeholder_color': template.signature.placeholder_color,
                'placeholder_shape': tuple(int(x) for x in template.signature.shape),
                'target_shape': tuple(int(x) for x in target_shape),
                'fill_colors': fill_colors,
            }
            self.hypotheses.append(
                SpatialHypothesis(
                    name=f"placeholder_template_{idx}",
                    description="Apply learned placeholder fill pattern",
                    confidence=0.95,
                    construction_rule=template_rule,
                    metadata=metadata,
                    complexity=0.9,
                )
            )
            print(
                f"  Placeholder template hypothesis {idx}: "
                f"shape={template.signature.shape}, color={template.signature.placeholder_color}"
            )

    def _generate_replacement_hypotheses(self, train_pairs: List[Tuple[Array, Array]], placeholders: List[Dict[str, Any]]):
        """Generate hypotheses for replacing placeholder regions."""
        
        # Group placeholders by target shape from training examples
        target_shapes = set()
        for inp, out in train_pairs:
            target_shapes.add(out.shape)
        
        print(f"DEBUG: Target shapes from training: {target_shapes}")
        
        for placeholder in placeholders:
            fill_color = placeholder['fill_color']
            target_shape = placeholder['shape']
            
            # Only generate hypotheses for placeholders that match expected output shapes
            if target_shape in target_shapes:
                print(f"DEBUG: Generating hypotheses for {target_shape} placeholder")
                
                # Hypothesis: Replace with content from symmetric position
                def symmetric_replacement(inp: Array) -> Array:
                    return self._extract_symmetric_content(inp, placeholder, 'horizontal')
                
                self.hypotheses.append(SpatialHypothesis(
                    name=f"symmetric_replacement_{fill_color}_{target_shape[0]}x{target_shape[1]}",
                    description=f"Replace {fill_color}-filled {target_shape} region with horizontally symmetric content",
                    confidence=0.8,
                    construction_rule=symmetric_replacement,
                    complexity=1.2,
                ))
                
                # Hypothesis: Replace with content from mirrored position
                def mirrored_replacement(inp: Array) -> Array:
                    return self._extract_mirrored_content(inp, placeholder)
                
                self.hypotheses.append(SpatialHypothesis(
                    name=f"mirrored_replacement_{fill_color}_{target_shape[0]}x{target_shape[1]}",
                    description=f"Replace {fill_color}-filled {target_shape} region with mirrored content from opposite side",
                    confidence=0.7,
                    construction_rule=mirrored_replacement,
                    complexity=1.4,
                ))
                
                # Hypothesis: Replace with content from adjacent region
                def adjacent_replacement(inp: Array) -> Array:
                    return self._extract_adjacent_content(inp, placeholder)
                    
                self.hypotheses.append(SpatialHypothesis(
                    name=f"adjacent_replacement_{fill_color}_{target_shape[0]}x{target_shape[1]}",
                    description=f"Replace {fill_color}-filled {target_shape} region with content from adjacent area",
                    confidence=0.6,
                    construction_rule=adjacent_replacement,
                    complexity=1.6,
                ))

                # Hypothesis: Use RFT-derived transformation (if available)
                fact = self._select_best_transformation_fact(target_shape)
                if fact is not None:
                    def targeted_extraction(inp: Array, *, _placeholder=placeholder, _fact=fact) -> Array:
                        return self._extract_using_transformation(inp, _placeholder, _fact)

                    translation = fact.metadata.get('translation', (0.0, 0.0))
                    self.hypotheses.append(
                        SpatialHypothesis(
                            name=f"transformation_extraction_{fact.object[0]}_{target_shape[0]}x{target_shape[1]}",
                            description="Apply RFT-guided translation extraction",
                            confidence=0.85,
                            construction_rule=targeted_extraction,
                            metadata={
                                'type': 'transformation_extraction',
                                'target_shape': target_shape,
                                'translation': translation,
                                'match_score': fact.metadata.get('match_score', 0.0),
                                'size_similarity': fact.metadata.get('size_similarity', 0.0),
                                'distance': fact.metadata.get('distance', 0.0),
                                'subject_signature': fact.subject,
                                'object_signature': fact.object,
                            },
                            complexity=1.8,
                        )
                    )
    
    def _generate_symmetry_hypotheses(self, train_pairs: List[Tuple[Array, Array]], symmetries: List[Dict[str, Any]]):
        """Generate hypotheses based on symmetry patterns."""
        
        for symmetry in symmetries:
            if symmetry['type'] == 'partial_horizontal_mirror':
                def symmetric_extraction(inp: Array) -> Array:
                    # Extract the output from the symmetric relationship
                    left_region = symmetry['left_region']  
                    right_region = symmetry['right_region']
                    
                    # Find which region contains the placeholder and extract from the other
                    return self._extract_from_symmetric_region(inp, left_region, right_region)
                
                self.hypotheses.append(SpatialHypothesis(
                    name="symmetric_extraction",
                    description="Extract output from symmetric region relationship",
                    confidence=symmetry['confidence'],
                    construction_rule=symmetric_extraction,
                    complexity=1.8,
                ))
    
    def _generate_pattern_hypotheses(self, train_pairs: List[Tuple[Array, Array]], patterns: List[Dict[str, Any]]):
        """Generate hypotheses based on repeated patterns."""
        
        for pattern in patterns:
            def pattern_based_construction(inp: Array) -> Array:
                # Use the pattern relationship to construct output
                pos1 = pattern['region1_pos']
                pos2 = pattern['region2_pos']
                transformation = pattern['transformation']
                
                return self._construct_from_pattern(inp, pos1, pos2, transformation)
            
            self.hypotheses.append(SpatialHypothesis(
                name=f"pattern_construction_{pattern['transformation']}",
                description=f"Construct output using {pattern['transformation']} relationship",
                confidence=pattern['confidence'],
                construction_rule=pattern_based_construction,
                complexity=2.0,
            ))
    
    def _generate_composition_hypotheses(self, train_pairs: List[Tuple[Array, Array]], elements: List[Dict[str, Any]]):
        """Generate creative composition hypotheses (most human-like)."""
        
        # Hypothesis: Output is composed by combining multiple regions with transformations
        def multi_region_composition(inp: Array) -> Array:
            return self._compose_from_multiple_regions(inp, train_pairs[0][1].shape)
        
        self.hypotheses.append(SpatialHypothesis(
            name="multi_region_composition",
            description="Compose output by intelligently combining multiple regions",
            confidence=0.5,  # Lower initial confidence, but can be very powerful
            construction_rule=multi_region_composition,
            complexity=2.5,
        ))
        
        # Hypothesis: Output follows a spatial formula (row/column relationships)  
        def spatial_formula_construction(inp: Array) -> Array:
            return self._construct_by_spatial_formula(inp, train_pairs[0][1].shape)
            
        self.hypotheses.append(SpatialHypothesis(
            name="spatial_formula",
            description="Construct output using spatial position formulas",
            confidence=0.4,
            construction_rule=spatial_formula_construction,
            complexity=3.0,
        ))
    
    def _extract_symmetric_content(self, inp: Array, placeholder: Dict[str, Any], symmetry_type: str) -> Array:
        """Extract content from symmetric position."""
        fill_color = placeholder['fill_color']
        h, w = inp.shape
        target_shape = placeholder['shape']
        
        # ENHANCED: Better search for actual content instead of just mirroring
        # Find the actual placeholder region in the current input
        current_placeholders = self._find_placeholder_regions(inp)
        
        # Find the placeholder with the same fill color and similar size
        target_placeholder = None
        
        for ph in current_placeholders:
            if (ph['color'] == fill_color and 
                abs(ph['shape'][0] - target_shape[0]) <= 1 and
                abs(ph['shape'][1] - target_shape[1]) <= 1):
                target_placeholder = ph
                break
        
        if not target_placeholder:
            # ENHANCED: Search entire grid for best matching region
            return self._find_best_extraction_region(inp, target_shape)
            
        r1, c1, r2, c2 = target_placeholder['bounds']
        
        # Try multiple extraction strategies
        candidates = []
        
        # Strategy 1: Horizontal mirror
        if symmetry_type == 'horizontal':
            center_col = w // 2
            mirror_c1 = center_col - (c2 - center_col)
            mirror_c2 = center_col - (c1 - center_col)
            
            if mirror_c1 >= 0 and mirror_c2 <= w:
                candidates.append(inp[r1:r2, mirror_c1:mirror_c2].copy())
        
        # Strategy 2: Vertical mirror
        elif symmetry_type == 'vertical':
            center_row = h // 2  
            mirror_r1 = center_row - (r2 - center_row)
            mirror_r2 = center_row - (r1 - center_row)
            
            if mirror_r1 >= 0 and mirror_r2 <= h:
                candidates.append(inp[mirror_r1:mirror_r2, c1:c2].copy())
        
        # Strategy 3: Search for exact matches in the grid
        exact_match = self._find_exact_pattern_match(inp, target_shape, (r1, c1, r2, c2))
        if exact_match is not None:
            candidates.append(exact_match)
        
        # Strategy 4: Find most diverse region 
        diverse_region = self._find_best_extraction_region(inp, target_shape)
        candidates.append(diverse_region)
        
        # Return the best candidate (most diverse)
        if candidates:
            best_candidate = max(candidates, key=lambda x: len(np.unique(x)) if x.size > 0 else 0)
            return best_candidate
        
        # Ultimate fallback
        return np.zeros(target_shape, dtype=inp.dtype)
    
    def _find_exact_pattern_match(self, inp: Array, target_shape: Tuple[int, int], 
                                 exclude_region: Tuple[int, int, int, int]) -> Optional[Array]:
        """Find exact pattern matches in the grid, excluding the specified region."""
        target_h, target_w = target_shape
        h, w = inp.shape
        ex_r1, ex_c1, ex_r2, ex_c2 = exclude_region
        
        best_region = None
        best_diversity = 0
        
        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                # Skip if this overlaps with excluded region
                if not (r >= ex_r2 or r + target_h <= ex_r1 or c >= ex_c2 or c + target_w <= ex_c1):
                    continue
                
                region = inp[r:r+target_h, c:c+target_w]
                diversity = len(np.unique(region))
                
                # Prefer regions with multiple colors and interesting patterns
                if diversity > best_diversity and diversity >= 3:
                    best_diversity = diversity
                    best_region = region.copy()
        
        return best_region
    
    def _find_best_extraction_region(self, inp: Array, target_shape: Tuple[int, int]) -> Array:
        """Find the most diverse and interesting region of the target shape."""
        target_h, target_w = target_shape
        h, w = inp.shape
        
        if target_h > h or target_w > w:
            return np.zeros(target_shape, dtype=inp.dtype)
        
        best_region = None
        best_score = -1
        
        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                region = inp[r:r+target_h, c:c+target_w]
                
                # Score based on diversity and pattern complexity
                diversity = len(np.unique(region))
                
                # Avoid regions that are mostly one color
                most_common_count = np.max(np.bincount(region.flatten()))
                complexity = 1.0 - (most_common_count / region.size)
                
                score = diversity * complexity
                
                if score > best_score:
                    best_score = score
                    best_region = region.copy()
        
        return best_region if best_region is not None else np.zeros(target_shape, dtype=inp.dtype)

    def _select_best_transformation_fact(self, target_shape: Tuple[int, int]) -> Optional[RelationalFact]:
        if not self.relational_facts:
            return None

        best_fact = None
        best_score = -float('inf')

        for fact in self.relational_facts.get('transformation', []):
            _, obj_h, obj_w = fact.object
            if (obj_h, obj_w) != target_shape:
                continue

            match_score = fact.metadata.get('match_score', 0.0)
            size_similarity = fact.metadata.get('size_similarity', 0.0)
            distance = fact.metadata.get('distance', 1.0)
            score = match_score + size_similarity - distance

            if score > best_score:
                best_score = score
                best_fact = fact

        return best_fact

    def _match_placeholder_in_input(
        self,
        inp: Array,
        fill_color: int,
        target_shape: Tuple[int, int],
        tolerance: int = 1,
    ) -> Optional[Dict[str, Any]]:
        candidates = []
        for current in self._find_placeholder_regions(inp):
            if current['color'] != fill_color:
                continue
            dh = abs(current['shape'][0] - target_shape[0])
            dw = abs(current['shape'][1] - target_shape[1])
            if dh <= tolerance and dw <= tolerance:
                candidates.append((dh + dw, current))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _extract_using_transformation(
        self,
        inp: Array,
        placeholder: Dict[str, Any],
        fact: RelationalFact,
    ) -> Array:
        target_shape = placeholder['shape']
        fill_color = placeholder['fill_color']

        matched_placeholder = self._match_placeholder_in_input(inp, fill_color, target_shape)
        if matched_placeholder is None:
            return self._find_best_extraction_region(inp, target_shape)

        r1, c1, r2, c2 = matched_placeholder['bounds']
        translation = fact.metadata.get('translation', (0.0, 0.0))
        col_offset = int(round(float(translation[0])))
        row_offset = int(round(float(translation[1])))

        if row_offset == 0 and col_offset == 0:
            return self._find_best_extraction_region(inp, target_shape)

        source_r1 = r1 - row_offset
        source_r2 = r2 - row_offset
        source_c1 = c1 - col_offset
        source_c2 = c2 - col_offset

        h, w = inp.shape
        if source_r1 < 0 or source_c1 < 0 or source_r2 > h or source_c2 > w:
            return self._find_best_extraction_region(inp, target_shape)

        candidate = inp[source_r1:source_r2, source_c1:source_c2]
        if candidate.shape != target_shape:
            return self._find_best_extraction_region(inp, target_shape)

        # Reject candidate if it is predominantly placeholder color
        if np.all(candidate == fill_color):
            return self._find_best_extraction_region(inp, target_shape)

        return candidate.copy()

    def _generate_relational_hypotheses(self, train_pairs: List[Tuple[Array, Array]]):
        if not self.relational_facts:
            return

        if not train_pairs:
            return

        target_shape = train_pairs[0][1].shape
        considered: Set[Tuple[int, int]] = set()

        relational_sources = []
        relational_sources.extend(self.relational_facts.get('transformation', []))
        relational_sources.extend(self.relational_facts.get('composite', []))
        relational_sources.extend(self.relational_facts.get('inverse', []))

        max_relational_hypotheses = 12

        for fact in relational_sources:
            if len(considered) >= max_relational_hypotheses:
                break

            translation = self._fact_translation_vector(fact)
            if translation is None:
                continue

            if translation == (0, 0):
                continue

            if translation in considered:
                continue

            if abs(translation[0]) > target_shape[0] or abs(translation[1]) > target_shape[1]:
                continue

            match_score = fact.metadata.get('match_score', fact.confidence if fact.metadata else fact.confidence)
            if match_score < 1.0:
                continue

            considered.add(translation)

            def relational_translation(inp: Array, *, _vector=translation, _target_shape=target_shape) -> Array:
                return self._translate_grid(inp, _vector, _target_shape)

            self.hypotheses.append(
                SpatialHypothesis(
                    name=f"relation_translation_{translation[0]}_{translation[1]}",
                    description="Translate grid using derived relational vector",
                    confidence=min(0.75, fact.confidence + 0.2),
                    construction_rule=relational_translation,
                    verification_score=min(1.0, match_score),
                    metadata={
                        'type': 'relation_translation',
                        'translation': translation,
                        'source_relation': fact.relation,
                    },
                    complexity=1.9,
                )
            )

    def _fact_translation_vector(self, fact: RelationalFact) -> Optional[Tuple[int, int]]:
        translation = fact.metadata.get('translation') if fact.metadata else None
        if translation and isinstance(translation, tuple) and len(translation) == 2:
            col_offset, row_offset = translation
            return int(round(float(row_offset))), int(round(float(col_offset)))

        src_center = fact.metadata.get('src_center') if fact.metadata else None
        tgt_center = fact.metadata.get('tgt_center') if fact.metadata else None
        if src_center and tgt_center:
            row_offset = int(round(float(tgt_center[0] - src_center[0])))
            col_offset = int(round(float(tgt_center[1] - src_center[1])))
            if row_offset != 0 or col_offset != 0:
                return row_offset, col_offset

        return None

    def _translate_grid(self, grid: Array, vector: Tuple[int, int], target_shape: Tuple[int, int]) -> Array:
        dr, dc = vector
        src_h, src_w = grid.shape
        tgt_h, tgt_w = target_shape
        background = self._dominant_color(grid)
        result = np.full(target_shape, background, dtype=grid.dtype)

        for r in range(src_h):
            for c in range(src_w):
                value = grid[r, c]
                if value == background:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < tgt_h and 0 <= nc < tgt_w:
                    result[nr, nc] = value

        return result

    @staticmethod
    def _dominant_color(grid: Array) -> int:
        values, counts = np.unique(grid, return_counts=True)
        if len(values) == 0:
            return 0
        idx = int(np.argmax(counts))
        return int(values[idx])
    
    def _extract_mirrored_content(self, inp: Array, placeholder: Dict[str, Any]) -> Array:
        """Extract content from mirrored position with transformations."""
        symmetric_content = self._extract_symmetric_content(inp, placeholder, 'horizontal')
        
        # Try different transformations of the symmetric content
        candidates = [
            symmetric_content,
            np.fliplr(symmetric_content),
            np.flipud(symmetric_content),
            np.rot90(symmetric_content),
            np.rot90(symmetric_content, 2),
            np.rot90(symmetric_content, 3)
        ]
        
        # Return the first non-placeholder candidate (not all same color)
        fill_color = placeholder['fill_color']
        for candidate in candidates:
            if not np.all(candidate == fill_color) and len(np.unique(candidate)) > 1:
                return candidate
        
        return symmetric_content
    
    def _extract_adjacent_content(self, inp: Array, placeholder: Dict[str, Any]) -> Array:
        """Extract content from adjacent regions."""
        r1, c1, r2, c2 = placeholder['bounds']
        target_h, target_w = r2 - r1, c2 - c1
        h, w = inp.shape
        
        # Try adjacent regions in all directions
        candidates = []
        
        # Left
        if c1 - target_w >= 0:
            candidates.append(inp[r1:r2, c1-target_w:c1])
            
        # Right  
        if c2 + target_w <= w:
            candidates.append(inp[r1:r2, c2:c2+target_w])
            
        # Above
        if r1 - target_h >= 0:
            candidates.append(inp[r1-target_h:r1, c1:c2])
            
        # Below
        if r2 + target_h <= h:
            candidates.append(inp[r2:r2+target_h, c1:c2])
        
        # Return the most diverse candidate (highest color variety)
        fill_color = placeholder['fill_color']
        best_candidate = None
        best_diversity = 0
        
        for candidate in candidates:
            if candidate.shape == (target_h, target_w):
                diversity = len(np.unique(candidate))
                # Prefer candidates that aren't just the placeholder color
                if not np.all(candidate == fill_color) and diversity > best_diversity:
                    best_diversity = diversity
                    best_candidate = candidate
        
        if best_candidate is not None:
            return best_candidate
            
        # Fallback
        return inp[r1:r2, c1:c2].copy()
    
    def _compose_from_multiple_regions(self, inp: Array, target_shape: Tuple[int, int]) -> Array:
        """Compose output by intelligently combining multiple regions (most human-like)."""
        target_h, target_w = target_shape
        output = np.zeros((target_h, target_w), dtype=inp.dtype)
        
        # CRITICAL FIX: Handle pattern completion for same-size tasks
        h, w = inp.shape
        if target_shape == (h, w):
            return self._complete_repeating_patterns(inp)
        
        # Strategy: Look for interesting patterns in the input and compose them
        # Find regions with high color diversity (interesting content)
        interesting_regions = []
        for r in range(h - target_h + 1):
            for c in range(w - target_w + 1):
                region = inp[r:r+target_h, c:c+target_w]
                diversity = len(np.unique(region))
                if diversity >= 3:  # At least 3 different colors
                    interesting_regions.append((region, diversity, r, c))
        
        # Sort by diversity (most interesting first)
        interesting_regions.sort(key=lambda x: x[1], reverse=True)
        
        if interesting_regions:
            # Use the most interesting region as base
            return interesting_regions[0][0].copy()
        
        # Fallback: extract from center  
        center_r, center_c = h // 2 - target_h // 2, w // 2 - target_w // 2
        if center_r >= 0 and center_c >= 0 and center_r + target_h <= h and center_c + target_w <= w:
            return inp[center_r:center_r+target_h, center_c:center_c+target_w].copy()
        
        return output
    
    def _complete_repeating_patterns(self, inp: Array) -> Array:
        """Complete repeating patterns in same-size grids with anchor variants."""
        # Try multiple anchor offsets for pattern completion
        candidates = []
        
        for anchor_offset in [0, 1, -1]:  # Try different starting positions
            result = self._complete_repeating_patterns_with_anchor(inp, anchor_offset)
            candidates.append(result)
        
        # Return the candidate with most pattern completion
        best_candidate = inp.copy()
        best_completions = 0
        
        for candidate in candidates:
            # Count how much was completed (non-zero differences from input)
            completions = np.sum(candidate != inp)
            if completions > best_completions:
                best_completions = completions
                best_candidate = candidate
        
        return best_candidate
    
    def _complete_repeating_patterns_with_anchor(self, inp: Array, anchor_offset: int) -> Array:
        """Complete repeating patterns with specific anchor offset."""
        result = inp.copy()
        h, w = inp.shape
        
        # Process each row looking for incomplete patterns
        for r in range(h):
            row = result[r, :]
            
            # Look for sections that have repeating patterns with gaps
            # Common pattern: border + repeating middle + border
            if w >= 9:  # Need sufficient length
                # Try to detect pattern in middle section with anchor offset
                start_idx = max(0, 2 + anchor_offset)
                end_idx = min(w, w - 2 + anchor_offset)
                
                if end_idx > start_idx + 4:  # Need enough space for pattern
                    middle_section = row[start_idx:end_idx]
                    
                    # Try different pattern lengths
                    for pattern_len in [2, 3, 4]:
                        if self._try_complete_pattern(row, start_idx, end_idx, pattern_len):
                            result[r, :] = row
                            break
        
        return result
    
    def _try_complete_pattern(self, row: np.ndarray, start_idx: int, end_idx: int, pattern_len: int) -> bool:
        """Try to complete a pattern of given length in the middle section."""
        middle_section = row[start_idx:end_idx]
        section_len = len(middle_section)
        
        if section_len < pattern_len * 2:
            return False
        
        # Extract the first pattern_len elements as candidate pattern
        candidate_pattern = middle_section[:pattern_len]
        
        # Check if this pattern, when repeated, matches most of the section
        expected_repeats = section_len // pattern_len
        reconstructed = np.tile(candidate_pattern, expected_repeats)
        
        # Allow for partial match at the end
        if len(reconstructed) > section_len:
            reconstructed = reconstructed[:section_len]
        elif len(reconstructed) < section_len:
            # Extend with pattern
            remaining = section_len - len(reconstructed)
            reconstructed = np.concatenate([reconstructed, candidate_pattern[:remaining]])
        
        # Count matches
        matches = np.sum(middle_section == reconstructed)
        match_ratio = matches / section_len
        
        # If high match ratio, apply the pattern
        if match_ratio >= 0.7:  # 70% match threshold
            # Apply the complete pattern
            for i in range(section_len):
                pattern_idx = i % pattern_len
                row[start_idx + i] = candidate_pattern[pattern_idx]
            return True
        
        return False
    
    def _construct_by_spatial_formula(self, inp: Array, target_shape: Tuple[int, int]) -> Array:
        """Construct output using spatial formulas (human mathematical reasoning)."""
        # Try multiple anchor variants to fix off-by-one errors
        candidates = []
        
        for anchor_r in range(2):  # Try 0 and 1 offset
            for anchor_c in range(2):  # Try 0 and 1 offset
                result = self._construct_by_spatial_formula_with_anchor(inp, target_shape, anchor_r, anchor_c)
                candidates.append(result)
        
        # Return the first non-zero candidate, or the last one as fallback
        for candidate in candidates:
            if np.sum(candidate) > 0:  # Has some content
                return candidate
        
        return candidates[-1] if candidates else np.zeros(target_shape, dtype=inp.dtype)
    
    def _construct_by_spatial_formula_with_anchor(self, inp: Array, target_shape: Tuple[int, int], 
                                                 anchor_r: int, anchor_c: int) -> Array:
        """Construct output using spatial formulas with anchor offset."""
        target_h, target_w = target_shape
        output = np.zeros((target_h, target_w), dtype=inp.dtype)
        h, w = inp.shape
        
        # Formula-based construction: each output position calculated from input positions
        for out_r in range(target_h):
            for out_c in range(target_w):
                # Try different spatial formulas with anchor offset
                formulas = [
                    # Direct mapping with anchor offset
                    (out_r + anchor_r, out_c + anchor_c),
                    (h - 1 - out_r - anchor_r, w - 1 - out_c - anchor_c),  # Opposite corner
                    (out_r + anchor_r, w - 1 - out_c - anchor_c),  # Horizontal mirror
                    (h - 1 - out_r - anchor_r, out_c + anchor_c),  # Vertical mirror
                    
                    # Proportional mapping with anchor
                    (int((out_r + anchor_r) * h / target_h), int((out_c + anchor_c) * w / target_w)),
                    
                    # Center-relative mapping with anchor
                    (h // 2 + out_r - target_h // 2 + anchor_r, w // 2 + out_c - target_w // 2 + anchor_c)
                ]
                
                for inp_r, inp_c in formulas:
                    if 0 <= inp_r < h and 0 <= inp_c < w:
                        output[out_r, out_c] = inp[inp_r, inp_c]
                        break
        
        return output
    
    def _verify_hypotheses(self, train_pairs: List[Tuple[Array, Array]]):
        """Test each hypothesis across all training examples."""
        for hypothesis in self.hypotheses:
            total_score = 0.0
            valid_tests = 0
            
            for inp, expected_out in train_pairs:
                try:
                    predicted_out = hypothesis.construction_rule(inp)
                    if predicted_out.shape == expected_out.shape:
                        # Calculate similarity score
                        matches = np.sum(predicted_out == expected_out)
                        total_matches = expected_out.size
                        score = matches / total_matches
                        total_score += score
                        valid_tests += 1
                except Exception as e:
                    # Hypothesis failed on this example
                    continue
            
            if valid_tests > 0:
                hypothesis.verification_score = total_score / valid_tests
            else:
                hypothesis.verification_score = 0.0
    
    def solve_task(self, train_pairs: List[Tuple[Array, Array]], test_input: Array) -> Array:
        """Solve a task using human-grade reasoning with shape governance."""
        hypotheses = self.analyze_task(train_pairs)
        
        if not hypotheses:
            return test_input  # Fallback
        
        # CRITICAL: Determine target shape from training data
        expected_output_shape = None
        if train_pairs:
            output_shapes = [out.shape for _, out in train_pairs]
            if len(set(output_shapes)) == 1:  # Consistent output shape
                expected_output_shape = output_shapes[0]
                print(f"DEBUG: Target shape from training: {expected_output_shape}")
            else:
                # INCONSISTENT SHAPES: Choose representative shape using test input
                print(f"DEBUG: Inconsistent output shapes: {output_shapes}")
                # Strategy: Find best matching shape based on test input size and pattern
                expected_output_shape = self._choose_representative_output_shape(output_shapes, test_input.shape)
                print(f"DEBUG: Chose representative target shape: {expected_output_shape}")
        
        # Try hypotheses with shape enforcement and anchor variants
        for hypothesis in hypotheses:
            if hypothesis.verification_score > 0.3:  # Reasonable threshold
                print(f"Trying hypothesis: {hypothesis.name} (score: {hypothesis.verification_score:.3f})")
                
                try:
                    # SHAPE GOVERNANCE: Try multiple anchor variants
                    best_result = None
                    best_accuracy = 0
                    
                    for anchor_r in range(3):  # Try 0, 1, 2 offsets
                        for anchor_c in range(3):
                            try:
                                # Apply hypothesis with anchor variant
                                if 'spatial_formula' in hypothesis.name:
                                    result = self._construct_by_spatial_formula_with_anchor(
                                        test_input, expected_output_shape or test_input.shape, anchor_r, anchor_c)
                                elif 'multi_region' in hypothesis.name:
                                    result = self._compose_from_multiple_regions_with_anchor(
                                        test_input, expected_output_shape or test_input.shape, anchor_r, anchor_c)
                                else:
                                    result = hypothesis.construction_rule(test_input)
                                
                                # SHAPE CONSTRAINT: Force target shape if specified
                                if expected_output_shape and result.shape != expected_output_shape:
                                    result = self._force_target_shape(result, expected_output_shape)
                                
                                # Evaluate against training data if available
                                if train_pairs and expected_output_shape:
                                    accuracy = self._evaluate_result_accuracy(result, train_pairs, test_input)
                                    if accuracy > best_accuracy:
                                        best_accuracy = accuracy
                                        best_result = result
                                        print(f"  -> Anchor ({anchor_r},{anchor_c}) accuracy: {accuracy:.3f}")
                                else:
                                    best_result = result
                                    break  # No training data to compare
                                    
                            except Exception:
                                continue
                    
                    if best_result is not None:
                        print(f"  -> Best result shape: {best_result.shape}, accuracy: {best_accuracy:.3f}")
                        return best_result
                        
                except Exception as e:
                    print(f"  -> Failed: {e}")
                    continue
        
        # ULTIMATE FALLBACK with shape enforcement
        if hypotheses and expected_output_shape:
            print(f"Fallback: forcing shape {expected_output_shape}")
            try:
                result = hypotheses[0].construction_rule(test_input)
                return self._force_target_shape(result, expected_output_shape)
            except Exception:
                pass
        
        return test_input  # Ultimate fallback
    
    def _choose_representative_output_shape(self, output_shapes: List[Tuple[int, int]], 
                                          test_input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Choose representative output shape for inconsistent training data."""
        if not output_shapes:
            return test_input_shape
        
        # Strategy 1: If one shape appears multiple times, use it
        from collections import Counter
        shape_counts = Counter(output_shapes)
        most_common = shape_counts.most_common(1)[0]
        if most_common[1] > 1:  # Appears more than once
            return most_common[0]
        
        # Strategy 2: Choose shape with best aspect ratio match to input
        input_ratio = test_input_shape[0] / test_input_shape[1] if test_input_shape[1] > 0 else 1
        best_shape = output_shapes[0]
        best_ratio_diff = float('inf')
        
        for shape in output_shapes:
            shape_ratio = shape[0] / shape[1] if shape[1] > 0 else 1
            ratio_diff = abs(input_ratio - shape_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_shape = shape
        
        return best_shape
    
    def _force_target_shape(self, result: Array, target_shape: Tuple[int, int]) -> Array:
        """Force result to target shape using smart adaptation."""
        if result.shape == target_shape:
            return result
        
        target_h, target_w = target_shape
        result_h, result_w = result.shape
        
        # Strategy 1: If result is smaller, tile it
        if result_h <= target_h and result_w <= target_w:
            tile_h = (target_h + result_h - 1) // result_h  # Ceiling division
            tile_w = (target_w + result_w - 1) // result_w
            tiled = np.tile(result, (tile_h, tile_w))
            return tiled[:target_h, :target_w]
        
        # Strategy 2: If result is larger, find best crop
        if result_h >= target_h and result_w >= target_w:
            best_crop = None
            best_diversity = 0
            
            for r in range(result_h - target_h + 1):
                for c in range(result_w - target_w + 1):
                    crop = result[r:r+target_h, c:c+target_w]
                    diversity = len(np.unique(crop))
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_crop = crop
            
            if best_crop is not None:
                return best_crop
        
        # Strategy 3: Center placement with padding
        output = np.zeros(target_shape, dtype=result.dtype)
        start_r = max(0, (target_h - result_h) // 2)
        start_c = max(0, (target_w - result_w) // 2)
        end_r = min(target_h, start_r + result_h)
        end_c = min(target_w, start_c + result_w)
        
        src_r = min(result_h, end_r - start_r)
        src_c = min(result_w, end_c - start_c)
        
        output[start_r:end_r, start_c:end_c] = result[:src_r, :src_c]
        return output
    
    def _compose_from_multiple_regions_with_anchor(self, inp: Array, target_shape: Tuple[int, int],
                                                  anchor_r: int, anchor_c: int) -> Array:
        """Multi-region composition with anchor offset."""
        target_h, target_w = target_shape
        h, w = inp.shape
        
        if target_shape == (h, w):
            return self._complete_repeating_patterns_with_anchor(inp, anchor_r)
        
        # Find interesting regions with anchor offset
        interesting_regions = []
        for r in range(max(0, anchor_r), min(h - target_h + 1, h - anchor_r)):
            for c in range(max(0, anchor_c), min(w - target_w + 1, w - anchor_c)):
                region = inp[r:r+target_h, c:c+target_w]
                diversity = len(np.unique(region))
                if diversity >= 2:  # Lower threshold with anchor
                    interesting_regions.append((region, diversity, r, c))
        
        if interesting_regions:
            interesting_regions.sort(key=lambda x: x[1], reverse=True)
            return interesting_regions[0][0].copy()
        
        # Fallback with anchor
        center_r = max(0, h // 2 - target_h // 2 + anchor_r)
        center_c = max(0, w // 2 - target_w // 2 + anchor_c)
        
        if center_r + target_h <= h and center_c + target_w <= w:
            return inp[center_r:center_r+target_h, center_c:center_c+target_w].copy()
        
        return np.zeros(target_shape, dtype=inp.dtype)
    
    def _evaluate_result_accuracy(self, result: Array, train_pairs: List[Tuple[Array, Array]], 
                                 test_input: Array) -> float:
        """Evaluate result accuracy by testing pattern on training data."""
        if not train_pairs:
            return 0.0
        
        # Use the pattern from result to predict training outputs
        total_accuracy = 0.0
        for inp, expected_out in train_pairs:
            try:
                # Simple pattern matching - if result shape matches expected output
                if result.shape == expected_out.shape:
                    accuracy = np.sum(result == expected_out) / expected_out.size
                    total_accuracy += accuracy
            except Exception:
                continue
        
        return total_accuracy / len(train_pairs)
