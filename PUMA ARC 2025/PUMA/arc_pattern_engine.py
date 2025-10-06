#!/usr/bin/env python3
"""Enhanced pattern recognition for ARC using RFT principles."""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass 
class TransformationRule:
    """Represents a learned transformation rule from RFT analysis."""
    name: str
    preconditions: List[str]  # What conditions must be met
    transformations: List[Dict[str, Any]]  # What transformations to apply
    confidence: float
    examples: List[Tuple[np.ndarray, np.ndarray]]  # Training examples

class ARCPatternEngine:
    """Enhanced pattern recognition engine using RFT principles."""
    
    def __init__(self):
        self.learned_rules: List[TransformationRule] = []
        self.spatial_templates = {
            "translation": self._detect_translation_pattern,
            "rotation": self._detect_rotation_pattern,
            "reflection": self._detect_reflection_pattern,
            "scaling": self._detect_scaling_pattern,
            "color_mapping": self._detect_color_mapping
        }
    
    def learn_from_examples(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[TransformationRule]:
        """Learn transformation rules from training examples."""
        rules = []
        
        # Detect each type of pattern
        for pattern_name, detector in self.spatial_templates.items():
            rule = detector(train_pairs)
            if rule:
                rules.append(rule)
        
        # Look for composite patterns
        composite_rule = self._detect_composite_patterns(train_pairs, rules)
        if composite_rule:
            rules.append(composite_rule)
        
        self.learned_rules = rules
        return rules
    
    def _detect_translation_pattern(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """Detect consistent translation patterns."""
        from arc_solver.rft import RelationalFrameAnalyzer
        
        analyzer = RelationalFrameAnalyzer()
        facts = analyzer.analyze(pairs)
        
        # Look for consistent spatial transformations
        spatial_facts = facts.get("spatial", [])
        if not spatial_facts:
            return None
        
        # Group by transformation type
        translations = []
        for fact in spatial_facts:
            if fact.relation == "spatial_transform":
                direction = fact.direction_vector
                distance = fact.metadata.get("distance", 0)
                translations.append((direction, distance))
        
        if len(translations) >= 2:
            # Check for consistency
            avg_direction = np.mean([t[0] for t in translations], axis=0)
            avg_distance = np.mean([t[1] for t in translations])
            
            # Calculate consistency score
            direction_consistency = np.mean([
                np.dot(t[0], avg_direction) / (np.linalg.norm(t[0]) * np.linalg.norm(avg_direction))
                for t in translations if np.linalg.norm(t[0]) > 0
            ])
            
            if direction_consistency > 0.8:  # High consistency
                return TransformationRule(
                    name="consistent_translation",
                    preconditions=["objects_present", "same_colors"],
                    transformations=[{
                        "type": "translate",
                        "direction": avg_direction.tolist(),
                        "distance": avg_distance
                    }],
                    confidence=direction_consistency,
                    examples=pairs
                )
        
        return None
    
    def _detect_rotation_pattern(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """Detect rotation patterns by comparing object orientations."""
        # Simplified rotation detection - compare object shapes
        consistent_rotations = []
        
        for inp, out in pairs:
            # Simple heuristic: check if output looks like rotated input
            rotations = [np.rot90(inp, k) for k in range(1, 4)]
            for k, rotated in enumerate(rotations, 1):
                if np.array_equal(rotated, out):
                    consistent_rotations.append(k * 90)
                    break
        
        if consistent_rotations and len(set(consistent_rotations)) == 1:
            # All examples show same rotation
            angle = consistent_rotations[0]
            return TransformationRule(
                name="consistent_rotation",
                preconditions=["grid_rotation_applicable"],
                transformations=[{"type": "rotate", "angle": angle}],
                confidence=1.0,
                examples=pairs
            )
        
        return None
    
    def _detect_reflection_pattern(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """Detect reflection patterns."""
        consistent_reflections = []
        
        for inp, out in pairs:
            if np.array_equal(np.flipud(inp), out):
                consistent_reflections.append("horizontal")
            elif np.array_equal(np.fliplr(inp), out):
                consistent_reflections.append("vertical")
        
        if consistent_reflections and len(set(consistent_reflections)) == 1:
            axis = consistent_reflections[0]
            return TransformationRule(
                name="consistent_reflection",
                preconditions=["grid_reflection_applicable"],
                transformations=[{"type": "reflect", "axis": axis}],
                confidence=1.0,
                examples=pairs
            )
        
        return None
    
    def _detect_scaling_pattern(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """Detect scaling patterns."""
        scale_factors = []
        
        for inp, out in pairs:
            h_scale = out.shape[0] / inp.shape[0]
            w_scale = out.shape[1] / inp.shape[1]
            if abs(h_scale - w_scale) < 0.1:  # Uniform scaling
                scale_factors.append(h_scale)
        
        if scale_factors and len(set(scale_factors)) == 1:
            factor = scale_factors[0]
            return TransformationRule(
                name="consistent_scaling",
                preconditions=["uniform_scaling_applicable"],
                transformations=[{"type": "scale", "factor": factor}],
                confidence=1.0,
                examples=pairs
            )
        
        return None
    
    def _detect_color_mapping(self, pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """Detect consistent color mappings."""
        color_maps = []
        
        for inp, out in pairs:
            if inp.shape == out.shape:
                # Build color mapping
                mapping = {}
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        inp_color = inp[i, j]
                        out_color = out[i, j]
                        if inp_color in mapping:
                            if mapping[inp_color] != out_color:
                                break  # Inconsistent mapping
                        else:
                            mapping[inp_color] = out_color
                else:
                    color_maps.append(mapping)
        
        if color_maps and all(cm == color_maps[0] for cm in color_maps):
            # Consistent color mapping across all examples
            return TransformationRule(
                name="consistent_color_mapping",
                preconditions=["same_shape", "color_remapping_applicable"],
                transformations=[{"type": "color_map", "mapping": color_maps[0]}],
                confidence=1.0,
                examples=pairs
            )
        
        return None
    
    def _detect_composite_patterns(self, pairs: List[Tuple[np.ndarray, np.ndarray]], 
                                   simple_rules: List[TransformationRule]) -> Optional[TransformationRule]:
        """Detect patterns that combine multiple simple transformations."""
        if len(simple_rules) >= 2:
            # Try combining the top 2 rules
            rule1, rule2 = simple_rules[:2]
            combined_transforms = rule1.transformations + rule2.transformations
            
            return TransformationRule(
                name="composite_transformation",
                preconditions=rule1.preconditions + rule2.preconditions,
                transformations=combined_transforms,
                confidence=min(rule1.confidence, rule2.confidence) * 0.9,
                examples=pairs
            )
        
        return None
    
    def apply_rules(self, test_input: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """Apply learned rules to generate test output predictions."""
        predictions = []
        
        for rule in self.learned_rules:
            if self._check_preconditions(test_input, rule.preconditions):
                prediction = self._apply_transformations(test_input, rule.transformations)
                if prediction is not None:
                    predictions.append((prediction, rule.confidence))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def _check_preconditions(self, grid: np.ndarray, preconditions: List[str]) -> bool:
        """Check if preconditions are met for applying a rule."""
        # Simplified precondition checking
        for condition in preconditions:
            if condition == "objects_present" and np.all(grid == 0):
                return False
            elif condition == "same_colors" and len(np.unique(grid)) < 2:
                return False
        return True
    
    def _apply_transformations(self, grid: np.ndarray, transformations: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Apply a sequence of transformations to a grid."""
        result = grid.copy()
        
        for transform in transformations:
            if transform["type"] == "translate":
                # Apply translation (simplified)
                direction = np.array(transform["direction"])
                # For now, just return original - implement actual translation
                continue
            elif transform["type"] == "rotate":
                angle = transform["angle"]
                k = angle // 90
                result = np.rot90(result, k)
            elif transform["type"] == "reflect":
                axis = transform["axis"]
                if axis == "horizontal":
                    result = np.flipud(result)
                elif axis == "vertical":
                    result = np.fliplr(result)
            elif transform["type"] == "color_map":
                mapping = transform["mapping"]
                for old_color, new_color in mapping.items():
                    result[result == old_color] = new_color
            elif transform["type"] == "scale":
                # Implement scaling if needed
                continue
        
        return result

# Integration function to use with existing solver
def enhance_solver_with_patterns(solver):
    """Enhance existing solver with pattern recognition capabilities."""
    pattern_engine = ARCPatternEngine()
    
    # Add pattern engine to solver
    solver.pattern_engine = pattern_engine
    
    # Override or extend solve method to use patterns
    original_solve = solver.solve
    
    def enhanced_solve(task):
        # Learn patterns from training examples
        train_pairs = [(np.array(ex["input"]), np.array(ex["output"])) 
                      for ex in task["train"]]
        
        rules = pattern_engine.learn_from_examples(train_pairs)
        print(f"Learned {len(rules)} transformation rules")
        
        # Get predictions from pattern engine
        test_input = np.array(task["test"][0]["input"])
        pattern_predictions = pattern_engine.apply_rules(test_input)
        
        if pattern_predictions:
            # Return best pattern-based prediction
            best_prediction, confidence = pattern_predictions[0]
            print(f"Pattern-based prediction with confidence {confidence:.3f}")
            return best_prediction.tolist()
        else:
            # Fall back to original solver
            return original_solve(task)
    
    solver.solve = enhanced_solve
    return solver
