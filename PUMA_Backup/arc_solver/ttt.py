"""
Test-time training (TTT) module for ARC solver.

This module implements lightweight test-time adaptation that fine-tunes
scoring functions on each individual task using the provided training
demonstrations. This helps specialize the solver to each task's specific
patterns and requirements.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

from .grid import Array, eq
from .dsl import apply_program

logger = logging.getLogger(__name__)

__all__ = ["AdaptiveScorer", "TestTimeTrainer", "DataAugmentation"]


class AdaptiveScorer:
    """Adaptive scoring function that can be fine-tuned at test time."""
    
    def __init__(self, feature_dim: int = 11):
        self.feature_dim = feature_dim
        self.weights = np.ones(feature_dim) / feature_dim  # Initialize uniformly
        self.bias = 0.0
        self.learning_rate = 0.1
        self.history = []
        self.detector_hint: Optional[str] = None

    def set_detector_hint(self, hint: Optional[str]) -> None:
        self.detector_hint = hint
    
    def extract_program_features(self, program: List[Tuple[str, Dict[str, Any]]], 
                                train_pairs: List[Tuple[Array, Array]]) -> np.ndarray:
        """Extract features from a program and its performance on training pairs."""
        features = np.zeros(self.feature_dim)
        
        # Feature 0: Program length
        features[0] = len(program) / 5.0  # Normalize by typical max length
        
        # Feature 1: Number of unique operations
        unique_ops = len(set(op_name for op_name, _ in program))
        features[1] = unique_ops / len(program) if program else 0
        
        # Feature 2-3: Exact match rate on training pairs
        exact_matches = 0
        partial_matches = 0
        for inp, target_out in train_pairs:
            try:
                pred_out = apply_program(inp, program)
                if eq(pred_out, target_out):
                    exact_matches += 1
                else:
                    # Compute partial match (e.g., correct shape)
                    if pred_out.shape == target_out.shape:
                        partial_matches += 1
            except Exception as exc:
                logger.warning(
                    "Program execution failed during feature extraction: %s", exc
                )
                continue
        
        features[2] = exact_matches / len(train_pairs)
        features[3] = partial_matches / len(train_pairs)
        
        # Feature 4-8: Operation type indicators
        op_indicators = {'rotate': 4, 'flip': 5, 'transpose': 6, 'translate': 7, 'recolor': 8}
        for op_name, _ in program:
            if op_name in op_indicators:
                features[op_indicators[op_name]] = 1.0

        # Feature 9: Complexity score (based on parameter diversity)
        if program:
            param_complexity = sum(len(params) for _, params in program) / len(program)
            features[9] = min(param_complexity / 3.0, 1.0)  # Normalize

        # Feature 10: Alignment with detector hint
        if self.detector_hint is not None:
            features[10] = 1.0 if self._program_matches_hint(program) else 0.0

        return features

    def _program_matches_hint(self, program: List[Tuple[str, Dict[str, Any]]]) -> bool:
        if not program or self.detector_hint is None:
            return False

        hint = self.detector_hint
        if hint in {"translate_objects", "gravity_drop"}:
            return any(op_name == "translate" for op_name, _ in program)
        if hint in {"expand_rare_color", "color_count_fill", "change_background"}:
            return any(op_name == "recolor" for op_name, _ in program)
        if hint == "encode_color_counts":
            return any(op_name in {"encode_color_counts", "histogram"} for op_name, _ in program)
        if hint == "add_border":
            return any(op_name in {"pad", "add_border"} for op_name, _ in program)
        if hint == "remove_border":
            return any(op_name in {"crop", "remove_border"} for op_name, _ in program)
        if hint == "remove_colors_and_crop":
            return any(op_name in {"recolor", "crop"} for op_name, _ in program)
        if hint == "recolor_crop_subgrid":
            return any(op_name in {"recolor", "crop"} for op_name, _ in program)
        if hint == "rotate_crop_subgrid":
            return any(op_name in {"rotate", "crop"} for op_name, _ in program)
        if hint == "draw_diagonal":
            return any(op_name in {"draw_line", "recolor"} for op_name, _ in program)
        if hint in {"tile_grid", "repeat_pattern"}:
            return any(op_name in {"tile", "repeat"} for op_name, _ in program)
        return False
    
    def score_program(self, program: List[Tuple[str, Dict[str, Any]]], 
                     train_pairs: List[Tuple[Array, Array]]) -> float:
        """Score a program using the current weights."""
        features = self.extract_program_features(program, train_pairs)
        return np.dot(self.weights, features) + self.bias
    
    def update_weights(self, positive_programs: List[List[Tuple[str, Dict[str, Any]]]], 
                      negative_programs: List[List[Tuple[str, Dict[str, Any]]]], 
                      train_pairs: List[Tuple[Array, Array]]):
        """Update weights based on positive and negative examples."""
        if not positive_programs and not negative_programs:
            return
        
        # Extract features for positive and negative examples
        pos_features = []
        for prog in positive_programs:
            features = self.extract_program_features(prog, train_pairs)
            pos_features.append(features)
        
        neg_features = []
        for prog in negative_programs:
            features = self.extract_program_features(prog, train_pairs)
            neg_features.append(features)
        
        # Simple gradient update
        if pos_features:
            pos_mean = np.mean(pos_features, axis=0)
            self.weights += self.learning_rate * pos_mean
        
        if neg_features:
            neg_mean = np.mean(neg_features, axis=0)
            self.weights -= self.learning_rate * neg_mean
        
        # Normalize weights
        self.weights = np.clip(self.weights, 0.01, 2.0)
        self.weights /= np.sum(self.weights)


class TestTimeTrainer:
    """Main test-time training orchestrator."""
    
    def __init__(self):
        self.base_scorer = AdaptiveScorer()
        self.adapted_scorer = None
        self.adaptation_history = []
    
    def adapt_to_task(self, train_pairs: List[Tuple[Array, Array]], 
                     candidate_programs: List[List[Tuple[str, Dict[str, Any]]]], 
                     num_iterations: int = 5,
                     detector_hint: Optional[str] = None) -> AdaptiveScorer:
        """Adapt the scorer to a specific task using training demonstrations."""
        # Start with a copy of the base scorer
        self.adapted_scorer = deepcopy(self.base_scorer)
        self.adapted_scorer.set_detector_hint(detector_hint)
        
        augmented_pairs = DataAugmentation.augment_training_pairs(train_pairs, max_augmentations=15)

        augmented_pairs = DataAugmentation.augment_training_pairs(
            train_pairs,
            max_augmentations=20,
            include_color_shuffle=True,
        )

        for iteration in range(num_iterations):
            # Score all candidate programs
            program_scores = []
            for program in candidate_programs:
                score = self.adapted_scorer.score_program(program, train_pairs)
                program_scores.append((score, program))
            
            # Separate into positive and negative examples based on actual performance
            positive_programs = []
            negative_programs = []
            
            for score, program in program_scores:
                # Check if program actually works on training pairs
                success_rate = self._evaluate_program(program, train_pairs)
                stability = self._stability_score(program, augmented_pairs)
                if success_rate >= 0.8 and stability >= 0.7:
                    positive_programs.append(program)
                elif success_rate <= 0.2 or stability <= 0.2:
                    negative_programs.append(program)
            
            # Update weights if we have examples
            if positive_programs or negative_programs:
                self.adapted_scorer.update_weights(
                    positive_programs, negative_programs, train_pairs
                )
            
            # Track adaptation progress
            self.adaptation_history.append({
                'iteration': iteration,
                'positive_count': len(positive_programs),
                'negative_count': len(negative_programs),
                'weights': self.adapted_scorer.weights.copy(),
            })
        
        return self.adapted_scorer
    
    def _evaluate_program(self, program: List[Tuple[str, Dict[str, Any]]], 
                         train_pairs: List[Tuple[Array, Array]]) -> float:
        """Evaluate how well a program performs on training pairs."""
        successes = 0
        for inp, target_out in train_pairs:
            try:
                pred_out = apply_program(inp, program)
                if eq(pred_out, target_out):
                    successes += 1
            except Exception as exc:
                logger.warning(
                    "Program evaluation failed during adaptation: %s", exc
                )
                continue
        
        return successes / len(train_pairs) if train_pairs else 0.0

    def _stability_score(self, program: List[Tuple[str, Dict[str, Any]]], 
                         augmented_pairs: List[Tuple[Array, Array]]) -> float:
        if not augmented_pairs:
            return 0.0
        successes = 0
        for inp, target_out in augmented_pairs:
            try:
                pred_out = apply_program(inp, program)
                if eq(pred_out, target_out):
                    successes += 1
            except Exception:
                continue
        return successes / len(augmented_pairs)
    
    def score_with_adaptation(self, program: List[Tuple[str, Dict[str, Any]]], 
                            train_pairs: List[Tuple[Array, Array]]) -> float:
        """Score a program using the adapted scorer."""
        if self.adapted_scorer is None:
            return self.base_scorer.score_program(program, train_pairs)
        return self.adapted_scorer.score_program(program, train_pairs)
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Get statistics about the adaptation process."""
        if not self.adaptation_history:
            return {}
        
        final_weights = self.adaptation_history[-1]['weights']
        total_positives = sum(h['positive_count'] for h in self.adaptation_history)
        total_negatives = sum(h['negative_count'] for h in self.adaptation_history)
        
        return {
            'iterations': len(self.adaptation_history),
            'total_positive_examples': total_positives,
            'total_negative_examples': total_negatives,
            'final_weights': final_weights.tolist(),
            'weight_variance': np.var(final_weights),
        }


class DataAugmentation:
    """Generate synthetic training examples for test-time adaptation."""
    
    @staticmethod
    def augment_training_pairs(train_pairs: List[Tuple[Array, Array]], 
                              max_augmentations: int = 10,
                              include_color_shuffle: bool = True) -> List[Tuple[Array, Array]]:
        """Generate augmented training pairs through simple transformations."""
        augmented = list(train_pairs)  # Start with original pairs
        
        for inp, out in train_pairs:
            # Try rotations if grids are square
            if inp.shape[0] == inp.shape[1] and out.shape[0] == out.shape[1]:
                for k in [1, 2, 3]:
                    if len(augmented) >= max_augmentations:
                        break
                    try:
                        aug_inp = np.rot90(inp, k)
                        aug_out = np.rot90(out, k)
                        augmented.append((aug_inp, aug_out))
                    except Exception as exc:
                        logger.warning(
                            "Rotation augmentation failed (k=%s): %s", k, exc
                        )
                        continue
            
            # Try reflections
            for axis in [0, 1]:
                if len(augmented) >= max_augmentations:
                    break
                try:
                    aug_inp = np.flip(inp, axis=axis)
                    aug_out = np.flip(out, axis=axis)
                    augmented.append((aug_inp, aug_out))
                except Exception as exc:
                    logger.warning(
                        "Reflection augmentation failed (axis=%s): %s", axis, exc
                    )
                    continue

            if include_color_shuffle and len(augmented) < max_augmentations:
                unique_colors = np.unique(inp)
                if unique_colors.size > 1:
                    perm = unique_colors.copy()
                    np.random.shuffle(perm)
                    mapping = {int(c): int(p) for c, p in zip(unique_colors, perm)}
                    aug_inp = np.vectorize(lambda v: mapping.get(int(v), int(v)))(inp)
                    aug_out = np.vectorize(lambda v: mapping.get(int(v), int(v)))(out)
                    augmented.append((aug_inp, aug_out))
        
        return augmented[:max_augmentations]
    
    @staticmethod
    def generate_negative_examples(train_pairs: List[Tuple[Array, Array]], 
                                  num_negatives: int = 5) -> List[Tuple[Array, Array]]:
        """Generate negative examples by applying wrong transformations."""
        negatives = []
        
        for inp, correct_out in train_pairs:
            # Wrong rotation
            if inp.shape[0] == inp.shape[1]:
                wrong_out = np.rot90(inp, 2)  # Arbitrary wrong rotation
                if not eq(wrong_out, correct_out):
                    negatives.append((inp, wrong_out))
            
            # Wrong color mapping
            wrong_out = inp.copy()
            if inp.max() > 0:
                wrong_out[inp == 0] = 1
                wrong_out[inp == 1] = 0
                if not eq(wrong_out, correct_out):
                    negatives.append((inp, wrong_out))
            
            if len(negatives) >= num_negatives:
                break
        
        return negatives
