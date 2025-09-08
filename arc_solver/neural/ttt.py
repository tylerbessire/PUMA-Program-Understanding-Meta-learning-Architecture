"""
Test-time training (TTT) module for ARC solver.

This module implements lightweight test-time adaptation that fine-tunes
scoring functions on each individual task using the provided training
demonstrations. This helps specialize the solver to each task's specific
patterns and requirements.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from copy import deepcopy

from ..grid import Array, eq
from ..dsl import apply_program


class AdaptiveScorer:
    """Adaptive scoring function that can be fine-tuned at test time."""
    
    def __init__(self, feature_dim: int = 10):
        self.feature_dim = feature_dim
        self.weights = np.ones(feature_dim) / feature_dim  # Initialize uniformly
        self.bias = 0.0
        self.learning_rate = 0.1
        self.history = []
    
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
            except Exception:
                pass
        
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
        
        return features
    
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
                     num_iterations: int = 5) -> AdaptiveScorer:
        """Adapt the scorer to a specific task using training demonstrations."""
        # Start with a copy of the base scorer
        self.adapted_scorer = deepcopy(self.base_scorer)
        
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
                if success_rate >= 0.8:  # High success threshold
                    positive_programs.append(program)
                elif success_rate <= 0.2:  # Low success threshold
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
            except Exception:
                pass
        
        return successes / len(train_pairs) if train_pairs else 0.0
    
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
                              max_augmentations: int = 10) -> List[Tuple[Array, Array]]:
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
                    except Exception:
                        pass
            
            # Try reflections
            for axis in [0, 1]:
                if len(augmented) >= max_augmentations:
                    break
                try:
                    aug_inp = np.flip(inp, axis=axis)
                    aug_out = np.flip(out, axis=axis)
                    augmented.append((aug_inp, aug_out))
                except Exception:
                    pass
        
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
