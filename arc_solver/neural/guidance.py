"""
Neural guidance module for ARC program synthesis.

This module implements a lightweight classifier that predicts which DSL operations
are likely to be relevant for a given ARC task. The classifier is trained on
features extracted from training pairs and guides the search process by scoring
candidate operations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..grid import Array
from ..features import extract_task_features


class SimpleClassifier:
    """A simple multi-layer perceptron for operation prediction.
    
    This lightweight classifier predicts which DSL operations are likely
    to be relevant for solving a given ARC task based on extracted features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias1 = np.zeros(hidden_dim)
        self.weights2 = np.random.randn(hidden_dim, 7)  # 7 operation types
        self.bias2 = np.zeros(7)
        
        # Operation mapping
        self.operations = ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        # First layer
        h = np.maximum(0, np.dot(x, self.weights1) + self.bias1)  # ReLU
        # Output layer with sigmoid
        out = 1.0 / (1.0 + np.exp(-(np.dot(h, self.weights2) + self.bias2)))
        return out.squeeze()
    
    def predict_operations(self, features: Dict[str, Any], threshold: float = 0.5) -> List[str]:
        """Predict which operations are likely relevant."""
        feature_vector = self._features_to_vector(features)
        probabilities = self.forward(feature_vector).ravel()

        relevant_ops = []
        for i, prob in enumerate(probabilities):
            if float(prob) > threshold:
                relevant_ops.append(self.operations[i])
        
        return relevant_ops if relevant_ops else ['identity']
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy vector."""
        # For now, use a simple heuristic mapping
        # In practice, you'd train this on labeled data
        vector = np.array([
            features.get('num_train_pairs', 0) / 10.0,
            features.get('input_height_mean', 0) / 30.0,
            features.get('input_width_mean', 0) / 30.0,
            features.get('shape_preserved', 0),
            features.get('input_colors_mean', 0) / 10.0,
            features.get('output_colors_mean', 0) / 10.0,
            features.get('background_color_consistent', 0),
            features.get('has_color_mapping', 0),
            features.get('input_objects_mean', 0) / 20.0,
            features.get('output_objects_mean', 0) / 20.0,
            features.get('object_count_preserved', 0),
            features.get('likely_rotation', 0),
            features.get('likely_reflection', 0),
            features.get('likely_translation', 0),
            features.get('likely_recolor', 0),
            features.get('likely_crop', 0),
            features.get('likely_pad', 0),
        ])
        
        return vector.reshape(1, -1)


class HeuristicGuidance:
    """Heuristic-based guidance for operation selection.
    
    This class provides operation guidance based on simple heuristics derived
    from task features. It serves as a baseline and fallback when neural
    guidance is not available.
    """
    
    def __init__(self):
        self.operations = ['rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad']
    
    def predict_operations(self, features: Dict[str, Any], threshold: float = 0.3) -> List[str]:
        """Predict operations using heuristic rules."""
        relevant_ops = []
        
        # Rotation: likely if square grids and rotation pattern detected
        if (features.get('likely_rotation', 0) > threshold and 
            abs(features.get('input_height_mean', 0) - features.get('input_width_mean', 0)) < 2):
            relevant_ops.append('rotate')
        
        # Reflection: likely if reflection pattern detected
        if features.get('likely_reflection', 0) > threshold:
            relevant_ops.extend(['flip', 'transpose'])
        
        # Translation: likely if shape preserved and translation detected
        if (features.get('shape_preserved', 0) and 
            features.get('likely_translation', 0) > threshold):
            relevant_ops.append('translate')
        
        # Recolor: likely if color mapping detected
        if features.get('likely_recolor', 0) > threshold:
            relevant_ops.append('recolor')
        
        # Crop: likely if output smaller than input
        if features.get('likely_crop', 0) > threshold:
            relevant_ops.append('crop')
        
        # Pad: likely if output larger than input
        if features.get('likely_pad', 0) > threshold:
            relevant_ops.append('pad')
        
        # Always include identity as fallback
        if not relevant_ops:
            relevant_ops = ['identity']
        
        return list(set(relevant_ops))  # Remove duplicates


class NeuralGuidance:
    """Main neural guidance interface.
    
    This class manages both neural and heuristic guidance, providing a unified
    interface for operation prediction during program synthesis.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.heuristic_guidance = HeuristicGuidance()
        self.neural_model = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # For now, create a dummy neural model
            self.neural_model = SimpleClassifier(17)  # 17 features
    
    def predict_operations(self, train_pairs: List[Tuple[Array, Array]]) -> List[str]:
        """Predict which operations are likely relevant for the task."""
        features = extract_task_features(train_pairs)
        
        # Try neural guidance first, fall back to heuristic
        if self.neural_model:
            try:
                return self.neural_model.predict_operations(features)
            except Exception:
                pass
        
        return self.heuristic_guidance.predict_operations(features)
    
    def score_operations(self, train_pairs: List[Tuple[Array, Array]]) -> Dict[str, float]:
        """Get operation relevance scores."""
        features = extract_task_features(train_pairs)
        
        scores = {
            'rotate': features.get('likely_rotation', 0),
            'flip': features.get('likely_reflection', 0) * 0.7,
            'transpose': features.get('likely_reflection', 0) * 0.3,
            'translate': features.get('likely_translation', 0),
            'recolor': features.get('likely_recolor', 0),
            'crop': features.get('likely_crop', 0),
            'pad': features.get('likely_pad', 0),
            'identity': 0.1,  # Always a small baseline
        }
        
        return scores
    
    def load_model(self, model_path: str) -> None:
        """Load a trained neural model from ``model_path``.

        The model is stored as JSON containing the network weights and
        configuration.  If loading fails, a :class:`ValueError` is raised to
        signal the caller that the model file is invalid.
        """
        try:
            with open(model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:  # pragma: no cover - defensive
            raise FileNotFoundError(f"model file not found: {model_path}") from exc
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"invalid model file: {exc}") from exc

        try:
            self.neural_model = SimpleClassifier(
                input_dim=int(data["input_dim"]),
                hidden_dim=int(data.get("hidden_dim", 32)),
            )
            self.neural_model.weights1 = np.array(data["weights1"], dtype=float)
            self.neural_model.bias1 = np.array(data["bias1"], dtype=float)
            self.neural_model.weights2 = np.array(data["weights2"], dtype=float)
            self.neural_model.bias2 = np.array(data["bias2"], dtype=float)
            if "operations" in data:
                self.neural_model.operations = list(data["operations"])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(f"missing field in model file: {exc}") from exc

    def save_model(self, model_path: str) -> None:
        """Persist the neural model to ``model_path`` in JSON format."""
        if self.neural_model is None:
            raise ValueError("no neural model to save")

        data = {
            "input_dim": self.neural_model.input_dim,
            "hidden_dim": self.neural_model.hidden_dim,
            "weights1": self.neural_model.weights1.tolist(),
            "bias1": self.neural_model.bias1.tolist(),
            "weights2": self.neural_model.weights2.tolist(),
            "bias2": self.neural_model.bias2.tolist(),
            "operations": self.neural_model.operations,
        }

        tmp_path = f"{model_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, model_path)
