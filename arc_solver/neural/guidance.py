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
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..grid import Array
from ..features import extract_task_features
from ..rft_engine.engine import RFTInference


class SimpleClassifier:
    """A simple multi-layer perceptron for operation prediction.
    
    This lightweight classifier predicts which DSL operations are likely
    to be relevant for solving a given ARC task based on extracted features.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        rng = np.random.default_rng(0)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights1 = rng.standard_normal((input_dim, hidden_dim)) * 0.1
        self.bias1 = np.zeros(hidden_dim)
        self.weights2 = rng.standard_normal((hidden_dim, 7))
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

    def train(
        self, X: np.ndarray, Y: np.ndarray, epochs: int = 50, lr: float = 0.1
    ) -> None:
        """Train the network using simple gradient descent."""
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have matching first dimension")

        for _ in range(epochs):
            # Forward pass
            h = np.maximum(0, X @ self.weights1 + self.bias1)
            out = 1.0 / (1.0 + np.exp(-(h @ self.weights2 + self.bias2)))

            # Gradients for output layer (sigmoid + BCE)
            grad_out = (out - Y) / X.shape[0]
            grad_w2 = h.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)

            # Backprop into hidden layer (ReLU)
            grad_h = grad_out @ self.weights2.T
            grad_h[h <= 0] = 0
            grad_w1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            # Parameter update
            self.weights2 -= lr * grad_w2
            self.bias2 -= lr * grad_b2
            self.weights1 -= lr * grad_w1
            self.bias1 -= lr * grad_b1

    def online_update(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        reward: float,
        lr: float = 0.05,
    ) -> None:
        """Perform a single weighted gradient step using reward scaling."""

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)
        reward = max(0.0, min(1.0, float(reward)))
        h = np.maximum(0, X @ self.weights1 + self.bias1)
        out = 1.0 / (1.0 + np.exp(-(h @ self.weights2 + self.bias2)))
        grad_out = (out - Y) * reward
        grad_w2 = h.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_h = grad_out @ self.weights2.T
        grad_h[h <= 0] = 0
        grad_w1 = X.T @ grad_h
        grad_b1 = grad_h.sum(axis=0)
        self.weights2 -= lr * grad_w2
        self.bias2 -= lr * grad_b2
        self.weights1 -= lr * grad_w1
        self.bias1 -= lr * grad_b1
    
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
        self.operations = [
            'rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad',
            'extract_content_region', 'extract_bounded_region', 'extract_largest_rect',
            'extract_central_pattern', 'smart_crop_auto'
        ]
    
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
            relevant_ops.extend(['crop', 'extract_content_region', 'extract_largest_rect'])
            
            # If size reduction is significant, prioritize extraction operations
            input_size = features.get('input_height_mean', 0) * features.get('input_width_mean', 0)
            output_size = features.get('output_height_mean', 0) * features.get('output_width_mean', 0)
            
            if input_size > 0 and output_size / input_size < 0.3:  # Significant size reduction
                relevant_ops.extend([
                    'extract_content_region', 'extract_bounded_region', 
                    'extract_central_pattern', 'smart_crop_auto'
                ])
        
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
        expected_dim = 17
        self.online_lr = 0.05
        self.operation_stats: Dict[str, Dict[str, float]] = {}

        if model_path and os.path.exists(model_path):
            try:
                self.load_model(model_path)
                if getattr(self.neural_model, "input_dim", expected_dim) != expected_dim:
                    raise ValueError("model input dimension mismatch")
            except Exception:
                self.neural_model = SimpleClassifier(expected_dim)
        else:
            self.neural_model = SimpleClassifier(expected_dim)
        for op in [
            'rotate', 'flip', 'transpose', 'translate', 'recolor', 'crop', 'pad', 'identity'
        ]:
            self.operation_stats.setdefault(op, {"count": 0.0, "mean_reward": 0.0})
    
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
        for op, stat in self.operation_stats.items():
            mean_reward = stat.get("mean_reward", 0.0)
            if op in scores:
                scores[op] = 0.7 * scores[op] + 0.3 * mean_reward
            else:
                scores[op] = mean_reward

        return scores

    def reinforce(
        self,
        train_pairs: List[Tuple[Array, Array]],
        program: List[Tuple[str, Dict[str, Any]]],
        reward: float,
        inference: Optional[RFTInference] = None,
    ) -> None:
        """Update neural guidance policy using reinforcement signal."""

        reward = max(0.0, min(1.0, float(reward)))
        operations: Set[str] = {op for op, _ in program}
        if inference is not None:
            for hints in inference.function_hints.values():
                operations.update(hints)
        for op in operations:
            stats = self.operation_stats.setdefault(op, {"count": 0.0, "mean_reward": 0.0})
            stats["count"] += 1.0
            stats["mean_reward"] += (reward - stats["mean_reward"]) / stats["count"]

        if not self.neural_model:
            return

        features = extract_task_features(train_pairs)
        feature_vec = self.neural_model._features_to_vector(features)
        label = np.zeros(len(self.neural_model.operations))
        for op in operations:
            if op in self.neural_model.operations:
                idx = self.neural_model.operations.index(op)
                label[idx] = 1.0
        self.neural_model.online_update(feature_vec, label, reward, self.online_lr)

    def train_from_episode_db(
        self, db_path: str, epochs: int = 50, lr: float = 0.1
    ) -> None:
        """Train the neural model from an episodic memory database."""
        if self.neural_model is None:
            raise ValueError("neural model not initialised")

        from .episodic import EpisodeDatabase  # Local import to avoid cycle

        db = EpisodeDatabase(db_path)
        db.load()
        features_list: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for episode in db.episodes.values():
            feat = extract_task_features(episode.train_pairs)
            features_list.append(self.neural_model._features_to_vector(feat).ravel())
            label_vec = np.zeros(len(self.neural_model.operations))
            for program in episode.programs:
                for op, _ in program:
                    if op in self.neural_model.operations:
                        idx = self.neural_model.operations.index(op)
                        label_vec[idx] = 1.0
            labels.append(label_vec)

        if not features_list:
            raise ValueError("episode database is empty")

        X = np.vstack(features_list)
        Y = np.vstack(labels)
        self.neural_model.train(X, Y, epochs=epochs, lr=lr)

    def train_from_task_pairs(
        self, tasks: List[List[Tuple[Array, Array]]], epochs: int = 50, lr: float = 0.1
    ) -> None:
        """Train the neural model from raw ARC tasks.

        Tasks are provided as lists of training input/output pairs. Operation
        labels are derived heuristically from extracted features.  This enables
        supervised training even when explicit programs are unavailable.

        Parameters
        ----------
        tasks:
            Iterable of tasks where each task is a list of `(input, output)`
            array pairs.
        epochs:
            Number of training epochs for gradient descent.
        lr:
            Learning rate for gradient descent.
        """  # [S:ALG v1] train_from_task_pairs pass
        if self.neural_model is None:
            raise ValueError("neural model not initialised")

        features_list: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for train_pairs in tasks:
            feat = extract_task_features(train_pairs)
            features_list.append(self.neural_model._features_to_vector(feat).ravel())
            label_vec = np.zeros(len(self.neural_model.operations))
            if feat.get("likely_rotation", 0) > 0.5:
                idx = self.neural_model.operations.index("rotate")
                label_vec[idx] = 1.0
            if feat.get("likely_reflection", 0) > 0.5:
                idx_flip = self.neural_model.operations.index("flip")
                idx_tr = self.neural_model.operations.index("transpose")
                label_vec[idx_flip] = 1.0
                label_vec[idx_tr] = 1.0
            if feat.get("likely_translation", 0) > 0.5:
                idx = self.neural_model.operations.index("translate")
                label_vec[idx] = 1.0
            if feat.get("likely_recolor", 0) > 0.5:
                idx = self.neural_model.operations.index("recolor")
                label_vec[idx] = 1.0
            if feat.get("likely_crop", 0) > 0.5:
                idx = self.neural_model.operations.index("crop")
                label_vec[idx] = 1.0
            if feat.get("likely_pad", 0) > 0.5:
                idx = self.neural_model.operations.index("pad")
                label_vec[idx] = 1.0
            labels.append(label_vec)

        if not features_list:
            raise ValueError("no tasks provided")

        X = np.vstack(features_list)
        Y = np.vstack(labels)
        self.neural_model.train(X, Y, epochs=epochs, lr=lr)
    
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
