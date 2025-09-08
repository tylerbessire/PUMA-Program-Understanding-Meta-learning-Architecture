"""
Training script for neural guidance component.

This script demonstrates how to train the neural guidance classifier on
ARC training data. In practice, you would use the 1000 public training
tasks to train the classifier to predict relevant operations.
"""

from __future__ import annotations

import json
import numpy as np
from typing import List, Tuple, Dict, Any
import os

from arc_solver.grid import to_array
from arc_solver.neural.features import extract_task_features
from arc_solver.neural.guidance import SimpleClassifier


def load_training_data(data_path: str) -> List[Dict[str, Any]]:
    """Load ARC training tasks from JSON file."""
    if not os.path.exists(data_path):
        print(f"Training data not found at {data_path}")
        return []
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return list(data.values())


def extract_features_and_labels(tasks: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and operation labels from training tasks."""
    features_list = []
    labels_list = []
    
    for task in tasks:
        # Convert train pairs to arrays
        train_pairs = []
        for pair in task.get('train', []):
            inp = to_array(pair['input'])
            out = to_array(pair['output'])
            train_pairs.append((inp, out))
        
        if not train_pairs:
            continue
        
        # Extract features
        task_features = extract_task_features(train_pairs)
        
        # Convert to feature vector (must match SimpleClassifier expectations)
        feature_vector = np.array([
            task_features.get('num_train_pairs', 0) / 10.0,
            task_features.get('input_height_mean', 0) / 30.0,
            task_features.get('input_width_mean', 0) / 30.0,
            task_features.get('shape_preserved', 0),
            task_features.get('input_colors_mean', 0) / 10.0,
            task_features.get('output_colors_mean', 0) / 10.0,
            task_features.get('background_color_consistent', 0),
            task_features.get('has_color_mapping', 0),
            task_features.get('input_objects_mean', 0) / 20.0,
            task_features.get('output_objects_mean', 0) / 20.0,
            task_features.get('object_count_preserved', 0),
            task_features.get('likely_rotation', 0),
            task_features.get('likely_reflection', 0),
            task_features.get('likely_translation', 0),
            task_features.get('likely_recolor', 0),
            task_features.get('likely_crop', 0),
            task_features.get('likely_pad', 0),
        ])
        
        # Create operation labels (multi-hot encoding)
        # In practice, you'd derive these from known solutions or manual annotation
        operation_labels = np.array([
            task_features.get('likely_rotation', 0) > 0.5,
            task_features.get('likely_reflection', 0) > 0.5,
            task_features.get('likely_reflection', 0) > 0.3,  # transpose
            task_features.get('likely_translation', 0) > 0.5,
            task_features.get('likely_recolor', 0) > 0.5,
            task_features.get('likely_crop', 0) > 0.5,
            task_features.get('likely_pad', 0) > 0.5,
        ], dtype=float)
        
        features_list.append(feature_vector)
        labels_list.append(operation_labels)
    
    if not features_list:
        return np.array([]), np.array([])
    
    return np.array(features_list), np.array(labels_list)


def train_neural_guidance(training_data_path: str = "arc_training_data.json", 
                         epochs: int = 100):
    """Train the neural guidance classifier."""
    print("Loading training data...")
    tasks = load_training_data(training_data_path)
    
    if not tasks:
        print("No training data available. Using synthetic data for demonstration.")
        # Generate some synthetic training examples
        features, labels = generate_synthetic_data(100)
    else:
        print(f"Loaded {len(tasks)} training tasks")
        features, labels = extract_features_and_labels(tasks)
    
    if features.size == 0:
        print("No valid training examples found!")
        return
    
    print(f"Training on {len(features)} examples with {features.shape[1]} features")
    
    # Initialize classifier
    classifier = SimpleClassifier(input_dim=features.shape[1])
    
    # Simple training loop (in practice, use proper ML framework)
    print("Training neural guidance classifier...")
    
    for epoch in range(epochs):
        # Forward pass
        predictions = np.array([classifier.forward(x.reshape(1, -1))[0] for x in features])
        
        # Simple loss (binary cross-entropy)
        loss = -np.mean(labels * np.log(predictions + 1e-8) + 
                       (1 - labels) * np.log(1 - predictions + 1e-8))
        
        # Very simple gradient update (not a full implementation)
        # In practice, use proper backpropagation
        for i in range(len(features)):
            error = predictions[i] - labels[i]
            # Simplified weight update
            classifier.weights2 -= 0.001 * np.outer(classifier.forward(features[i].reshape(1, -1)), error)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Save the trained model
    model_path = "neural_guidance_model.json"
    save_model(classifier, model_path)
    print(f"Model saved to {model_path}")
    
    # Evaluate performance
    evaluate_model(classifier, features, labels)


def generate_synthetic_data(num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for demonstration."""
    print("Generating synthetic training data...")
    
    features = []
    labels = []
    
    for _ in range(num_samples):
        # Random features
        feature_vector = np.random.rand(17)
        
        # Synthetic labels based on simple rules
        operation_labels = np.array([
            feature_vector[11] > 0.6,  # rotation if likely_rotation high
            feature_vector[12] > 0.6,  # flip if likely_reflection high
            feature_vector[12] > 0.4,  # transpose if likely_reflection medium
            feature_vector[13] > 0.6,  # translate if likely_translation high
            feature_vector[14] > 0.6,  # recolor if likely_recolor high
            feature_vector[15] > 0.6,  # crop if likely_crop high
            feature_vector[16] > 0.6,  # pad if likely_pad high
        ], dtype=float)
        
        features.append(feature_vector)
        labels.append(operation_labels)
    
    return np.array(features), np.array(labels)


def save_model(classifier: SimpleClassifier, filepath: str):
    """Save trained classifier to file."""
    model_data = {
        'weights1': classifier.weights1.tolist(),
        'bias1': classifier.bias1.tolist(),
        'weights2': classifier.weights2.tolist(),
        'bias2': classifier.bias2.tolist(),
        'input_dim': classifier.input_dim,
        'hidden_dim': classifier.hidden_dim,
        'operations': classifier.operations,
    }
    
    with open(filepath, 'w') as f:
        json.dump(model_data, f, indent=2)


def load_model(filepath: str) -> SimpleClassifier:
    """Load trained classifier from file."""
    with open(filepath, 'r') as f:
        model_data = json.load(f)
    
    classifier = SimpleClassifier(model_data['input_dim'], model_data['hidden_dim'])
    classifier.weights1 = np.array(model_data['weights1'])
    classifier.bias1 = np.array(model_data['bias1'])
    classifier.weights2 = np.array(model_data['weights2'])
    classifier.bias2 = np.array(model_data['bias2'])
    classifier.operations = model_data['operations']
    
    return classifier


def evaluate_model(classifier: SimpleClassifier, features: np.ndarray, labels: np.ndarray):
    """Evaluate classifier performance."""
    predictions = np.array([classifier.forward(x.reshape(1, -1))[0] for x in features])
    
    # Threshold predictions
    binary_predictions = (predictions > 0.5).astype(float)
    
    # Calculate accuracy per operation
    operation_names = classifier.operations
    for i, op_name in enumerate(operation_names):
        op_accuracy = np.mean(binary_predictions[:, i] == labels[:, i])
        print(f"{op_name}: {op_accuracy:.3f} accuracy")
    
    # Overall accuracy
    overall_accuracy = np.mean(binary_predictions == labels)
    print(f"Overall accuracy: {overall_accuracy:.3f}")


if __name__ == "__main__":
    # Train the neural guidance model
    train_neural_guidance()
