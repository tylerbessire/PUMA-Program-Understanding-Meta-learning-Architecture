"""
Neural guidance training tool for ARC solver.

This script trains the neural guidance classifier that predicts which DSL operations
are likely relevant for solving a given ARC task. The classifier is trained on
features extracted from the training challenges and their known solutions.
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import extract_task_features
from arc_solver.guidance import SimpleClassifier


def load_training_data(challenges_path: str, solutions_path: str = None) -> List[Dict[str, Any]]:
    """Load ARC training challenges and solutions."""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    solutions = {}
    if solutions_path and Path(solutions_path).exists():
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
    
    tasks = []
    for task_id, task_data in challenges.items():
        task_info = {
            'task_id': task_id,
            'train': task_data['train'],
            'test': task_data['test']
        }
        if task_id in solutions:
            task_info['solutions'] = solutions[task_id]
        tasks.append(task_info)
    
    return tasks


def extract_training_features_and_labels(tasks: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and operation labels from training tasks."""
    features_list = []
    labels_list = []
    
    for task in tasks:
        # Convert train pairs to arrays
        train_pairs = []
        for pair in task['train']:
            inp = to_array(pair['input'])
            out = to_array(pair['output'])
            train_pairs.append((inp, out))
        
        if not train_pairs:
            continue
        
        # Extract comprehensive task features
        task_features = extract_task_features(train_pairs)
        
        # Convert to feature vector matching SimpleClassifier expectations
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
        
        # Create operation labels based on detected patterns
        # In production, these would be derived from known successful solutions
        operation_labels = np.array([
            task_features.get('likely_rotation', 0) > 0.5,    # rotate
            task_features.get('likely_reflection', 0) > 0.5,  # flip
            task_features.get('likely_reflection', 0) > 0.3,  # transpose
            task_features.get('likely_translation', 0) > 0.5, # translate
            task_features.get('likely_recolor', 0) > 0.5,     # recolor
            task_features.get('likely_crop', 0) > 0.5,        # crop
            task_features.get('likely_pad', 0) > 0.5,         # pad
        ], dtype=float)
        
        features_list.append(feature_vector)
        labels_list.append(operation_labels)
    
    return np.array(features_list), np.array(labels_list)


def train_classifier(features: np.ndarray, labels: np.ndarray, epochs: int = 100) -> SimpleClassifier:
    """Train the neural guidance classifier."""
    print(f"Training classifier on {len(features)} examples...")
    
    # Initialize classifier
    classifier = SimpleClassifier(input_dim=features.shape[1])
    
    # Training loop with basic gradient descent
    for epoch in range(epochs):
        total_loss = 0.0
        
        for i in range(len(features)):
            # Forward pass
            prediction = classifier.forward(features[i].reshape(1, -1))[0]
            target = labels[i]
            
            # Binary cross-entropy loss
            loss = -np.mean(target * np.log(prediction + 1e-8) + 
                           (1 - target) * np.log(1 - prediction + 1e-8))
            total_loss += loss
            
            # Simple gradient update (simplified backpropagation)
            error = prediction - target
            learning_rate = 0.001
            
            # Update output layer
            h = np.maximum(0, np.dot(features[i].reshape(1, -1), classifier.weights1) + classifier.bias1)
            classifier.weights2 -= learning_rate * np.outer(h.T, error)
            classifier.bias2 -= learning_rate * error
            
            # Update hidden layer (simplified)
            delta_h = np.dot(error, classifier.weights2.T) * (h > 0).astype(float)
            classifier.weights1 -= learning_rate * np.outer(features[i], delta_h)
            classifier.bias1 -= learning_rate * delta_h.flatten()
        
        avg_loss = total_loss / len(features)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    return classifier


def evaluate_classifier(classifier: SimpleClassifier, features: np.ndarray, labels: np.ndarray):
    """Evaluate classifier performance."""
    predictions = np.array([classifier.forward(x.reshape(1, -1))[0] for x in features])
    binary_predictions = (predictions > 0.5).astype(float)
    
    # Per-operation accuracy
    operation_names = classifier.operations
    print("\nPer-operation accuracy:")
    for i, op_name in enumerate(operation_names):
        accuracy = np.mean(binary_predictions[:, i] == labels[:, i])
        print(f"  {op_name}: {accuracy:.3f}")
    
    # Overall accuracy
    overall_accuracy = np.mean(binary_predictions == labels)
    print(f"\nOverall accuracy: {overall_accuracy:.3f}")
    
    return overall_accuracy


def save_classifier(classifier: SimpleClassifier, output_path: str):
    """Save trained classifier to pickle file."""
    model_data = {
        'weights1': classifier.weights1,
        'bias1': classifier.bias1,
        'weights2': classifier.weights2,
        'bias2': classifier.bias2,
        'input_dim': classifier.input_dim,
        'hidden_dim': classifier.hidden_dim,
        'operations': classifier.operations,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Classifier saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train neural guidance classifier for ARC solver")
    parser.add_argument('--train_json', required=True, help='Path to training challenges JSON')
    parser.add_argument('--solutions_json', help='Path to training solutions JSON (optional)')
    parser.add_argument('--out', required=True, help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load training data
    print(f"Loading training data from {args.train_json}")
    tasks = load_training_data(args.train_json, args.solutions_json)
    print(f"Loaded {len(tasks)} training tasks")
    
    # Extract features and labels
    features, labels = extract_training_features_and_labels(tasks)
    
    if len(features) == 0:
        print("No valid training examples found!")
        return
    
    print(f"Extracted {len(features)} feature vectors with {features.shape[1]} features each")
    
    # Train classifier
    classifier = train_classifier(features, labels, args.epochs)
    
    # Evaluate performance
    print("\nEvaluating trained classifier:")
    evaluate_classifier(classifier, features, labels)
    
    # Save trained model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_classifier(classifier, args.out)
    
    print(f"\nTraining complete! Model saved to {args.out}")


if __name__ == "__main__":
    main()
