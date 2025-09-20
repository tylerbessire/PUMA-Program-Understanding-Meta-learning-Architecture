"""
Neural guidance training tool for ARC solver.

This script trains the neural guidance classifier that predicts which DSL operations
are likely relevant for solving a given ARC task. The classifier is trained on
features extracted from the training challenges and their known solutions.
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from arc_solver.grid import to_array
from arc_solver.features import extract_task_features
from arc_solver.neural.guidance import SimpleClassifier


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


def train_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-2,
    batch_size: int = 128,
) -> SimpleClassifier:
    """Train the neural guidance classifier with mini-batch gradient descent."""
    num_examples = len(features)
    if num_examples == 0:
        raise ValueError("no training examples provided")

    print(f"Training classifier on {num_examples} examples...")
    classifier = SimpleClassifier(input_dim=features.shape[1])

    features = features.astype(np.float32, copy=False)
    labels = labels.astype(np.float32, copy=False)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = np.arange(num_examples)
    print_every = max(1, epochs // 10)

    for epoch in range(epochs):
        np.random.shuffle(indices)
        total_loss = 0.0

        for start in range(0, num_examples, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = features[batch_idx]
            Y_batch = labels[batch_idx]

            hidden_pre = X_batch @ classifier.weights1 + classifier.bias1
            hidden = np.maximum(0, hidden_pre)
            logits = hidden @ classifier.weights2 + classifier.bias2
            probs = classifier._sigmoid(logits)
            probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)

            batch_loss = -np.mean(
                Y_batch * np.log(probs_clipped)
                + (1.0 - Y_batch) * np.log(1.0 - probs_clipped)
            )
            total_loss += batch_loss * X_batch.shape[0]

            grad_out = (probs - Y_batch) / X_batch.shape[0]
            grad_w2 = hidden.T @ grad_out
            grad_b2 = grad_out.sum(axis=0)

            grad_hidden = grad_out @ classifier.weights2.T
            grad_hidden[hidden_pre <= 0] = 0.0
            grad_w1 = X_batch.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)

            classifier.weights2 -= lr * grad_w2
            classifier.bias2 -= lr * grad_b2
            classifier.weights1 -= lr * grad_w1
            classifier.bias1 -= lr * grad_b1

        avg_loss = total_loss / num_examples
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    return classifier


def evaluate_classifier(classifier: SimpleClassifier, features: np.ndarray, labels: np.ndarray):
    """Evaluate classifier performance."""
    if len(features) == 0:
        raise ValueError("no features provided for evaluation")

    predictions = np.vstack([
        classifier.forward(x.reshape(1, -1)).ravel() for x in features
    ])
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

    tp = np.logical_and(binary_predictions == 1.0, labels == 1.0).sum()
    fp = np.logical_and(binary_predictions == 1.0, labels == 0.0).sum()
    fn = np.logical_and(binary_predictions == 0.0, labels == 1.0).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    micro_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    print(f"Micro-F1: {micro_f1:.3f} (precision={precision:.3f}, recall={recall:.3f})")

    return overall_accuracy


def save_classifier(classifier: SimpleClassifier, output_path: str):
    """Save trained classifier to JSON compatible with ``NeuralGuidance``."""

    model_data = {
        "input_dim": classifier.input_dim,
        "hidden_dim": classifier.hidden_dim,
        "weights1": classifier.weights1.tolist(),
        "bias1": classifier.bias1.tolist(),
        "weights2": classifier.weights2.tolist(),
        "bias2": classifier.bias2.tolist(),
        "operations": classifier.operations,
    }

    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f)
    os.replace(tmp_path, output_path)

    print(f"Classifier saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train neural guidance classifier for ARC solver")
    parser.add_argument('--train_json', required=True, help='Path to training challenges JSON')
    parser.add_argument('--solutions_json', help='Path to training solutions JSON (optional)')
    parser.add_argument('--out', required=True, help='Output path for trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate for optimisation')
    parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
    
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
    classifier = train_classifier(
        features,
        labels,
        epochs=args.epochs,
        lr=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    # Evaluate performance
    print("\nEvaluating trained classifier:")
    evaluate_classifier(classifier, features, labels)
    
    # Save trained model
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_classifier(classifier, args.out)
    
    print(f"\nTraining complete! Model saved to {args.out}")


if __name__ == "__main__":
    main()
