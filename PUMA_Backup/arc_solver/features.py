"""
Feature extraction for neural guidance in ARC tasks.

This module extracts meaningful features from ARC training pairs that can be used
to train classifiers and guide program search. Features are designed to capture
the types of transformations and patterns commonly seen in ARC tasks.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Any

from .grid import Array, histogram, bg_color, eq
from .objects import connected_components
from .canonical import canonicalize_D4


def extract_task_features(train_pairs: List[Tuple[Array, Array]]) -> Dict[str, Any]:
    """Extract a comprehensive feature vector from training pairs.

    These features capture task-level properties that can help predict which
    DSL operations are likely to be relevant for solving the task.
    """

    # Ensure arrays are integer typed for canonicalisation
    try:
        original_pairs = [
            (np.asarray(inp, dtype=int), np.asarray(out, dtype=int))
            for inp, out in train_pairs
        ]
        canonical_pairs = [
            (canonicalize_D4(inp), canonicalize_D4(out))
            for inp, out in original_pairs
        ]
    except Exception as exc:
        raise ValueError(f"invalid grid in train_pairs: {exc}") from exc

    features: Dict[str, Any] = {}

    # Basic grid statistics using original shapes
    input_shapes = [inp.shape for inp, _ in original_pairs]
    output_shapes = [out.shape for _, out in original_pairs]

    features.update({
        'num_train_pairs': len(original_pairs),
        'input_height_mean': np.mean([s[0] for s in input_shapes]),
        'input_width_mean': np.mean([s[1] for s in input_shapes]),
        'output_height_mean': np.mean([s[0] for s in output_shapes]),
        'output_width_mean': np.mean([s[1] for s in output_shapes]),
        'shape_preserved': all(inp.shape == out.shape for inp, out in original_pairs),
        'size_ratio_mean': np.mean([
            (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1])
            for inp, out in original_pairs
        ]),
    })

    # Color analysis on canonical pairs
    input_colors: List[int] = []
    output_colors: List[int] = []
    color_mappings: List[int] = []

    for inp, out in canonical_pairs:
        inp_hist = histogram(inp)
        out_hist = histogram(out)
        input_colors.append(len(inp_hist))
        output_colors.append(len(out_hist))

        # Try to detect color mappings
        if inp.shape == out.shape:
            mapping: Dict[int, int] = {}
            valid_mapping = True
            for i_val, o_val in zip(inp.flatten(), out.flatten()):
                if i_val in mapping and mapping[i_val] != o_val:
                    valid_mapping = False
                    break
                mapping[i_val] = o_val
            if valid_mapping:
                color_mappings.append(len(mapping))

    features.update({
        'input_colors_mean': np.mean(input_colors),
        'output_colors_mean': np.mean(output_colors),
        'background_color_consistent': len(set(bg_color(inp) for inp, _ in canonical_pairs)) == 1,
        'has_color_mapping': len(color_mappings) > 0,
        'color_mapping_size': np.mean(color_mappings) if color_mappings else 0,
    })

    # Object analysis on canonical pairs
    input_obj_counts: List[int] = []
    output_obj_counts: List[int] = []

    for inp, out in canonical_pairs:
        inp_objects = connected_components(inp)
        out_objects = connected_components(out)
        input_obj_counts.append(len(inp_objects))
        output_obj_counts.append(len(out_objects))

    features.update({
        'input_objects_mean': np.mean(input_obj_counts),
        'output_objects_mean': np.mean(output_obj_counts),
        'object_count_preserved': all(
            len(connected_components(inp)) == len(connected_components(out))
            for inp, out in canonical_pairs
        ),
    })

    # Transformation hints from original pairs
    features.update({
        'likely_rotation': _detect_rotation_patterns(original_pairs),
        'likely_reflection': _detect_reflection_patterns(original_pairs),
        'likely_translation': _detect_translation_patterns(original_pairs),
        'likely_recolor': _detect_recolor_patterns(original_pairs),
        'likely_crop': _detect_crop_patterns(original_pairs),
        'likely_pad': _detect_pad_patterns(original_pairs),
    })

    return features


def _detect_rotation_patterns(train_pairs: List[Tuple[Array, Array]]) -> float:
    """Detect if rotation transformations are likely."""
    if not train_pairs:
        return 0.0
    rotation_score = 0.0
    for inp, out in train_pairs:
        if inp.shape[0] == inp.shape[1] and out.shape[0] == out.shape[1]:
            # Check 90-degree rotations
            for k in [1, 2, 3]:
                if eq(np.rot90(inp, k), out):
                    rotation_score += 1.0
                    break
    return rotation_score / len(train_pairs)


def _detect_reflection_patterns(train_pairs: List[Tuple[Array, Array]]) -> float:
    """Detect if reflection transformations are likely."""
    if not train_pairs:
        return 0.0
    reflection_score = 0.0
    for inp, out in train_pairs:
        if inp.shape == out.shape:
            if eq(np.flip(inp, axis=0), out) or eq(np.flip(inp, axis=1), out):
                reflection_score += 1.0
            elif eq(inp.T, out):
                reflection_score += 1.0
    return reflection_score / len(train_pairs)


def _detect_translation_patterns(train_pairs: List[Tuple[Array, Array]]) -> float:
    """Detect if translation transformations are likely."""
    if not train_pairs or not all(inp.shape == out.shape for inp, out in train_pairs):
        return 0.0
    
    translation_score = 0.0
    for inp, out in train_pairs:
        # Simple check: if output has same content but shifted
        if not eq(inp, out):
            inp_objects = connected_components(inp)
            out_objects = connected_components(out)
            if len(inp_objects) == len(out_objects):
                translation_score += 0.5  # Partial evidence
    
    return translation_score / len(train_pairs)


def _detect_recolor_patterns(train_pairs: List[Tuple[Array, Array]]) -> float:
    """Detect if recoloring transformations are likely."""
    if not train_pairs:
        return 0.0
    recolor_score = 0.0
    for inp, out in train_pairs:
        if inp.shape == out.shape:
            # Check if this could be a pure recoloring
            mapping = {}
            valid_mapping = True
            for i_val, o_val in zip(inp.flatten(), out.flatten()):
                if i_val in mapping and mapping[i_val] != o_val:
                    valid_mapping = False
                    break
                mapping[i_val] = o_val
            if valid_mapping and len(mapping) > 1:
                recolor_score += 1.0
    return recolor_score / len(train_pairs)


def _detect_crop_patterns(train_pairs: List[Tuple[Array, Array]]) -> float:
    """Detect if cropping transformations are likely."""
    if not train_pairs:
        return 0.0
    crop_score = 0.0
    for inp, out in train_pairs:
        if (inp.shape[0] > out.shape[0] or inp.shape[1] > out.shape[1]):
            crop_score += 1.0
    return crop_score / len(train_pairs)


def _detect_pad_patterns(train_pairs: List[Tuple[Array, Array]]) -> float:
    """Detect if padding transformations are likely."""
    if not train_pairs:
        return 0.0
    pad_score = 0.0
    for inp, out in train_pairs:
        if (inp.shape[0] < out.shape[0] or inp.shape[1] < out.shape[1]):
            pad_score += 1.0
    return pad_score / len(train_pairs)


def compute_task_signature(train_pairs: List[Tuple[Array, Array]]) -> str:
    """Compute a compact signature for episodic retrieval."""
    features = extract_task_features(train_pairs)
    
    # Create a simple signature from key features
    signature_parts = [
        f"pairs:{features['num_train_pairs']}",
        f"shape:{int(features['shape_preserved'])}",
        f"colors:{int(features['input_colors_mean'])}-{int(features['output_colors_mean'])}",
        f"objs:{int(features['input_objects_mean'])}-{int(features['output_objects_mean'])}",
        f"ops:{_operation_hints(features)}",
    ]
    
    return "|".join(signature_parts)


def _operation_hints(features: Dict[str, Any]) -> str:
    """Generate operation hints from features."""
    hints = []
    threshold = 0.5
    
    if features['likely_rotation'] > threshold:
        hints.append('R')
    if features['likely_reflection'] > threshold:
        hints.append('F')
    if features['likely_translation'] > threshold:
        hints.append('T')
    if features['likely_recolor'] > threshold:
        hints.append('C')
    if features['likely_crop'] > threshold:
        hints.append('X')
    if features['likely_pad'] > threshold:
        hints.append('P')
    
    return "".join(hints) if hints else "U"  # U for unknown


def compute_numerical_features(train_pairs: List[Tuple[Array, Array]]) -> np.ndarray:
    """Convert task features to a numerical vector.

    This utility is primarily used by learning components that expect a fixed
    numeric representation.  The order of features is deterministic to ensure
    reproducibility across runs.

    Args:
        train_pairs: List of training input/output grid pairs.

    Returns:
        A 1-D numpy array of feature values. Boolean features are encoded as
        ``0.0`` or ``1.0``.
    """

    features = extract_task_features(train_pairs)

    numerical_keys = [
        'num_train_pairs',
        'input_height_mean',
        'input_width_mean',
        'output_height_mean',
        'output_width_mean',
        'shape_preserved',
        'size_ratio_mean',
        'input_colors_mean',
        'output_colors_mean',
        'background_color_consistent',
        'has_color_mapping',
        'color_mapping_size',
        'input_objects_mean',
        'output_objects_mean',
        'object_count_preserved',
        'likely_rotation',
        'likely_reflection',
        'likely_translation',
        'likely_recolor',
        'likely_crop',
        'likely_pad',
    ]

    values: List[float] = []
    for key in numerical_keys:
        val = features.get(key, 0)
        if isinstance(val, bool):
            values.append(1.0 if val else 0.0)
        else:
            try:
                values.append(float(val))
            except (TypeError, ValueError):  # pragma: no cover - defensive path
                values.append(0.0)

    return np.array(values, dtype=float)
