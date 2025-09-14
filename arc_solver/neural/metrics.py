from __future__ import annotations

"""Metrics utilities for neural guidance evaluation."""

# [S:ALG v1] metric=top_k_micro_f1 pass

import numpy as np


def top_k_micro_f1(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Compute micro-F1 at top-k for multi-label predictions.

    Parameters
    ----------
    probs : ndarray (n_samples, n_classes)
        Predicted probabilities for each class.
    labels : ndarray (n_samples, n_classes)
        Binary ground-truth labels.
    k : int
        Number of top predictions to consider per sample.

    Returns
    -------
    float
        Micro-averaged F1 score considering the top-k predictions per sample.
    """
    if probs.shape != labels.shape:
        raise ValueError("probs and labels must have the same shape")
    n_classes = probs.shape[1]
    if k <= 0 or k > n_classes:
        raise ValueError("k must be between 1 and number of classes")

    topk_indices = np.argsort(-probs, axis=1)[:, :k]
    pred = np.zeros_like(labels, dtype=bool)
    for i, idxs in enumerate(topk_indices):
        pred[i, idxs] = True

    labels = labels.astype(bool)
    tp = np.logical_and(pred, labels).sum()
    fp = np.logical_and(pred, ~labels).sum()
    fn = np.logical_and(~pred, labels).sum()
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
