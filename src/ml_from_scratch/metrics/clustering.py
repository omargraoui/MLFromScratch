"""Clustering scores that do not depend on the names of cluster IDs."""

import numpy as np
from numpy.typing import ArrayLike

from ml_from_scratch.metrics.regression import _paired_vectors
from ml_from_scratch.utils.validation import check_array, check_n_features, check_vector


def inertia_score(X: ArrayLike, centers: ArrayLike, labels: ArrayLike) -> float:
    """Sum squared Euclidean distances to assigned centers (smaller is better)."""
    features = check_array(X)
    centroids = check_array(centers)
    check_n_features(centroids, features.shape[1])
    assignments = check_vector(labels, name="labels")
    if assignments.size != features.shape[0]:
        raise ValueError("labels and X must have the same number of samples.")
    if np.any(assignments != np.floor(assignments)) or np.any(
        (assignments < 0) | (assignments >= centroids.shape[0])
    ):
        raise ValueError("labels must be integer indices into centers.")
    residuals = features - centroids[assignments.astype(np.intp)]
    return float(np.sum(residuals * residuals))


def adjusted_rand_score(labels_true: ArrayLike, labels_pred: ArrayLike) -> float:
    """Compute the chance-adjusted pair agreement of two numeric partitions.

    A score of 1 means identical partitions up to label permutation. Random
    agreement is near zero; some partitions score below zero. Single-sample
    and identical degenerate partitions score 1. Inputs must be nonempty.
    """
    truth, prediction = _paired_vectors(labels_true, labels_pred)
    _, true_codes = np.unique(truth, return_inverse=True)
    _, pred_codes = np.unique(prediction, return_inverse=True)
    contingency = np.zeros((true_codes.max() + 1, pred_codes.max() + 1), dtype=np.float64)
    np.add.at(contingency, (true_codes, pred_codes), 1)
    same_both = float(np.sum(contingency * (contingency - 1) / 2))
    row_counts = contingency.sum(axis=1)
    col_counts = contingency.sum(axis=0)
    same_true = float(np.sum(row_counts * (row_counts - 1) / 2))
    same_pred = float(np.sum(col_counts * (col_counts - 1) / 2))
    total_pairs = truth.size * (truth.size - 1) / 2
    if total_pairs == 0:
        return 1.0
    expected = same_true * same_pred / total_pairs
    maximum = (same_true + same_pred) / 2
    if maximum == expected:
        return 1.0
    return (same_both - expected) / (maximum - expected)
