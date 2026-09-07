"""Regression metrics for single-target predictions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.utils.validation import check_vector


def _paired_vectors(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    truth = check_vector(y_true, name="y_true")
    prediction = check_vector(y_pred, name="y_pred")
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return truth, prediction


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean squared residual, in squared target units."""
    truth, prediction = _paired_vectors(y_true, y_pred)
    return float(np.mean((truth - prediction) ** 2))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the square root of MSE, in target units."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute residual, in target units."""
    truth, prediction = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(truth - prediction)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return 1 - residual sum of squares / total sum of squares.

    Requires at least two samples. For constant targets, return 1 for perfect
    predictions and 0 otherwise (the finite convention used by scikit-learn).
    """
    truth, prediction = _paired_vectors(y_true, y_pred)
    if truth.size < 2:
        raise ValueError("R squared requires at least two samples.")
    residual = float(np.sum((truth - prediction) ** 2))
    total = float(np.sum((truth - truth.mean()) ** 2))
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return 1.0 - residual / total
