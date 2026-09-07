"""Classification metrics with explicit binary-label and zero-division rules."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.metrics.regression import _paired_vectors


def _binary_pairs(
    y_true: ArrayLike, y_pred: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    truth, prediction = _paired_vectors(y_true, y_pred)
    if not np.all(np.isin(truth, [0, 1])) or not np.all(np.isin(prediction, [0, 1])):
        raise ValueError("Binary metrics require labels 0 and 1.")
    return truth, prediction


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return fraction of correct predictions; numeric multiclass labels are allowed."""
    truth, prediction = _paired_vectors(y_true, y_pred)
    return float(np.mean(truth == prediction))


def _confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int]:
    truth, prediction = _binary_pairs(y_true, y_pred)
    true_positive = int(np.sum((truth == 1) & (prediction == 1)))
    false_positive = int(np.sum((truth == 0) & (prediction == 1)))
    false_negative = int(np.sum((truth == 1) & (prediction == 0)))
    return true_positive, false_positive, false_negative


def precision_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return TP/(TP+FP), or zero when there are no positive predictions."""
    tp, fp, _ = _confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return TP/(TP+FN), or zero when there are no positive targets."""
    tp, _, fn = _confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return 2TP/(2TP+FP+FN), with undefined results set to zero."""
    tp, fp, fn = _confusion_counts(y_true, y_pred)
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def log_loss(y_true: ArrayLike, y_proba: ArrayLike) -> float:
    """Mean binary cross entropy for a 1D vector of positive-class probabilities.

    Probabilities must be in [0, 1]. Clip only at float64 machine precision to
    give finite losses for exactly zero/one predictions. The training objective
    uses logits directly and therefore does not need this clipping.
    """
    truth, probabilities = _paired_vectors(y_true, y_proba)
    if not np.all(np.isin(truth, [0, 1])):
        raise ValueError("Binary log loss requires labels 0 and 1.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in [0, 1].")
    eps = np.finfo(np.float64).eps
    probabilities = np.clip(probabilities, eps, 1.0 - eps)
    return -float(np.mean(truth * np.log(probabilities) + (1 - truth) * np.log1p(-probabilities)))
