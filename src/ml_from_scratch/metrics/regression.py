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
    """Return the root mean squared residual, in target units.

    Scale residuals by a power of two before squaring, since MSE can overflow
    or underflow even when RMSE is representable. A result exceeding the
    float64 range raises ``FloatingPointError``.
    """
    truth, prediction = _paired_vectors(y_true, y_pred)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            correction = 0
            try:
                # Subtract first: large matching values must not erase tiny
                # residuals elsewhere by determining the normalization scale.
                residual = truth - prediction
            except FloatingPointError:
                # Opposite finite extremes can have an unrepresentable
                # difference, yet their RMS over enough samples can be finite.
                residual = truth * 0.5 - prediction * 0.5
                correction = 1
            _, exponent = np.frexp(np.max(np.abs(residual)))
            scaled = np.ldexp(residual, -int(exponent))
            # Squared scaled values are < 1. Only negligible contributions or
            # the final subnormal rounding may underflow; neither is an error.
            result = np.ldexp(np.sqrt(np.mean(scaled**2)), int(exponent) + correction)
    except FloatingPointError as error:
        raise FloatingPointError("RMSE is outside the float64 range.") from error
    return float(result)


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean absolute residual, in target units."""
    truth, prediction = _paired_vectors(y_true, y_pred)
    return float(np.mean(np.abs(truth - prediction)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return 1 - residual sum of squares / total sum of squares.

    Requires at least two samples. For constant targets, return 1 for perfect
    predictions and 0 otherwise (the finite convention used by scikit-learn).
    Power-of-two scaling avoids overflow/underflow from squaring the original
    units; translated centering preserves variation around large offsets.
    A score outside the float64 range raises ``FloatingPointError``.
    """
    truth, prediction = _paired_vectors(y_true, y_pred)
    if truth.size < 2:
        raise ValueError("R squared requires at least two samples.")
    # Determine degeneracy from the data, not rounded means or squared errors.
    if np.all(truth == truth[0]):
        return float(np.array_equal(truth, prediction))

    largest = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))))
    _, exponent = np.frexp(largest)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            # ldexp also handles subnormal inputs and a largest value near
            # float64.max, without constructing 2**exponent or its reciprocal.
            scaled_truth = np.ldexp(truth, -int(exponent))
            scaled_prediction = np.ldexp(prediction, -int(exponent))
            translated = scaled_truth - scaled_truth[0]
            centered = translated - np.mean(translated)
            residual = np.sum((scaled_truth - scaled_prediction) ** 2)
            total = np.sum(centered**2)
            # A vanishing denominator here is numerical, not a constant target.
            score = float(1.0 - residual / total)
    except FloatingPointError as error:
        raise FloatingPointError("R squared is outside the float64 range.") from error
    return score
