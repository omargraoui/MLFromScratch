"""Reject invalid inputs at the boundary, without changing caller arrays."""

import numpy as np
import pytest

from ml_from_scratch.utils.numerical import sigmoid
from ml_from_scratch.utils.validation import (
    NotFittedError,
    check_array,
    check_is_fitted,
    check_n_features,
    check_vector,
    check_X_y,
    validate_positive_int,
    validate_real,
)


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1, 2],
        [[[1]]],
        np.empty((2, 0)),
        [[np.nan]],
        [[np.inf]],
        [["1"]],
        [[1j]],
        [[object()]],
        [[1], [2, 3]],
    ],
)
def test_invalid_feature_arrays(value):
    with pytest.raises(ValueError):
        check_array(value)


def test_numeric_input_shape_and_no_mutation():
    original = np.arange(12, dtype=np.int32).reshape(4, 3)
    saved = original.copy()
    validated, targets = check_X_y(original, [1, 2, 3, 4])
    assert validated.dtype == targets.dtype == np.float64
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_array_equal(validated, original)
    assert check_array([[True]]).item() == 1


@pytest.mark.parametrize("value", [[], [[1], [2]], [np.nan], ["a"], 1])
def test_invalid_target_vector(value):
    with pytest.raises(ValueError):
        check_vector(value)


def test_sample_and_feature_counts():
    with pytest.raises(ValueError, match="same number"):
        check_X_y([[1], [2]], [1])
    with pytest.raises(ValueError, match="at least 2"):
        check_array([[1]], min_samples=2)
    with pytest.raises(ValueError, match="expected 2"):
        check_n_features(np.ones((3, 1)), 2)
    check_n_features(np.ones((3, 2)), 2)


def test_fitted_state():
    class Estimator:
        coef_ = [1]

    estimator = Estimator()
    check_is_fitted(estimator, "coef_")
    check_is_fitted(estimator, ("coef_",))
    with pytest.raises(NotFittedError, match="Call fit"):
        check_is_fitted(estimator, ("coef_", "intercept_"))


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True, np.bool_(True), np.nan])
def test_invalid_positive_int(value):
    with pytest.raises(ValueError, match="positive integer"):
        validate_positive_int(value, "n")


def test_valid_hyperparameters():
    assert validate_positive_int(np.int64(2), "n") == 2
    assert validate_real(np.float64(0.5), "rate", maximum=1) == 0.5
    assert validate_real(0, "tol") == 0
    assert validate_real(1, "rate", strict=True) == 1


@pytest.mark.parametrize("value", [True, np.bool_(False), "1", 1j, np.nan, np.inf, -1])
def test_invalid_real(value):
    with pytest.raises(ValueError):
        validate_real(value, "rate")


def test_real_bounds():
    with pytest.raises(ValueError, match="greater than"):
        validate_real(0, "rate", strict=True)
    with pytest.raises(ValueError, match="at most"):
        validate_real(2, "threshold", maximum=1)


def test_sigmoid_extreme_values_and_symmetry():
    logits = np.array([-np.inf, -1000, -1, 0, 1, 1000, np.inf])
    with np.errstate(all="raise"):
        probabilities = sigmoid(logits)
    assert probabilities[0] == probabilities[1] == 0
    assert probabilities[-1] == probabilities[-2] == 1
    assert probabilities[3] == 0.5
    np.testing.assert_allclose(probabilities + sigmoid(-logits), 1)
    assert float(sigmoid(0)) == 0.5
