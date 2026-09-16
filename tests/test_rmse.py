"""RMSE range checks using exact squared errors and a high-precision square root."""

from decimal import Decimal, localcontext
from fractions import Fraction
from math import isqrt

import numpy as np
import pytest
from sklearn.metrics import root_mean_squared_error as reference_rmse

from ml_from_scratch import LinearRegression
from ml_from_scratch.metrics import mean_squared_error, root_mean_squared_error


def exact_rmse(truth, prediction):
    # Start with the actual binary inputs, not rounded decimal representations.
    squared_error = sum(
        (Fraction(float(y)) - Fraction(float(p))) ** 2
        for y, p in zip(truth, prediction, strict=True)
    ) / len(truth)
    numerator, denominator = isqrt(squared_error.numerator), isqrt(squared_error.denominator)
    if numerator**2 == squared_error.numerator and denominator**2 == squared_error.denominator:
        # Exact roots include float64 rounding ties; Decimal rounding before
        # conversion could put such a tie on the wrong side of the midpoint.
        return float(Fraction(numerator, denominator))
    with localcontext() as context:
        context.prec = 100
        return float((Decimal(squared_error.numerator) / Decimal(squared_error.denominator)).sqrt())


@pytest.mark.parametrize("scale", [1e-200, 1e-160, 1.0, -1.0, 1e154, 1e200, -1e200])
def test_rmse_preserves_target_units_across_scales(scale):
    truth = scale * np.array([0.0, 2.0, -3.0, 4.0])
    prediction = scale * np.array([1.0, 0.0, -2.0, 4.0])
    with np.errstate(all="raise"):
        actual = root_mean_squared_error(truth, prediction)
    assert actual == pytest.approx(exact_rmse(truth, prediction), rel=2e-15, abs=0)


@pytest.mark.parametrize(
    "error",
    [
        np.finfo(float).smallest_subnormal,
        np.finfo(float).tiny,
        1e-160,
        1e-200,
        1e200,
        np.finfo(float).max,
    ],
)
def test_rmse_single_residual_equals_its_absolute_value(error):
    with np.errstate(all="raise"):
        assert root_mean_squared_error([-error], [0.0]) == error


def test_rmse_does_not_overflow_when_summing_finite_squares():
    with np.errstate(all="raise"):
        assert root_mean_squared_error([1e154, 1e154], [0.0, 0.0]) == 1e154


@pytest.mark.parametrize("offset", [0.0, 1e16, -1e16])
def test_rmse_preserves_residuals_at_large_offsets(offset):
    truth = offset + np.array([0.0, 2.0, 0.0, 2.0])
    prediction = np.full(4, offset)
    with np.errstate(all="raise"):
        actual = root_mean_squared_error(truth, prediction)
    assert actual == pytest.approx(np.sqrt(2.0), rel=2e-15, abs=0)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_rmse_preserves_residual_at_float64_limit(sign):
    truth = [sign * np.finfo(float).max]
    prediction = [np.nextafter(truth[0], 0.0)]
    with np.errstate(all="raise"):
        assert root_mean_squared_error(truth, prediction) == exact_rmse(truth, prediction)


@pytest.mark.parametrize("sample_count", [4, 9])
def test_rmse_can_be_finite_when_a_residual_exceeds_float64(sample_count):
    truth = np.zeros(sample_count)
    truth[0] = np.finfo(float).max
    prediction = -truth
    original_truth, original_prediction = truth.copy(), prediction.copy()
    truth.flags.writeable = prediction.flags.writeable = False
    with np.errstate(all="raise"):
        actual = root_mean_squared_error(truth, prediction)
        assert root_mean_squared_error(prediction, truth) == actual
    assert actual == pytest.approx(exact_rmse(truth, prediction), rel=2e-15, abs=0)
    np.testing.assert_array_equal(truth, original_truth)
    np.testing.assert_array_equal(prediction, original_prediction)


def test_rmse_does_not_erase_tiny_residual_beside_large_matching_values():
    maximum = np.finfo(float).max
    truth, prediction = [maximum, 1e-200], [maximum, 0.0]
    with np.errstate(all="raise"):
        actual = root_mean_squared_error(truth, prediction)
    assert actual == pytest.approx(exact_rmse(truth, prediction), rel=2e-15, abs=0)


@pytest.mark.parametrize(
    "truth",
    [
        [1e200, 1e-200],
        [1.0, np.finfo(float).smallest_subnormal],
        [np.finfo(float).tiny, np.finfo(float).smallest_subnormal],
    ],
)
def test_rmse_handles_widely_differing_residual_magnitudes(truth):
    prediction = np.zeros(len(truth))
    with np.errstate(all="raise"):
        actual = root_mean_squared_error(truth, prediction)
    assert actual == pytest.approx(exact_rmse(truth, prediction), rel=2e-15, abs=0)


@pytest.mark.parametrize(
    ("multiples", "expected_units"),
    [([1, 0, 0, 0, 0], 0), ([1, 1, 0], 1), ([2, 1], 2), ([1, 0, 0, 0], 0), ([3, 0, 0, 0], 2)],
)
def test_rmse_rounds_subnormal_results_only_at_the_final_scale(multiples, expected_units):
    smallest = np.finfo(float).smallest_subnormal
    truth = np.array(multiples) * smallest
    prediction = np.zeros(len(truth))
    expected = expected_units * smallest
    assert exact_rmse(truth, prediction) == expected
    with np.errstate(all="raise"):
        assert root_mean_squared_error(truth, prediction) == expected


def test_rmse_identical_predictions_are_zero():
    truth = [0.0, -0.0, 0.1, 1e-200, np.finfo(float).max, np.finfo(float).smallest_subnormal]
    with np.errstate(all="raise"):
        assert root_mean_squared_error(truth, truth) == 0.0


@pytest.mark.parametrize("scale", [1e-200, 1.0, np.finfo(float).max])
def test_rmse_is_symmetric_and_does_not_mutate_readonly_strided_inputs(scale):
    truth = np.array([scale, 0.0, -scale, 0.0])[::2]
    prediction = np.array([0.0, scale, 0.0, -scale])[::2]
    original_truth, original_prediction = truth.copy(), prediction.copy()
    truth.flags.writeable = prediction.flags.writeable = False
    with np.errstate(all="raise"):
        assert root_mean_squared_error(truth, prediction) == scale
        assert root_mean_squared_error(prediction, truth) == scale
    np.testing.assert_array_equal(truth, original_truth)
    np.testing.assert_array_equal(prediction, original_prediction)


def test_rmse_matches_numpy_sklearn_and_mse_on_ordinary_data():
    rng = np.random.default_rng(42)
    truth = rng.normal(size=101)
    for noise in (0.0, 0.1, 1.0, 10.0):
        prediction = truth + rng.normal(scale=noise, size=truth.size)
        actual = root_mean_squared_error(truth, prediction)
        assert isinstance(actual, float)
        for expected in (
            reference_rmse(truth, prediction),
            np.sqrt(np.mean((truth - prediction) ** 2)),
            np.sqrt(mean_squared_error(truth, prediction)),
            exact_rmse(truth, prediction),
        ):
            assert actual == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("solver", ["normal", "gd"])
@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_rmse_evaluates_linear_regression_predictions_at_extreme_scales(solver, scale):
    features = [[0.0], [1.0]]
    model = LinearRegression(solver=solver, fit_intercept=False).fit(features, [0.0, 0.0])
    truth, prediction = [0.0, scale], model.predict(features)
    with np.errstate(all="raise"):
        actual = root_mean_squared_error(truth, prediction)
    assert actual == pytest.approx(exact_rmse(truth, prediction), rel=2e-15, abs=0)


@pytest.mark.parametrize("error_mode", ["ignore", "raise"])
def test_rmse_reports_unrepresentable_result(error_mode):
    maximum = np.finfo(float).max
    with np.errstate(all=error_mode):
        with pytest.raises(FloatingPointError, match="RMSE.*float64"):
            root_mean_squared_error([maximum], [-maximum])


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [
        ([], []),
        ([0.0], [0.0, 1.0]),
        ([[0.0]], [[0.0]]),
        ([np.inf], [0.0]),
        ([0.0], [np.nan]),
        (["1"], [0.0]),
        ([1j], [0.0]),
    ],
)
def test_rmse_preserves_input_validation(truth, prediction):
    with pytest.raises(ValueError):
        root_mean_squared_error(truth, prediction)
