"""Scale and translation invariance with an exact rational R-squared oracle."""

from fractions import Fraction

import numpy as np
import pytest
from sklearn.metrics import r2_score as reference_r2_score

from ml_from_scratch import LinearRegression
from ml_from_scratch.metrics import r2_score


def exact_r2(truth, prediction):
    truth = [Fraction(float(value)) for value in truth]
    prediction = [Fraction(float(value)) for value in prediction]
    mean = sum(truth) / len(truth)
    residual = sum((y - p) ** 2 for y, p in zip(truth, prediction, strict=True))
    total = sum((y - mean) ** 2 for y in truth)
    return float(1 - residual / total) if total else float(residual == 0)


@pytest.mark.parametrize("scale", [1e-200, 1e-160, 1.0, -1.0, 1e154, 1e200, -1e200])
@pytest.mark.parametrize("prediction", [[0.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 2.0, 4.0]])
def test_r2_preserves_scores_across_target_scales(scale, prediction):
    truth = np.array([0.0, 2.0, 4.0]) * scale
    prediction = np.array(prediction) * scale
    with np.errstate(all="raise"):
        actual = r2_score(truth, prediction)
    assert actual == pytest.approx(exact_r2(truth, prediction), rel=2e-14, abs=2e-14)


@pytest.mark.parametrize("offset", [0.0, 1e16, -1e16])
def test_r2_centers_without_rounding_away_target_variation(offset):
    truth = np.array([0.0, 2.0, 0.0, 2.0]) + offset
    prediction = np.full(4, offset)
    assert r2_score(truth, prediction) == pytest.approx(-1.0, abs=1e-14)


@pytest.mark.parametrize("prediction_factor", [0.0, -1.0, 1.0])
def test_r2_handles_opposite_float64_limits(prediction_factor):
    truth = np.array([-np.finfo(float).max, np.finfo(float).max])
    prediction = prediction_factor * truth
    with np.errstate(all="raise"):
        actual = r2_score(truth, prediction)
    assert actual == pytest.approx(exact_r2(truth, prediction), abs=1e-14)


def test_r2_handles_smallest_subnormal_targets():
    smallest = np.finfo(float).smallest_subnormal
    truth = np.array([0.0, smallest])
    with np.errstate(all="raise"):
        assert r2_score(truth, [0.0, 0.0]) == -1.0


@pytest.mark.parametrize("value", [0.0, 0.1, 1e-200, 1e308, -1e308])
def test_r2_constant_targets_use_equality_before_arithmetic(value):
    truth = np.full(7, value)
    prediction = truth.copy()
    prediction[0] = np.nextafter(value, np.inf)
    with np.errstate(all="raise"):
        assert r2_score(truth, truth) == 1.0
        assert r2_score(truth, prediction) == 0.0


def test_r2_matches_sklearn_on_ordinary_data_without_mutating_inputs():
    rng = np.random.default_rng(42)
    for noise in (0.0, 0.1, 1.0, 10.0):
        truth = rng.normal(size=101)
        prediction = truth + rng.normal(scale=noise, size=101)
        original_truth, original_prediction = truth.copy(), prediction.copy()
        assert r2_score(truth, prediction) == pytest.approx(
            reference_r2_score(truth, prediction), rel=2e-14, abs=2e-14
        )
        np.testing.assert_array_equal(truth, original_truth)
        np.testing.assert_array_equal(prediction, original_prediction)


def test_linear_regression_score_does_not_report_false_perfect_fit():
    model = LinearRegression(fit_intercept=False).fit([[0.0], [1.0]], [0.0, 0.0])
    assert model.score([[0.0], [1.0]], [0.0, 2e-200]) == -1.0


@pytest.mark.parametrize("truth", [[0.0, 1e-200], [0.0, 1.0]])
def test_r2_reports_unrepresentable_score(truth):
    with pytest.raises(FloatingPointError, match="R squared"):
        r2_score(truth, [1e200, -1e200])
