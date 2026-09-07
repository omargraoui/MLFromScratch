"""Independent analytic and scikit-learn checks for both least-squares solvers."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from ml_from_scratch.linear_model import LinearRegression
from ml_from_scratch.optimization import GradientDescent
from ml_from_scratch.utils.validation import ConvergenceWarning, NotFittedError


@pytest.mark.parametrize("solver", ["normal", "gd"])
def test_known_line_and_api(solver):
    features = np.linspace(-1.0, 1.0, 21).reshape(-1, 1)
    targets = 3.0 * features[:, 0] + 2.0
    model = LinearRegression(solver=solver, tol=1e-10)
    assert model.fit(features, targets) is model
    np.testing.assert_allclose(model.coef_, [3.0], atol=2e-10)
    assert model.intercept_ == pytest.approx(2.0, abs=1e-10)
    assert model.coef_.shape == (1,)
    assert isinstance(model.intercept_, float)
    assert model.n_features_in_ == 1
    assert model.predict([[0.0], [1.0]]).shape == (2,)
    assert model.score(features, targets) == pytest.approx(1.0)
    assert model.converged_
    assert len(model.loss_history_) == model.n_iter_ + 1
    assert model.loss_history_[-1] < 1e-18


@pytest.mark.parametrize("solver", ["normal", "gd"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_multifeature_agreement_with_sklearn(solver, fit_intercept):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 4))
    targets = features @ np.array([2.0, -3.0, 0.5, 1.0]) + rng.normal(0.0, 0.05, 80)
    if fit_intercept:
        targets += 1.7
    model = LinearRegression(solver=solver, fit_intercept=fit_intercept, tol=1e-10)
    reference = SklearnLinearRegression(fit_intercept=fit_intercept).fit(features, targets)
    model.fit(features, targets)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-9)
    assert model.intercept_ == pytest.approx(reference.intercept_, abs=1e-9)
    np.testing.assert_allclose(model.predict(features), reference.predict(features), atol=1e-9)
    if not fit_intercept:
        assert model.intercept_ == 0.0


def test_rank_deficient_features_use_a_valid_minimum_norm_solution():
    features = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    targets = np.array([4.0, 7.0, 10.0])
    model = LinearRegression().fit(features, targets)
    np.testing.assert_allclose(model.predict(features), targets)
    np.testing.assert_allclose(model.coef_, [0.6, 1.2])
    assert model.intercept_ == pytest.approx(1.0)
    assert model.n_iter_ == 0


def test_centered_solver_is_stable_with_large_feature_offset():
    features = (1e10 + np.arange(20.0)).reshape(-1, 1)
    targets = 2.5 * (features[:, 0] - 1e10) + 7.0
    model = LinearRegression().fit(features, targets)
    np.testing.assert_allclose(model.predict(features), targets, atol=1e-5)
    np.testing.assert_allclose(model.coef_, [2.5], atol=1e-12)


def test_one_sample_with_intercept_predicts_its_target():
    model = LinearRegression().fit([[8.0, 4.0]], [3.0])
    np.testing.assert_array_equal(model.coef_, [0.0, 0.0])
    assert model.intercept_ == 3.0


def test_gd_loss_decreases_and_iteration_warning_is_visible():
    with pytest.warns(ConvergenceWarning):
        model = LinearRegression(solver="gd", max_iter=2, tol=0.0).fit(
            [[-1.0], [0.0], [1.0]], [-1.0, 1.0, 3.0]
        )
    assert not model.converged_
    assert model.n_iter_ == 2
    assert np.all(np.diff(model.loss_history_) < 0.0)


def test_refit_updates_feature_count_and_coefficients():
    model = LinearRegression().fit([[0.0], [1.0]], [1.0, 3.0])
    model.fit([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [2.0, 3.0, 1.0])
    assert model.n_features_in_ == 2
    np.testing.assert_allclose(model.coef_, [1.0, 2.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": "inverse"},
        {"fit_intercept": 1},
        {"learning_rate": 0},
        {"max_iter": 0},
        {"tol": -1.0},
    ],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        LinearRegression(**kwargs)


@pytest.mark.parametrize(
    ("features", "targets"),
    [
        ([1, 2], [1, 2]),
        ([[1], [2]], [[1], [2]]),
        ([[1], [2]], [1]),
        ([[np.inf], [2]], [1, 2]),
        ([[1], [2]], [1, np.nan]),
        ([["a"], ["b"]], [1, 2]),
        ([], []),
    ],
)
def test_invalid_training_data(features, targets):
    with pytest.raises(ValueError):
        LinearRegression().fit(features, targets)


def test_prediction_requires_fit_and_matching_dimensions():
    model = LinearRegression()
    with pytest.raises(NotFittedError):
        model.predict([[1.0]])
    model.fit([[0.0], [1.0]], [1.0, 3.0])
    with pytest.raises(ValueError):
        model.predict([[1.0, 2.0]])
    with pytest.raises(ValueError):
        model.predict([[np.nan]])


def test_gd_reports_divergence():
    with pytest.raises(FloatingPointError, match="learning_rate"):
        LinearRegression(solver="gd", learning_rate=10.0).fit([[1.0], [2.0]], [2.0, 4.0])


def test_solver_assignment_is_validated_on_refit():
    model = LinearRegression()
    model.solver = "invalid"
    with pytest.raises(ValueError, match="solver"):
        model.fit([[1.0]], [1.0])


def test_linear_gradient_matches_central_finite_differences(monkeypatch):
    original_minimize = GradientDescent.minimize

    def checked_minimize(optimizer, objective, initial_params):
        params = np.array([0.4, -0.8, 0.3])
        _, gradient = objective(params)
        step = 1e-6
        numerical = np.empty_like(params)
        for index in range(params.size):
            direction = np.zeros_like(params)
            direction[index] = step
            numerical[index] = (
                objective(params + direction)[0] - objective(params - direction)[0]
            ) / (2.0 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-9)
        return original_minimize(optimizer, objective, initial_params)

    monkeypatch.setattr(GradientDescent, "minimize", checked_minimize)
    LinearRegression(solver="gd").fit([[-1.0, 0.5], [0.0, -1.0], [1.0, 0.5]], [0.4, -0.8, 0.3])


def test_prediction_overflow_is_explicit_and_failed_refit_retains_model():
    model = LinearRegression().fit([[-1.0], [1.0]], [-2.0, 2.0])
    with pytest.raises(FloatingPointError):
        model.predict([[1e308]])
    with pytest.raises(ValueError):
        model.fit([[np.nan]], [0.0])
    np.testing.assert_allclose(model.predict([[1.0]]), [2.0])
