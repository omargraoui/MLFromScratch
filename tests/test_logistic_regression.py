"""Test stable probabilities, objective scaling, and binary classification behavior."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from ml_from_scratch.linear_model import LogisticRegression
from ml_from_scratch.optimization import GradientDescent
from ml_from_scratch.utils.validation import ConvergenceWarning, NotFittedError


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(150, 3))
    logits = features @ np.array([1.3, -0.8, 0.4]) + 0.3
    targets = (rng.random(150) < 1 / (1 + np.exp(-logits))).astype(int)
    return features, targets


def test_separable_data_probability_shapes_and_api():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.array([0, 0, 1, 1])
    model = LogisticRegression(learning_rate=0.5, l2=0.1, tol=1e-9)
    assert model.fit(features, targets) is model
    probabilities = model.predict_proba(features)
    assert probabilities.shape == (4, 2)
    assert model.predict(features).shape == (4,)
    assert model.coef_.shape == (1,)
    assert model.n_features_in_ == 1
    assert isinstance(model.intercept_, float)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    np.testing.assert_array_equal(model.predict(features), targets)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert model.score(features, targets) == 1.0
    assert model.converged_
    assert len(model.loss_history_) == model.n_iter_ + 1
    assert model.loss_history_[0] == pytest.approx(np.log(2.0))
    assert np.all(np.diff(model.loss_history_) <= 1e-14)


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("l2", [0.0, 0.2])
def test_probabilities_and_coefficients_match_sklearn(binary_data, fit_intercept, l2):
    features, targets = binary_data
    model = LogisticRegression(
        learning_rate=0.5, fit_intercept=fit_intercept, l2=l2, tol=1e-10
    ).fit(features, targets)
    # sklearn scales its penalty against the SUM of sample losses.
    reference = SklearnLogisticRegression(
        C=1.0 / (features.shape[0] * l2) if l2 else 1e12,
        fit_intercept=fit_intercept,
        solver="lbfgs",
        tol=1e-10,
        max_iter=2000,
    ).fit(features, targets)
    np.testing.assert_allclose(model.coef_, reference.coef_[0], atol=2e-6)
    expected_intercept = float(reference.intercept_[0]) if fit_intercept else 0.0
    assert model.intercept_ == pytest.approx(expected_intercept, abs=2e-6)
    np.testing.assert_allclose(
        model.predict_proba(features), reference.predict_proba(features), atol=1e-6
    )
    np.testing.assert_array_equal(model.predict(features), reference.predict(features))


def test_l2_shrinks_coefficients_and_loss_matches_documented_objective(binary_data):
    features, targets = binary_data
    unregularized = LogisticRegression(learning_rate=0.5, tol=1e-9).fit(features, targets)
    regularized = LogisticRegression(learning_rate=0.5, l2=0.5, tol=1e-9).fit(features, targets)
    assert np.linalg.norm(regularized.coef_) < np.linalg.norm(unregularized.coef_)
    probabilities = regularized.predict_proba(features)[:, 1]
    cross_entropy = -np.mean(
        targets * np.log(probabilities) + (1 - targets) * np.log1p(-probabilities)
    )
    expected = cross_entropy + 0.25 * np.sum(regularized.coef_**2)
    assert regularized.loss_history_[-1] == pytest.approx(expected, abs=1e-14)


def test_regularization_never_penalizes_intercept():
    features = np.zeros((20, 1))
    targets = np.array([0] * 5 + [1] * 15)
    model = LogisticRegression(learning_rate=0.5, l2=100.0, tol=1e-10).fit(features, targets)
    assert model.intercept_ == pytest.approx(np.log(3.0), abs=1e-8)
    np.testing.assert_array_equal(model.coef_, [0.0])
    np.testing.assert_allclose(model.predict_proba(features)[:, 1], 0.75, atol=1e-9)


def test_determinism_and_refit(binary_data):
    features, targets = binary_data
    model = LogisticRegression(learning_rate=0.5, tol=1e-8).fit(features, targets)
    previous = model.coef_.copy()
    history = model.loss_history_.copy()
    model.fit(features, targets)
    np.testing.assert_array_equal(model.coef_, previous)
    assert model.loss_history_ == history


def test_configurable_threshold_including_ties_and_endpoints(binary_data):
    features, targets = binary_data
    model = LogisticRegression(learning_rate=0.5, tol=1e-8).fit(features, targets)
    probabilities = model.predict_proba(features)[:, 1]
    model.threshold = 0.7
    np.testing.assert_array_equal(model.predict(features), probabilities >= 0.7)
    model.threshold = 0.0
    np.testing.assert_array_equal(model.predict(features), np.ones(len(targets)))
    model.threshold = 1.0
    np.testing.assert_array_equal(model.predict(features), np.zeros(len(targets)))
    model.threshold = float(probabilities[0])
    assert model.predict(features[:1])[0] == 1
    model.threshold = -0.1
    with pytest.raises(ValueError, match="threshold"):
        model.predict(features)


def test_large_logits_remain_finite_during_training_and_prediction():
    features = np.array([[-1000.0], [-500.0], [500.0], [1000.0]])
    model = LogisticRegression(learning_rate=1.0, tol=1e-12).fit(features, [0, 0, 1, 1])
    assert np.all(np.isfinite(model.loss_history_))
    probabilities = model.predict_proba([[-1e6], [0.0], [1e6]])
    np.testing.assert_allclose(probabilities, [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])


@pytest.mark.parametrize("magnitude", [0, 1, 40, 100, 700, 1000])
def test_logistic_gradient_matches_high_precision_at_extreme_logits(monkeypatch, magnitude):
    original_minimize = GradientDescent.minimize

    def checked_minimize(optimizer, objective, initial_params):
        # Identity features isolate each sample's derivative, including both
        # labels at both logit signs and confidently incorrect predictions.
        params = np.array([-magnitude, -magnitude, magnitude, magnitude], dtype=float)
        loss, gradient = objective(params)
        with localcontext() as context:
            context.prec = 80
            tail = Decimal(1) / (1 + Decimal(magnitude).exp())
            expected = np.array([float(value / 4) for value in (tail, tail - 1, 1 - tail, -tail)])
        assert np.isfinite(loss)
        # An absolute tolerance would silently accept zero for the small tails.
        np.testing.assert_allclose(gradient, expected, rtol=1e-14, atol=0)
        return original_minimize(optimizer, objective, initial_params)

    monkeypatch.setattr(GradientDescent, "minimize", checked_minimize)
    # No optimization is needed after the objective probe: the initial norm is 0.25.
    LogisticRegression(fit_intercept=False, l2=0.0, tol=0.3).fit(np.eye(4), [0, 1, 0, 1])


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("tol", [1e-20, 1e-16])
def test_saturated_gradient_stopping_is_symmetric_under_label_inversion(fit_intercept, tol):
    features = np.array([[1.0], [-2.0]])
    targets = np.array([1, 0])
    models = []
    for labels in (targets, 1 - targets):
        model = LogisticRegression(
            learning_rate=160 / 3, max_iter=1, tol=tol, fit_intercept=fit_intercept
        )
        # The first update produces logits +/-40 and -/+80. The true gradient
        # norm is between these tolerances, even when sigmoid(40) rounds to 1.
        if tol == 1e-20:
            with pytest.warns(ConvergenceWarning, match="max_iter"):
                model.fit(features, labels)
            assert not model.converged_
        else:
            model.fit(features, labels)
            assert model.converged_
        assert model.n_iter_ == 1
        models.append(model)

    np.testing.assert_array_equal(models[0].coef_, [40.0])
    np.testing.assert_array_equal(models[1].coef_, -models[0].coef_)
    assert models[1].intercept_ == -models[0].intercept_
    np.testing.assert_array_equal(models[0].loss_history_, models[1].loss_history_)


def test_iteration_limit_is_explicit(binary_data):
    features, targets = binary_data
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        model = LogisticRegression(max_iter=1, tol=0.0).fit(features, targets)
    assert model.n_iter_ == 1
    assert not model.converged_
    assert len(model.loss_history_) == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"max_iter": -1},
        {"tol": -1.0},
        {"fit_intercept": "yes"},
        {"threshold": -0.1},
        {"threshold": 1.1},
        {"threshold": np.nan},
        {"l2": -1.0},
        {"l2": np.inf},
    ],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        LogisticRegression(**kwargs)


@pytest.mark.parametrize("targets", [[0, 0], [1, 1], [-1, 1], [0, 2], [0.0, 0.5], [0, np.nan]])
def test_invalid_labels(targets):
    with pytest.raises(ValueError):
        LogisticRegression().fit([[0.0], [1.0]], targets)


@pytest.mark.parametrize(
    ("features", "targets"),
    [([0, 1], [0, 1]), ([[0], [1]], [[0], [1]]), ([[0], [1]], [0]), ([[0], [np.inf]], [0, 1])],
)
def test_invalid_training_data(features, targets):
    with pytest.raises(ValueError):
        LogisticRegression().fit(features, targets)


def test_fitted_state_and_feature_validation(binary_data):
    model = LogisticRegression(learning_rate=0.5, tol=1e-8)
    with pytest.raises(NotFittedError):
        model.predict([[0.0]])
    with pytest.raises(NotFittedError):
        model.predict_proba([[0.0]])
    model.fit(*binary_data)
    with pytest.raises(ValueError):
        model.predict([[0.0]])


def test_diverging_regularized_training_is_reported(binary_data):
    with pytest.raises(FloatingPointError, match="learning_rate"):
        LogisticRegression(learning_rate=100.0, l2=1.0).fit(*binary_data)


@pytest.mark.parametrize("l2", [0.0, 0.3])
def test_logistic_gradient_matches_central_finite_differences(monkeypatch, l2):
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
    LogisticRegression(learning_rate=0.5, l2=l2).fit(
        [[-1.0, 0.5], [0.0, -1.0], [1.0, 0.5]] * 2, [0, 0, 1, 1, 1, 0]
    )


def test_zero_regularization_does_not_square_extreme_finite_coefficients():
    model = LogisticRegression(learning_rate=1e200).fit([[-1.0], [1.0]], [0, 1])
    assert np.all(np.isfinite(model.coef_))
    assert model.loss_history_[-1] == 0.0
    assert model.converged_


def test_failed_refit_retains_previous_fitted_model(binary_data):
    model = LogisticRegression(learning_rate=0.5).fit(*binary_data)
    before = model.predict_proba(binary_data[0])
    with pytest.raises(ValueError):
        model.fit([[0.0], [1.0]], [1, 1])
    np.testing.assert_array_equal(model.predict_proba(binary_data[0]), before)
