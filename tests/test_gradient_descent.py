"""Check optimizer mathematics, stopping, and informative numerical failures."""

import numpy as np
import pytest

from ml_from_scratch.optimization import GradientDescent
from ml_from_scratch.utils.validation import ConvergenceWarning


def quadratic(params):
    residual = params - np.array([2.0, -3.0])
    return float(residual @ residual), 2.0 * residual


def test_convex_minimum_history_and_original_input_preserved():
    initial = np.array([8.0, 7.0])
    optimizer = GradientDescent(learning_rate=0.1, tol=1e-9)
    assert optimizer.minimize(quadratic, initial) is optimizer
    np.testing.assert_allclose(optimizer.params_, [2.0, -3.0], atol=1e-9)
    np.testing.assert_array_equal(initial, [8.0, 7.0])
    assert optimizer.converged_
    assert 0 < optimizer.n_iter_ < 1000
    assert len(optimizer.loss_history_) == optimizer.n_iter_ + 1
    assert optimizer.loss_history_[0] == 136.0
    assert np.all(np.diff(optimizer.loss_history_) <= 0.0)
    assert np.linalg.norm(quadratic(optimizer.params_)[1]) <= 1e-9


def test_optimal_initial_parameters_need_no_updates():
    optimizer = GradientDescent().minimize(quadratic, [2.0, -3.0])
    assert optimizer.n_iter_ == 0
    assert optimizer.converged_
    assert optimizer.loss_history_ == [0.0]


def test_convergence_on_last_permitted_update_is_not_a_warning():
    optimizer = GradientDescent(learning_rate=0.5, max_iter=1, tol=0.0)
    optimizer.minimize(quadratic, [0.0, 0.0])
    assert optimizer.n_iter_ == 1
    assert optimizer.converged_


def test_iteration_limit_retains_best_available_state():
    optimizer = GradientDescent(max_iter=1, tol=0.0)
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        optimizer.minimize(quadratic, [0.0, 0.0])
    assert not optimizer.converged_
    assert optimizer.n_iter_ == 1
    np.testing.assert_allclose(optimizer.params_, [0.4, -0.6])
    assert len(optimizer.loss_history_) == 2


def test_deterministic_refit_replaces_history():
    optimizer = GradientDescent(tol=1e-8)
    optimizer.minimize(quadratic, [0.0, 0.0])
    previous_params = optimizer.params_.copy()
    previous_history = optimizer.loss_history_.copy()
    optimizer.minimize(quadratic, [0.0, 0.0])
    np.testing.assert_array_equal(optimizer.params_, previous_params)
    assert optimizer.loss_history_ == previous_history
    optimizer.minimize(quadratic, [2.0, -3.0])
    assert optimizer.loss_history_ == [0.0]


def test_tolerance_controls_stopping():
    loose = GradientDescent(tol=1e-2).minimize(quadratic, [0.0, 0.0])
    tight = GradientDescent(tol=1e-8).minimize(quadratic, [0.0, 0.0])
    assert tight.n_iter_ > loose.n_iter_
    assert tight.loss_history_[-1] < loose.loss_history_[-1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"learning_rate": -1.0},
        {"learning_rate": np.inf},
        {"learning_rate": True},
        {"max_iter": 0},
        {"max_iter": 1.5},
        {"max_iter": True},
        {"tol": -1.0},
        {"tol": np.nan},
    ],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        GradientDescent(**kwargs)


@pytest.mark.parametrize("params", [[], [[1.0]], [np.nan], [np.inf], [1j], ["1"], [True]])
def test_invalid_initial_parameters(params):
    with pytest.raises(ValueError, match="initial_params"):
        GradientDescent().minimize(quadratic, params)


def test_non_callable_objective():
    with pytest.raises(TypeError, match="callable"):
        GradientDescent().minimize(None, [1.0])


@pytest.mark.parametrize(
    ("objective", "message"),
    [
        (lambda p: ([1.0], p), "scalar loss"),
        (lambda p: (1j, p), "scalar loss"),
        (lambda p: (1.0, p.astype(complex)), "real-valued"),
        (lambda p: (1.0, np.zeros((1, 1))), "same shape"),
    ],
)
def test_invalid_objective_contract(objective, message):
    with pytest.raises(ValueError, match=message):
        GradientDescent().minimize(objective, [1.0])


@pytest.mark.parametrize(
    "objective",
    [lambda p: (np.nan, p), lambda p: (0.0, p * np.inf)],
)
def test_nonfinite_objective_rejected(objective):
    with pytest.raises(FloatingPointError, match="non-finite"):
        GradientDescent().minimize(objective, [1.0])


def test_objective_arithmetic_overflow_has_actionable_message():
    def objective(params):
        return float(np.sum(params**2)), 2 * params

    with pytest.raises(FloatingPointError, match="scale features"):
        GradientDescent().minimize(objective, [1e200])


def test_nonfinite_update_and_large_gradient_norm_are_handled():
    def objective(params):
        return 0.0, np.full_like(params, -1e308)

    with pytest.raises(FloatingPointError, match="non-finite parameters"):
        GradientDescent(learning_rate=10).minimize(objective, [1.0])


def test_diverging_step_is_rejected_without_recording_it():
    optimizer = GradientDescent(learning_rate=2.0)
    with pytest.raises(FloatingPointError, match="increased the objective"):
        optimizer.minimize(quadratic, [0.0, 0.0])
    assert optimizer.n_iter_ == 0
    assert optimizer.loss_history_ == [13.0]
    np.testing.assert_array_equal(optimizer.params_, [0.0, 0.0])


@pytest.mark.parametrize("initial", [1.0, 1e-9, 1e-100, 1e-160])
@pytest.mark.parametrize("offset_factor", [0.0, -2.0])
def test_diverging_quadratic_is_rejected_at_small_loss_scales(initial, offset_factor):
    # J(theta) = theta**2 + c has curvature 2 regardless of scale or offset.
    # alpha=2 maps theta to -3*theta, increasing its squared error ninefold.
    offset = offset_factor * initial**2

    def objective(params):
        return float(params @ params) + offset, 2.0 * params

    parameters = np.array([initial])
    initial_loss = objective(parameters)[0]
    optimizer = GradientDescent(learning_rate=2.0, max_iter=1, tol=0.0)
    with pytest.raises(FloatingPointError, match="increased the objective"):
        optimizer.minimize(objective, parameters)
    assert optimizer.n_iter_ == 0
    assert not optimizer.converged_
    assert optimizer.loss_history_ == [initial_loss]
    np.testing.assert_array_equal(optimizer.params_, parameters)
    np.testing.assert_array_equal(parameters, [initial])


@pytest.mark.parametrize("initial", [1e-100, 1.0, 1e100])
def test_stable_quadratic_step_reaches_analytic_minimum_across_scales(initial):
    optimizer = GradientDescent(learning_rate=0.5, max_iter=1, tol=0.0)
    optimizer.minimize(lambda p: (float(p @ p), 2.0 * p), [initial])
    np.testing.assert_array_equal(optimizer.params_, [0.0])
    assert optimizer.loss_history_ == [initial**2, 0.0]
    assert optimizer.n_iter_ == 1
    assert optimizer.converged_


@pytest.mark.parametrize("initial", [1e-100, 1.0, 1e100])
def test_roundoff_sized_increase_is_allowed_across_loss_scales(initial):
    def objective(params):
        loss = float(params @ params)
        # alpha=1 changes only the sign: the exact quadratic loss is unchanged.
        # Simulate one ULP of objective-evaluation error at the candidate.
        if params[0] < 0:
            loss = float(np.nextafter(loss, np.inf))
        return loss, 2.0 * params

    optimizer = GradientDescent(learning_rate=1.0, max_iter=1, tol=0.0)
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        optimizer.minimize(objective, [initial])
    np.testing.assert_array_equal(optimizer.params_, [-initial])
    assert optimizer.loss_history_[1] > optimizer.loss_history_[0]
    assert optimizer.n_iter_ == 1
    assert not optimizer.converged_
