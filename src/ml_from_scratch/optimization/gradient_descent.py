"""Fixed-step gradient descent with explicit numerical failure reporting."""

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.utils.validation import (
    ConvergenceWarning,
    validate_positive_int,
    validate_real,
)

Objective = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


class GradientDescent:
    """Minimize a differentiable scalar objective using a fixed learning rate.

    The objective receives a one-dimensional parameter vector and returns
    ``(loss, gradient)``. Iterations stop when the Euclidean gradient norm is at
    most ``tol``. ``loss_history_`` includes the initial objective followed by
    each accepted update; ``n_iter_`` counts updates, not objective evaluations.

    A materially increasing loss or non-finite arithmetic raises
    ``FloatingPointError``. The allowed rounding error scales with the compared
    losses, without an absolute floor that could hide divergence near zero.
    This deliberately simple optimizer does not perform
    line search: use a smaller learning rate or rescale features in that case.
    Reaching ``max_iter`` emits ``ConvergenceWarning`` and retains the result.
    """

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000, tol: float = 1e-6) -> None:
        self.learning_rate = validate_real(learning_rate, "learning_rate", strict=True)
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = validate_real(tol, "tol")

    @staticmethod
    def _evaluate(
        objective: Objective, params: NDArray[np.float64]
    ) -> tuple[float, NDArray[np.float64]]:
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                loss, gradient = objective(params)
        except FloatingPointError as error:
            raise FloatingPointError(
                "Objective evaluation failed numerically; lower learning_rate or scale features."
            ) from error
        if np.ndim(loss) != 0 or np.iscomplexobj(loss):
            raise ValueError("The objective must return a real scalar loss.")
        loss = float(loss)
        if np.iscomplexobj(gradient):
            raise ValueError("The objective gradient must be real-valued.")
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != params.shape:
            raise ValueError("The objective gradient must have the same shape as parameters.")
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                "The objective returned non-finite loss or gradient; "
                "lower learning_rate or scale features."
            )
        return loss, gradient

    def minimize(self, objective: Objective, initial_params: ArrayLike) -> "GradientDescent":
        """Optimize from a copy of ``initial_params`` and return this optimizer."""
        self.learning_rate = validate_real(self.learning_rate, "learning_rate", strict=True)
        self.max_iter = validate_positive_int(self.max_iter, "max_iter")
        self.tol = validate_real(self.tol, "tol")
        if not callable(objective):
            raise TypeError("objective must be callable.")
        initial = np.asarray(initial_params)
        if initial.dtype.kind not in "iuf":
            raise ValueError("initial_params must contain real numeric values.")
        params = np.array(initial, dtype=np.float64, copy=True)
        if params.ndim != 1 or params.size == 0:
            raise ValueError("initial_params must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(params)):
            raise ValueError("initial_params must contain only finite values.")

        loss, gradient = self._evaluate(objective, params)
        self.params_ = params
        self.loss_history_ = [loss]
        self.n_iter_ = 0
        self.converged_ = False

        while True:
            # Scaling avoids squaring large finite gradients just to check a norm.
            largest = float(np.max(np.abs(gradient)))
            normalized_norm = float(np.linalg.norm(gradient / largest)) if largest else 0.0
            if largest == 0.0 or largest <= self.tol / normalized_norm:
                self.converged_ = True
                break
            if self.n_iter_ == self.max_iter:
                warnings.warn(
                    "Gradient descent reached max_iter before the gradient norm met tol.",
                    ConvergenceWarning,
                    stacklevel=2,
                )
                break

            with np.errstate(over="ignore", invalid="ignore"):
                candidate = self.params_ - self.learning_rate * gradient
            if not np.all(np.isfinite(candidate)):
                raise FloatingPointError(
                    "Gradient descent produced non-finite parameters; lower learning_rate."
                )
            candidate_loss, candidate_gradient = self._evaluate(objective, candidate)
            # A unit-sized floor would admit large relative increases in small
            # objectives. Compare the increase directly to avoid rounding the
            # tolerance away when adding it to the previous loss.
            loss_scale = max(abs(loss), abs(candidate_loss))
            rounding_slack = 64.0 * np.finfo(np.float64).eps * loss_scale
            if candidate_loss - loss > rounding_slack:
                raise FloatingPointError(
                    "Gradient descent increased the objective; lower learning_rate "
                    "or scale features."
                )

            self.params_ = candidate
            loss, gradient = candidate_loss, candidate_gradient
            self.loss_history_.append(loss)
            self.n_iter_ += 1

        return self
