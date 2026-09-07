"""Ordinary least squares using a direct solve or reusable gradient descent."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.metrics import r2_score
from ml_from_scratch.optimization import GradientDescent
from ml_from_scratch.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
    validate_positive_int,
    validate_real,
)


class LinearRegression:
    """Fit ``y = X @ coef_ + intercept_`` by minimizing mean squared error.

    ``solver='normal'`` uses ``numpy.linalg.lstsq`` on centered data, avoiding
    matrix inversion and handling rank-deficient feature matrices. ``'gd'``
    starts at zero and uses full-batch gradient descent; scale features and
    choose a suitable learning rate for this solver.

    The direct solver exposes a one-element ``loss_history_``, ``n_iter_=0``,
    and ``converged_=True``. The gradient solver records its initial loss and
    all accepted steps. Only single-output regression is supported.
    """

    def __init__(
        self,
        solver: str = "normal",
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        tol: float = 1e-6,
    ) -> None:
        if solver not in ("normal", "gd"):
            raise ValueError("solver must be 'normal' or 'gd'.")
        if not isinstance(fit_intercept, bool):
            raise ValueError("fit_intercept must be a bool.")
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.learning_rate = validate_real(learning_rate, "learning_rate", strict=True)
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = validate_real(tol, "tol")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """Learn coefficients from a finite feature matrix and one-dimensional targets."""
        features, targets = check_X_y(X, y)
        if self.solver not in ("normal", "gd"):
            raise ValueError("solver must be 'normal' or 'gd'.")
        if self.solver == "normal":
            feature_mean = (
                features.mean(axis=0) if self.fit_intercept else np.zeros(features.shape[1])
            )
            target_mean = float(targets.mean()) if self.fit_intercept else 0.0
            coefficients, _, _, _ = np.linalg.lstsq(
                features - feature_mean, targets - target_mean, rcond=None
            )
            intercept = float(target_mean - feature_mean @ coefficients)
            with np.errstate(over="raise", invalid="raise"):
                residual = features @ coefficients + intercept - targets
                loss = float(np.mean(residual**2))
            if not np.isfinite(loss):
                raise FloatingPointError("Least-squares solution produced a non-finite loss.")
            self.loss_history_ = [loss]
            self.n_iter_ = 0
            self.converged_ = True
        else:
            design = (
                np.column_stack((features, np.ones(features.shape[0])))
                if self.fit_intercept
                else features
            )

            def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
                residual = design @ params - targets
                loss = float(np.mean(residual**2))
                gradient = (2.0 / design.shape[0]) * (design.T @ residual)
                return loss, gradient

            optimizer = GradientDescent(self.learning_rate, self.max_iter, self.tol).minimize(
                objective, np.zeros(design.shape[1])
            )
            coefficients = optimizer.params_[:-1] if self.fit_intercept else optimizer.params_
            intercept = float(optimizer.params_[-1]) if self.fit_intercept else 0.0
            self.loss_history_ = optimizer.loss_history_
            self.n_iter_ = optimizer.n_iter_
            self.converged_ = optimizer.converged_

        self.coef_ = coefficients.copy()
        self.intercept_ = intercept
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict a one-dimensional target vector for new samples."""
        check_is_fitted(self, ("coef_", "intercept_", "n_features_in_"))
        features = check_array(X)
        check_n_features(features, self.n_features_in_)
        with np.errstate(over="raise", invalid="raise"):
            return np.asarray(features @ self.coef_ + self.intercept_, dtype=np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R²."""
        return r2_score(y, self.predict(X))
