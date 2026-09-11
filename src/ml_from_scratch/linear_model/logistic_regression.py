"""Numerically stable binary logistic regression with optional L2 regularization."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.metrics import accuracy_score
from ml_from_scratch.optimization import GradientDescent
from ml_from_scratch.utils.numerical import sigmoid
from ml_from_scratch.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_X_y,
    validate_positive_int,
    validate_real,
)


class LogisticRegression:
    """Binary logistic regression using deterministic full-batch gradient descent.

    Labels must include both numeric classes 0 and 1. The minimized objective is
    mean binary cross entropy plus ``l2 / 2 * sum(coef_ ** 2)``. The intercept
    is never penalized. Logit-space ``logaddexp`` and a stable sigmoid avoid
    exponent overflow and ``log(0)``. ``predict_proba`` returns two columns in
    class order ``[0, 1]``; ``predict`` uses ``p(class=1) >= threshold``.

    Features should be scaled for fixed-step optimization. Perfectly separable
    data without regularization has no finite maximum-likelihood coefficients;
    the optimizer may reach its iteration limit before the requested tolerance.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 10000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        threshold: float = 0.5,
        l2: float = 0.0,
    ) -> None:
        if not isinstance(fit_intercept, bool):
            raise ValueError("fit_intercept must be a bool.")
        self.learning_rate = validate_real(learning_rate, "learning_rate", strict=True)
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = validate_real(tol, "tol")
        self.fit_intercept = fit_intercept
        self.threshold = validate_real(threshold, "threshold", maximum=1.0)
        self.l2 = validate_real(l2, "l2")

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """Fit on a finite matrix with both binary classes present."""
        features, targets = check_X_y(X, y, min_samples=2)
        if not np.array_equal(np.unique(targets), np.array([0.0, 1.0])):
            raise ValueError("y must contain both binary numeric labels 0 and 1.")
        self.threshold = validate_real(self.threshold, "threshold", maximum=1.0)
        self.l2 = validate_real(self.l2, "l2")
        design = (
            np.column_stack((features, np.ones(features.shape[0])))
            if self.fit_intercept
            else features
        )
        penalty_mask = np.ones(design.shape[1])
        if self.fit_intercept:
            penalty_mask[-1] = 0.0
        label_sign = 1.0 - 2.0 * targets

        def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            logits = design @ params
            # Choosing the sign before logaddexp avoids cancellation when y=1
            # and logits are large, as in logaddexp(0, logits) - y * logits.
            signed_logits = label_sign * logits
            penalized = params * penalty_mask
            loss = float(np.mean(np.logaddexp(0.0, signed_logits)))
            # Differentiate the signed loss directly: sigmoid(z) - 1 loses
            # representable negative tails when sigmoid(z) rounds to one.
            residual = label_sign * sigmoid(signed_logits)
            gradient = design.T @ residual / design.shape[0]
            if self.l2:
                loss += 0.5 * self.l2 * float(penalized @ penalized)
                gradient += self.l2 * penalized
            return loss, gradient

        optimizer = GradientDescent(self.learning_rate, self.max_iter, self.tol).minimize(
            objective, np.zeros(design.shape[1])
        )
        self.coef_ = (
            optimizer.params_[:-1].copy() if self.fit_intercept else optimizer.params_.copy()
        )
        self.intercept_ = float(optimizer.params_[-1]) if self.fit_intercept else 0.0
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.n_features_in_ = features.shape[1]
        self.loss_history_ = optimizer.loss_history_
        self.n_iter_ = optimizer.n_iter_
        self.converged_ = optimizer.converged_
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return class probabilities with shape ``(n_samples, 2)``."""
        check_is_fitted(self, ("coef_", "intercept_", "n_features_in_"))
        features = check_array(X)
        check_n_features(features, self.n_features_in_)
        with np.errstate(over="raise", invalid="raise"):
            positive = sigmoid(features @ self.coef_ + self.intercept_)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: ArrayLike) -> NDArray[np.int64]:
        """Classify samples using the configured probability threshold."""
        threshold = validate_real(self.threshold, "threshold", maximum=1.0)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(np.int64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the fraction of correctly classified samples."""
        return accuracy_score(y, self.predict(X))
