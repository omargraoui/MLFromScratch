"""Small, explicit validators shared by estimators and metrics.

Inputs must be real numeric arrays; strings, complex values and nonfinite
values are rejected instead of being silently coerced. Arrays are represented
as float64 for numerical computations. Validation never mutates caller data.
"""

from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NotFittedError(ValueError):
    """An operation requires learned attributes that are not yet available."""


class ConvergenceWarning(UserWarning):
    """An iterative algorithm exhausted its budget before convergence."""


def _as_numeric_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array.") from exc
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def check_array(X: ArrayLike, *, min_samples: int = 1) -> NDArray[np.float64]:
    """Validate a nonempty 2D finite real feature matrix."""
    array = _as_numeric_array(X, "X")
    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional array (n_samples, n_features).")
    if array.shape[0] < min_samples:
        raise ValueError(f"X must contain at least {min_samples} sample(s).")
    if array.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    return array


def check_vector(y: ArrayLike, *, name: str = "y") -> NDArray[np.float64]:
    """Validate a nonempty 1D finite real vector; column vectors are rejected."""
    array = _as_numeric_array(y, name)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    return array


def check_X_y(
    X: ArrayLike, y: ArrayLike, *, min_samples: int = 1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate supervised inputs with consistent sample counts."""
    features = check_array(X, min_samples=min_samples)
    targets = check_vector(y)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    return features, targets


def check_n_features(X: NDArray[np.float64], n_features: int) -> None:
    """Reject prediction or transformation inputs with a different feature count."""
    if X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features; expected {n_features}.")


def check_is_fitted(estimator: Any, attributes: str | tuple[str, ...]) -> None:
    """Raise a meaningful error if required learned attributes are unavailable."""
    names = (attributes,) if isinstance(attributes, str) else attributes
    if not all(hasattr(estimator, name) for name in names):
        raise NotFittedError(f"{type(estimator).__name__} is not fitted. Call fit first.")


def validate_positive_int(value: int, name: str) -> int:
    """Accept positive integers, including NumPy integers, but reject booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def validate_real(
    value: float,
    name: str,
    *,
    minimum: float = 0.0,
    strict: bool = False,
    maximum: float | None = None,
) -> float:
    """Validate a finite scalar against an inclusive or strict lower bound."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number.")
    if result < minimum or (strict and result == minimum):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result
