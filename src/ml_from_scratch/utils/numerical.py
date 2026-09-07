"""Numerically stable elementary functions."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sigmoid(z: ArrayLike) -> NDArray[np.float64]:
    """Compute the logistic sigmoid without exponentiating positive arguments.

    Infinite logits have their limiting probabilities; NaN inputs remain NaN.
    Estimator input validation separately rejects nonfinite feature values.
    """
    values = np.asarray(z, dtype=np.float64)
    with np.errstate(under="ignore"):
        exp_negative_abs = np.exp(-np.abs(values))
    return np.where(
        values >= 0,
        1.0 / (1.0 + exp_negative_abs),
        exp_negative_abs / (1.0 + exp_negative_abs),
    )
