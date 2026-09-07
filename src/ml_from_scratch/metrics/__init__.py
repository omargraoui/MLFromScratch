"""Small, independently testable evaluation metrics implemented with NumPy."""

from ml_from_scratch.metrics.classification import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from ml_from_scratch.metrics.clustering import adjusted_rand_score, inertia_score
from ml_from_scratch.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

__all__ = [
    "accuracy_score",
    "adjusted_rand_score",
    "f1_score",
    "inertia_score",
    "log_loss",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "root_mean_squared_error",
]
