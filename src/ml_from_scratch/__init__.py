"""Foundational machine learning algorithms implemented with NumPy."""

from ml_from_scratch.cluster import KMeans
from ml_from_scratch.decomposition import PCA
from ml_from_scratch.linear_model import LinearRegression, LogisticRegression
from ml_from_scratch.optimization import GradientDescent
from ml_from_scratch.utils.validation import ConvergenceWarning, NotFittedError

__version__ = "0.1.0"
__all__ = [
    "PCA",
    "ConvergenceWarning",
    "GradientDescent",
    "KMeans",
    "LinearRegression",
    "LogisticRegression",
    "NotFittedError",
]
