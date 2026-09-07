"""Principal component analysis through the thin singular value decomposition."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.utils.validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    validate_positive_int,
)

FloatArray = NDArray[np.float64]


class PCA:
    """Project centered data onto directions of greatest sample variance.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of directions to retain, between one and min(n_samples,
        n_features). None retains all directions available from the thin SVD.

    Notes
    -----
    This estimator centers features but does not standardize them or whiten
    projections. Explained variance uses the sample denominator n_samples - 1.
    Constant data have zero explained variance ratios. Component signs are
    chosen so that the largest-magnitude loading in each component is positive;
    degenerate singular subspaces may still have nonunique bases.
    A two-part training mean preserves centering accuracy for large offsets.
    """

    def __init__(self, n_components: int | None = None) -> None:
        self.n_components = (
            None if n_components is None else validate_positive_int(n_components, "n_components")
        )

    def fit(self, X: ArrayLike) -> PCA:
        """Learn an orthonormal basis from at least two finite observations."""
        data = check_array(X, min_samples=2)
        available = min(data.shape)
        selected = available if self.n_components is None else self.n_components
        if selected > available:
            raise ValueError("n_components must not exceed min(n_samples, n_features).")

        try:
            with np.errstate(over="raise", invalid="raise"):
                # Translation avoids summing a potentially very large common offset.
                offset = data[0]
                translated = data - offset
                mean_offset = np.mean(translated, axis=0)
                mean = offset + mean_offset
                centered = translated - mean_offset
                _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
                if not np.all(np.isfinite(singular_values)):
                    raise ValueError("SVD produced nonfinite values; rescale the input features.")
                # Divide before squaring so a representable sample variance survives.
                variance = (singular_values / np.sqrt(data.shape[0] - 1)) ** 2
                if singular_values[0] == 0:
                    variance_ratio = np.zeros_like(singular_values)
                else:
                    relative_squared = (singular_values / singular_values[0]) ** 2
                    variance_ratio = relative_squared / np.sum(relative_squared)
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            raise ValueError(
                "PCA numerical computation failed; rescale the input features."
            ) from error

        components = right_vectors[:selected].copy()
        largest_loading = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(selected), largest_loading])
        components *= signs[:, np.newaxis]
        self.mean_ = mean
        self._offset_ = offset.copy()
        self._mean_offset_ = mean_offset
        self.components_ = components
        self.explained_variance_ = variance[:selected].copy()
        self.explained_variance_ratio_ = variance_ratio[:selected].copy()
        self.singular_values_ = singular_values[:selected].copy()
        self.n_features_in_ = data.shape[1]
        self.n_components_ = selected
        return self

    def transform(self, X: ArrayLike) -> FloatArray:
        """Center samples using the training mean and project onto components."""
        check_is_fitted(self, ("components_", "mean_", "n_features_in_"))
        data = check_array(X)
        check_n_features(data, self.n_features_in_)
        try:
            with np.errstate(over="raise", invalid="raise"):
                projected = ((data - self._offset_) - self._mean_offset_) @ self.components_.T
        except FloatingPointError as error:
            raise ValueError("PCA projection overflowed; rescale the input features.") from error
        if not np.all(np.isfinite(projected)):
            raise ValueError("PCA projection overflowed; rescale the input features.")
        return projected

    def fit_transform(self, X: ArrayLike) -> FloatArray:
        """Learn principal directions and return projected training samples."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X: ArrayLike) -> FloatArray:
        """Reconstruct original feature coordinates from retained components."""
        check_is_fitted(self, ("components_", "mean_", "n_components_"))
        projected = check_array(X)
        check_n_features(projected, self.n_components_)
        try:
            with np.errstate(over="raise", invalid="raise"):
                reconstructed = (projected @ self.components_ + self._mean_offset_) + self._offset_
        except FloatingPointError as error:
            raise ValueError(
                "PCA reconstruction overflowed; rescale the input features."
            ) from error
        if not np.all(np.isfinite(reconstructed)):
            raise ValueError("PCA reconstruction overflowed; rescale the input features.")
        return reconstructed
