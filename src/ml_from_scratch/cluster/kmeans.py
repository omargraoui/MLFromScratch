"""Lloyd's algorithm with reproducible random and k-means++ initialization."""

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_from_scratch.utils.validation import (
    ConvergenceWarning,
    check_array,
    check_is_fitted,
    check_n_features,
    validate_positive_int,
    validate_real,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _squared_distances(X: FloatArray, centers: FloatArray) -> FloatArray:
    """Compute distances directly, avoiding cancellation in the norm expansion.

    Looping over centers uses an (n_samples, n_features) temporary instead of
    materializing an (n_samples, n_clusters, n_features) tensor.
    """
    distances = np.empty((X.shape[0], centers.shape[0]), dtype=np.float64)
    for index, center in enumerate(centers):
        difference = X - center
        distances[:, index] = np.einsum("ij,ij->i", difference, difference)
    if not np.all(np.isfinite(distances)):
        raise ValueError("Squared distances overflowed; rescale the input features.")
    return distances


class KMeans:
    """Partition samples by minimizing within-cluster squared Euclidean distance.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of centers, at most the number of training samples.
    init : {"k-means++", "random"}, default="k-means++"
        Distance-weighted seeding or sampling without replacement.
    n_init : int, default=10
        Independent starts; retain the result with the smallest final inertia.
    max_iter : int, default=300
        Maximum number of centroid updates per start.
    tol : float, default=1e-4
        Absolute Frobenius norm tolerance for the change in centers. Unchanged
        assignments also count as convergence.
    random_state : int or None, default=None
        Seed for a local NumPy generator; does not modify NumPy's global state.

    Notes
    -----
    Empty clusters are reseeded with samples farthest from the updated centers,
    refreshing distances after each reseed. When there are fewer distinct
    observations than clusters, repeated centers and fewer occupied labels are
    unavoidable and permitted.
    ``inertia_history_`` includes initialization and each centroid update for
    the selected start. ``n_iter_`` counts those updates. Training data are
    internally translated for numerical accuracy; input arrays are not changed.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = validate_positive_int(n_clusters, "n_clusters")
        if init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")
        self.init = init
        self.n_init = validate_positive_int(n_init, "n_init")
        self.max_iter = validate_positive_int(max_iter, "max_iter")
        self.tol = validate_real(tol, "tol")
        if random_state is not None and (
            isinstance(random_state, bool)
            or not isinstance(random_state, Integral)
            or random_state < 0
        ):
            raise ValueError("random_state must be a nonnegative integer or None.")
        self.random_state = None if random_state is None else int(random_state)

    def _initialize(self, X: FloatArray, rng: np.random.Generator) -> FloatArray:
        if self.init == "random":
            return X[rng.choice(X.shape[0], size=self.n_clusters, replace=False)].copy()

        indices = [int(rng.integers(X.shape[0]))]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=np.float64)
        centers[0] = X[indices[0]]
        closest = _squared_distances(X, centers[:1])[:, 0]
        for index in range(1, self.n_clusters):
            largest = float(np.max(closest))
            if largest == 0.0:
                # All remaining distinct values have already been selected.
                available = np.setdiff1d(np.arange(X.shape[0]), indices)
                selected = int(rng.choice(available))
            else:
                # Scaling before summation prevents probability overflow.
                weights = closest / largest
                selected = int(rng.choice(X.shape[0], p=weights / np.sum(weights)))
            indices.append(selected)
            centers[index] = X[selected]
            closest = np.minimum(closest, _squared_distances(X, centers[index : index + 1])[:, 0])
        return centers

    def _single_run(
        self, X: FloatArray, rng: np.random.Generator
    ) -> tuple[FloatArray, list[float], int, bool]:
        centers = self._initialize(X, rng)
        distances = _squared_distances(X, centers)
        labels = np.argmin(distances, axis=1)
        nearest = distances[np.arange(X.shape[0]), labels]
        history = [float(np.sum(nearest))]
        converged = False

        for _ in range(self.max_iter):
            updated = np.empty_like(centers)
            counts = np.bincount(labels, minlength=self.n_clusters)
            for cluster in np.flatnonzero(counts):
                updated[cluster] = np.mean(X[labels == cluster], axis=0)
            empty = np.flatnonzero(counts == 0)
            if empty.size:
                reseed_distances = np.min(_squared_distances(X, updated[counts > 0]), axis=1)
                for cluster in empty:
                    farthest = int(np.argmax(reseed_distances))
                    updated[cluster] = X[farthest]
                    reseed_distances = np.minimum(
                        reseed_distances,
                        _squared_distances(X, updated[cluster : cluster + 1])[:, 0],
                    )
                    reseed_distances[farthest] = -np.inf

            distances = _squared_distances(X, updated)
            new_labels = np.argmin(distances, axis=1)
            nearest = distances[np.arange(X.shape[0]), new_labels]
            history.append(float(np.sum(nearest)))
            center_shift = float(np.linalg.norm(updated - centers))
            unchanged = np.array_equal(labels, new_labels)
            centers, labels = updated, new_labels
            if unchanged or center_shift <= self.tol:
                converged = True
                break
        return centers, history, len(history) - 1, converged

    def fit(self, X: ArrayLike) -> KMeans:
        """Fit centers and retain the best start, preserving prior state on failure."""
        data = check_array(X)
        if self.n_clusters > data.shape[0]:
            raise ValueError("n_clusters must not exceed the number of samples.")
        rng = np.random.default_rng(self.random_state)
        best_result: tuple[FloatArray, IntArray, float, int, list[float], bool] | None = None
        try:
            with np.errstate(over="raise", invalid="raise"):
                offset = data[0].copy()
                centered = data - offset
                for _ in range(self.n_init):
                    centers, history, iterations, converged = self._single_run(centered, rng)
                    centers = centers + offset
                    # Rounding when restoring the offset can change assignments.
                    distances = _squared_distances(data, centers)
                    labels = np.argmin(distances, axis=1).astype(np.int64)
                    inertia = float(np.sum(distances[np.arange(data.shape[0]), labels]))
                    if not np.isfinite(inertia):
                        raise ValueError("Inertia overflowed; rescale the input features.")
                    if best_result is None or inertia < best_result[2]:
                        history[-1] = inertia
                        best_result = centers, labels, inertia, iterations, history, converged
        except FloatingPointError as error:
            raise ValueError("KMeans arithmetic overflowed; rescale the input features.") from error
        # A positive n_init guarantees a candidate whenever computation succeeds.
        assert best_result is not None
        if not best_result[-1]:
            warnings.warn(
                "KMeans reached max_iter before convergence; increase max_iter or rescale data.",
                ConvergenceWarning,
                stacklevel=2,
            )
        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
            self.inertia_history_,
            self.converged_,
        ) = best_result
        self.n_features_in_ = data.shape[1]
        return self

    def predict(self, X: ArrayLike) -> IntArray:
        """Return the index of the closest learned center for each sample."""
        check_is_fitted(self, ("cluster_centers_", "n_features_in_"))
        data = check_array(X)
        check_n_features(data, self.n_features_in_)
        try:
            with np.errstate(over="raise", invalid="raise"):
                distances = _squared_distances(data, self.cluster_centers_)
        except FloatingPointError as error:
            raise ValueError("KMeans arithmetic overflowed; rescale the input features.") from error
        return np.argmin(distances, axis=1).astype(np.int64)

    def fit_predict(self, X: ArrayLike) -> IntArray:
        """Fit the estimator and return training-sample cluster assignments."""
        return self.fit(X).labels_.copy()
