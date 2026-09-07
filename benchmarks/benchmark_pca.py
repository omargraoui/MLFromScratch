"""PCA agreement through reconstruction and sign-invariant projection matrices."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.metrics import mean_squared_error

from benchmarks.common import Row, Run, metric_row, standardize, timed_fit
from ml_from_scratch import PCA


def run() -> tuple[list[Row], list[Run]]:
    """Use deterministic full SVD on identical centered feature matrices."""
    rows: list[Row] = []
    runs: list[Run] = []
    for name, loader, dimensions in (
        ("iris", load_iris, 2),
        ("wine", load_wine, 5),
        ("digits", load_digits, 20),
    ):
        features, _ = loader(return_X_y=True)
        if name == "digits":
            features = features / 16.0
            preprocessing = "Pixel intensities /16; centering occurs inside both estimators."
        else:
            features, _ = standardize(features)
            preprocessing = "Whole-data z-scores; centering occurs inside both estimators."
        model = PCA(n_components=dimensions)
        reference = SklearnPCA(n_components=dimensions, svd_solver="full")
        seconds = timed_fit(model, features)
        reference_seconds = timed_fit(reference, features)
        reconstruction = model.inverse_transform(model.transform(features))
        reference_reconstruction = reference.inverse_transform(reference.transform(features))
        projector = model.components_.T @ model.components_
        reference_projector = reference.components_.T @ reference.components_
        for metric, ours, sklearn in (
            (
                "explained_variance_ratio_sum",
                model.explained_variance_ratio_.sum(),
                reference.explained_variance_ratio_.sum(),
            ),
            (
                "reconstruction_mse",
                mean_squared_error(features, reconstruction),
                mean_squared_error(features, reference_reconstruction),
            ),
            ("projector_frobenius_distance", np.linalg.norm(projector - reference_projector), 0),
            ("fit_seconds", seconds, reference_seconds),
        ):
            rows.append(metric_row("PCA", name, metric, ours, sklearn))
        runs.append(
            {
                "algorithm": "PCA",
                "dataset": name,
                "n_samples": features.shape[0],
                "n_features": features.shape[1],
                "n_components": dimensions,
                "preprocessing": preprocessing,
                "ours": {"solver": "numpy.linalg.svd", "fit_seconds": seconds},
                "reference": {
                    "estimator": "sklearn.decomposition.PCA",
                    "svd_solver": "full",
                    "fit_seconds": reference_seconds,
                },
                "assumptions": (
                    "Both use sample variance (n-1 denominator), no whitening. "
                    "Reconstruction and projectors ignore component sign ambiguity. "
                    "These are in-sample dimensionality reduction diagnostics."
                ),
            }
        )
    return rows, runs
