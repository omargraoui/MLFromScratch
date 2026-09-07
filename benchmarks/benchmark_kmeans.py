"""Clustering comparisons use inertia and permutation-invariant agreement."""

from __future__ import annotations

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import adjusted_rand_score

from benchmarks.common import SEED, Row, Run, metric_row, standardize, timed_fit
from ml_from_scratch import KMeans


def run() -> tuple[list[Row], list[Run]]:
    """Evaluate full-data exploratory clustering; known labels are evaluation-only."""
    rows: list[Row] = []
    runs: list[Run] = []
    for name, loader in (("iris", load_iris), ("wine", load_wine)):
        features, targets = loader(return_X_y=True)
        features, _ = standardize(features)
        settings = {
            "n_clusters": 3,
            "init": "k-means++",
            "n_init": 10,
            "max_iter": 300,
            "tol": 1e-6,
            "random_state": SEED,
        }
        model = KMeans(**settings)
        reference = SklearnKMeans(**settings, algorithm="lloyd")
        seconds = timed_fit(model, features)
        reference_seconds = timed_fit(reference, features)
        for metric, ours, sklearn in (
            ("inertia", model.inertia_, reference.inertia_),
            (
                "ari_vs_known_labels",
                adjusted_rand_score(targets, model.labels_),
                adjusted_rand_score(targets, reference.labels_),
            ),
            ("ari_vs_reference", adjusted_rand_score(model.labels_, reference.labels_), 1.0),
            ("iterations", model.n_iter_, reference.n_iter_),
            ("fit_seconds", seconds, reference_seconds),
        ):
            rows.append(metric_row("KMeans", name, metric, ours, sklearn))
        runs.append(
            {
                "algorithm": "KMeans",
                "dataset": name,
                "n_samples": features.shape[0],
                "n_features": features.shape[1],
                "preprocessing": "Whole-data z-scores for exploratory clustering.",
                "settings": settings,
                "ours": {"n_iter": model.n_iter_, "fit_seconds": seconds},
                "reference": {
                    "estimator": "sklearn.cluster.KMeans",
                    "algorithm": "lloyd",
                    "n_iter": int(reference.n_iter_),
                    "fit_seconds": reference_seconds,
                },
                "assumptions": (
                    "Same seed and restart count, but NumPy Generator and sklearn RNG differ. "
                    "Our vanilla and sklearn greedy k-means++ may choose different starts. "
                    "Our absolute centroid displacement tolerance differs from sklearn's scaled "
                    "tolerance. Local minima and iteration counts may therefore differ."
                ),
            }
        )
    return rows, runs
