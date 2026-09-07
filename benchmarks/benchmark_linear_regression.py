"""Compare gradient descent and least squares with the same reference predictions."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from benchmarks.common import (
    SEED,
    Row,
    Run,
    metric_row,
    paired_metrics,
    safe_learning_rate,
    split_standardized,
    timed_fit,
)
from ml_from_scratch import LinearRegression


def run() -> tuple[list[Row], list[Run]]:
    """Run leakage-free held-out evaluation on generated data and diabetes."""
    synthetic = make_regression(
        n_samples=600, n_features=8, n_informative=6, noise=10, bias=12, random_state=SEED
    )
    datasets = {"synthetic_regression": synthetic, "diabetes": load_diabetes(return_X_y=True)}
    rows: list[Row] = []
    runs: list[Run] = []
    for name, (features, targets) in datasets.items():
        train, test, y_train, y_test = split_standardized(features, targets)
        reference = SklearnLinearRegression()
        reference_seconds = timed_fit(reference, train, y_train)
        reference_predictions = reference.predict(test)
        for solver in ("gd", "normal"):
            learning_rate = safe_learning_rate(train)
            model = LinearRegression(
                solver=solver, learning_rate=learning_rate, max_iter=100_000, tol=1e-7
            )
            seconds = timed_fit(model, train, y_train)
            predictions = model.predict(test)
            algorithm = f"LinearRegression({solver})"
            rows.extend(
                paired_metrics(
                    algorithm,
                    name,
                    y_test,
                    predictions,
                    reference_predictions,
                    {"test_mse": mean_squared_error, "test_r2": r2_score},
                )
            )
            rows.extend(
                [
                    metric_row(
                        algorithm,
                        name,
                        "prediction_rmse_vs_reference",
                        np.sqrt(mean_squared_error(reference_predictions, predictions)),
                        0,
                    ),
                    metric_row(algorithm, name, "fit_seconds", seconds, reference_seconds),
                ]
            )
            runs.append(
                {
                    "algorithm": algorithm,
                    "dataset": name,
                    "n_train": train.shape[0],
                    "n_test": test.shape[0],
                    "n_features": train.shape[1],
                    "preprocessing": "Feature z-scores fitted on training data; target unscaled.",
                    "ours": {
                        "solver": solver,
                        "learning_rate": learning_rate,
                        "max_iter": 100_000,
                        "tol": 1e-7,
                        "n_iter": model.n_iter_,
                        "converged": model.converged_,
                        "final_loss": model.loss_history_[-1],
                        "fit_seconds": seconds,
                    },
                    "reference": {
                        "estimator": "sklearn.linear_model.LinearRegression",
                        "fit_intercept": True,
                        "fit_seconds": reference_seconds,
                    },
                    "assumptions": "Both minimize unregularized MSE with an unpenalized intercept.",
                }
            )
    return rows, runs
