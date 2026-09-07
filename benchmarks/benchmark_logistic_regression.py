"""Binary classification with explicitly matched L2 objectives."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss

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
from ml_from_scratch import LogisticRegression


def run() -> tuple[list[Row], list[Run]]:
    """Compare BCE + l2/2 ||w||² against sklearn's equivalent C parameterization."""
    features, targets = load_breast_cancer(return_X_y=True)
    train, test, y_train, y_test = split_standardized(features, targets, classification=True)
    l2 = 0.01
    learning_rate = safe_learning_rate(train, logistic=True, l2=l2)
    model = LogisticRegression(learning_rate=learning_rate, max_iter=50_000, tol=1e-8, l2=l2)
    reference_c = 1 / (train.shape[0] * l2)
    reference = SklearnLogisticRegression(
        C=reference_c, solver="lbfgs", max_iter=50_000, tol=1e-10, random_state=SEED
    )
    seconds = timed_fit(model, train, y_train)
    reference_seconds = timed_fit(reference, train, y_train)
    predictions, reference_predictions = model.predict(test), reference.predict(test)
    probabilities, reference_probabilities = (
        model.predict_proba(test),
        reference.predict_proba(test),
    )
    rows = paired_metrics(
        "LogisticRegression",
        "breast_cancer",
        y_test,
        predictions,
        reference_predictions,
        {"test_accuracy": accuracy_score, "test_f1": f1_score},
    )
    rows.extend(
        [
            metric_row(
                "LogisticRegression",
                "breast_cancer",
                "test_log_loss",
                log_loss(y_test, probabilities),
                log_loss(y_test, reference_probabilities),
            ),
            metric_row(
                "LogisticRegression",
                "breast_cancer",
                "probability_max_abs_difference",
                np.max(np.abs(probabilities - reference_probabilities)),
                0,
            ),
            metric_row(
                "LogisticRegression", "breast_cancer", "fit_seconds", seconds, reference_seconds
            ),
        ]
    )
    run_metadata: Run = {
        "algorithm": "LogisticRegression",
        "dataset": "breast_cancer",
        "n_train": train.shape[0],
        "n_test": test.shape[0],
        "n_features": train.shape[1],
        "preprocessing": "Feature z-scores fitted on training data; stratified split.",
        "ours": {
            "learning_rate": learning_rate,
            "max_iter": 50_000,
            "tol": 1e-8,
            "l2": l2,
            "threshold": 0.5,
            "n_iter": model.n_iter_,
            "converged": model.converged_,
            "final_loss": model.loss_history_[-1],
            "fit_seconds": seconds,
        },
        "reference": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs",
            "C": reference_c,
            "max_iter": 50_000,
            "tol": 1e-10,
            "n_iter": int(reference.n_iter_[0]),
            "fit_seconds": reference_seconds,
        },
        "assumptions": (
            "Both minimize mean BCE + l2/2 * squared weight norm; intercept is unpenalized. "
            "sklearn C=1/(n_train*l2). Different optimizers have different stopping criteria. "
            "Positive label 1 is benign; F1 follows the dataset encoding."
        ),
    }
    return rows, [run_metadata]
