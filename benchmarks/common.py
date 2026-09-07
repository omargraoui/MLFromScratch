"""Shared data preparation, measurement, and reporting for experiments."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

SEED = 42
FloatArray = NDArray[np.float64]
Row = dict[str, str | float | None]
Run = dict[str, Any]


def standardize(
    training: FloatArray, other: FloatArray | None = None
) -> tuple[FloatArray, FloatArray | None]:
    """Use training population statistics; constant features get a unit scale."""
    mean = training.mean(axis=0)
    scale = training.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    transformed = (training - mean) / scale
    return transformed, None if other is None else (other - mean) / scale


def split_standardized(
    features: FloatArray, targets: FloatArray, *, classification: bool = False
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Make a fixed 75/25 split before estimating any preprocessing parameters."""
    train, test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=0.25,
        random_state=SEED,
        stratify=targets if classification else None,
    )
    train, test = standardize(train, test)
    assert test is not None
    return train, test, y_train, y_test


def timed_fit(estimator: Any, features: FloatArray, targets: FloatArray | None = None) -> float:
    """Measure fitting only, with data loading and metric evaluation excluded."""
    start = perf_counter()
    if targets is None:
        estimator.fit(features)
    else:
        estimator.fit(features, targets)
    return perf_counter() - start


def metric_row(algorithm: str, dataset: str, metric: str, ours: float, reference: float) -> Row:
    """A zero reference has no defined relative difference (JSON null / CSV blank)."""
    ours, reference = float(ours), float(reference)
    if not np.isfinite([ours, reference]).all():
        raise ValueError(f"Non-finite benchmark result: {algorithm}/{dataset}/{metric}.")
    difference = abs(ours - reference)
    return {
        "algorithm": algorithm,
        "dataset": dataset,
        "metric": metric,
        "from_scratch": ours,
        "scikit_learn": reference,
        "absolute_difference": difference,
        "relative_difference": difference / abs(reference) if reference else None,
    }


def paired_metrics(
    algorithm: str,
    dataset: str,
    target: FloatArray,
    ours: FloatArray,
    reference: FloatArray,
    metrics: dict[str, Callable[..., float]],
) -> list[Row]:
    """Apply independent scikit-learn metrics to predictions from both models."""
    return [
        metric_row(algorithm, dataset, name, metric(target, ours), metric(target, reference))
        for name, metric in metrics.items()
    ]


def safe_learning_rate(features: FloatArray, *, logistic: bool = False, l2: float = 0) -> float:
    """Use 0.9/L with a spectral upper bound on the objective's gradient Lipschitz constant."""
    augmented = np.column_stack((features, np.ones(features.shape[0])))
    largest_squared = np.linalg.norm(augmented, ord=2) ** 2 / features.shape[0]
    bound = 0.25 * largest_squared + l2 if logistic else 2 * largest_squared
    return float(0.9 / bound)


def environment_metadata() -> Run:
    """Record the environment used to generate numbers; runtimes remain machine dependent."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "blas_threads": 1,
        "timing": "One perf_counter fit measurement per estimator, excluding preprocessing.",
        "dataset_count": 6,
        "datasets": ["synthetic_regression", "diabetes", "breast_cancer", "iris", "wine", "digits"],
        "comparison_metrics": "scikit-learn metrics independently applied to both implementations",
        "split": "Supervised: 75% train / 25% test, seed=42, classification stratified.",
    }


def markdown_table(rows: list[Row]) -> str:
    """Render a compact comparison table without requiring a dataframe dependency."""
    lines = [
        "| Algorithm | Dataset | Metric | From scratch | scikit-learn | Absolute difference | "
        "Relative difference |",
        "| :-- | :-- | :-- | --: | --: | --: | --: |",
    ]
    for row in rows:
        numeric = [
            "—" if row[field] is None else f"{row[field]:.6g}"
            for field in (
                "from_scratch",
                "scikit_learn",
                "absolute_difference",
                "relative_difference",
            )
        ]
        lines.append(
            f"| {row['algorithm']} | {row['dataset']} | {row['metric']} | "
            + " | ".join(numeric)
            + " |"
        )
    return "\n".join(lines) + "\n"
