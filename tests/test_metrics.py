"""Reference agreement, degenerate cases, and metric input contracts."""

import numpy as np
import pytest
from sklearn import metrics as reference

from ml_from_scratch import metrics


@pytest.mark.parametrize(
    "name", ["mean_squared_error", "root_mean_squared_error", "mean_absolute_error", "r2_score"]
)
def test_regression_metrics_match_reference(name):
    truth = np.array([-3.0, 0, 1, 9])
    prediction = np.array([-2.0, 0.3, 1.1, 7])
    assert getattr(metrics, name)(truth, prediction) == pytest.approx(
        getattr(reference, name)(truth, prediction)
    )


def test_r_squared_constant_targets_and_negative_score():
    assert metrics.r2_score([2, 2], [2, 2]) == 1
    assert metrics.r2_score([2, 2], [1, 1]) == 0
    assert metrics.r2_score([0, 1], [10, 10]) < 0
    with pytest.raises(ValueError, match="two samples"):
        metrics.r2_score([1], [1])


@pytest.mark.parametrize("name", ["accuracy_score", "precision_score", "recall_score", "f1_score"])
def test_classification_matches_reference(name):
    truth = [0, 0, 1, 1, 1, 0]
    prediction = [0, 1, 1, 0, 1, 0]
    assert getattr(metrics, name)(truth, prediction) == pytest.approx(
        getattr(reference, name)(truth, prediction)
    )


@pytest.mark.parametrize("name", ["precision_score", "recall_score", "f1_score"])
def test_binary_zero_division_and_invalid_labels(name):
    function = getattr(metrics, name)
    assert function([0, 0], [0, 0]) == 0
    with pytest.raises(ValueError, match="labels 0 and 1"):
        function([0, 2], [0, 1])
    with pytest.raises(ValueError, match="labels 0 and 1"):
        function([0, 1], [0, 2])


def test_multiclass_accuracy():
    assert metrics.accuracy_score([0, 2, 3], [0, 3, 3]) == pytest.approx(2 / 3)


def test_log_loss_stable_at_probability_boundaries():
    truth = [0, 1, 0, 1]
    proba = [0, 1, 0.8, 0.2]
    assert metrics.log_loss(truth, proba) == pytest.approx(reference.log_loss(truth, proba))
    assert np.isfinite(metrics.log_loss([0, 1], [1, 0]))
    assert metrics.log_loss([0, 0], [0.1, 0.2]) > 0
    with pytest.raises(ValueError, match="labels"):
        metrics.log_loss([0, 2], [0.1, 0.8])
    with pytest.raises(ValueError, match="Probabilities"):
        metrics.log_loss([0, 1], [-0.1, 0.8])


@pytest.mark.parametrize("name", ["mean_squared_error", "accuracy_score", "adjusted_rand_score"])
def test_metrics_reject_mismatched_vectors(name):
    with pytest.raises(ValueError, match="same shape"):
        getattr(metrics, name)([0, 1], [0])


def test_inertia_known_answer():
    assert metrics.inertia_score([[0], [2], [10], [12]], [[1], [11]], [0, 0, 1, 1]) == 4


@pytest.mark.parametrize("labels", [[0], [0, 2], [0, -1], [0, 0.5]])
def test_invalid_inertia_labels(labels):
    with pytest.raises(ValueError):
        metrics.inertia_score([[0], [1]], [[0], [1]], labels)


@pytest.mark.parametrize(
    ("truth", "prediction"),
    [
        ([1], [4]),
        ([0, 0], [2, 2]),
        ([0, 1], [3, 4]),
        ([0, 0, 1, 1], [1, 1, 0, 0]),
        ([0, 0, 1, 1], [0, 1, 0, 1]),
        ([0, 0, 0, 1, 1], [1, 0, 2, 2, 0]),
    ],
)
def test_adjusted_rand_score_matches_reference(truth, prediction):
    assert metrics.adjusted_rand_score(truth, prediction) == pytest.approx(
        reference.adjusted_rand_score(truth, prediction)
    )


def test_random_partitions_ari():
    rng = np.random.default_rng(42)
    for _ in range(20):
        truth = rng.integers(0, 4, 30)
        prediction = rng.integers(0, 6, 30)
        assert metrics.adjusted_rand_score(truth, prediction) == pytest.approx(
            reference.adjusted_rand_score(truth, prediction)
        )
