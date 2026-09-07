"""Correctness, seeding, convergence, and numerical edge cases for K-means."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from ml_from_scratch.cluster import KMeans
from ml_from_scratch.utils.validation import ConvergenceWarning, NotFittedError


@pytest.fixture
def blobs():
    return make_blobs(
        n_samples=180,
        centers=[[-5.0, -4.0], [0.0, 5.0], [6.0, -2.0]],
        cluster_std=0.35,
        random_state=13,
    )


@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_obvious_clusters_and_learned_api(blobs, init):
    X, truth = blobs
    model = KMeans(3, init=init, random_state=7)
    assert model.fit(X) is model
    assert model.cluster_centers_.shape == (3, 2)
    assert model.labels_.shape == (X.shape[0],)
    assert model.n_features_in_ == 2
    assert len(np.unique(model.labels_)) == 3
    assert adjusted_rand_score(truth, model.labels_) == 1.0
    assert model.converged_
    assert 1 <= model.n_iter_ <= model.max_iter
    assert len(model.inertia_history_) == model.n_iter_ + 1
    assert model.inertia_history_[-1] == model.inertia_
    assert np.all(np.diff(model.inertia_history_) <= 1e-10)
    assert_array_equal(model.labels_, model.predict(X))
    expected = np.sum((X - model.cluster_centers_[model.labels_]) ** 2)
    assert_allclose(model.inertia_, expected, rtol=1e-14)
    for label in range(3):
        assert_allclose(model.cluster_centers_[label], X[model.labels_ == label].mean(axis=0))


def test_sklearn_agreement_is_permutation_invariant(blobs):
    X, _ = blobs
    model = KMeans(3, n_init=10, random_state=5).fit(X)
    reference = SklearnKMeans(n_clusters=3, n_init=10, random_state=5).fit(X)
    assert adjusted_rand_score(model.labels_, reference.labels_) == 1.0
    assert_allclose(model.inertia_, reference.inertia_, rtol=1e-12)


@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_seed_reproducibility_and_fit_predict(blobs, init):
    X, _ = blobs
    first = KMeans(3, init=init, random_state=42)
    second = KMeans(3, init=init, random_state=42)
    labels = first.fit_predict(X)
    second.fit(X)
    assert_array_equal(first.cluster_centers_, second.cluster_centers_)
    assert_array_equal(labels, second.labels_)
    # Mutating a returned prediction must not mutate learned assignments.
    labels[:] = -1
    assert np.all(first.labels_ >= 0)


def test_multiple_starts_cannot_worsen_the_same_first_start():
    X = np.random.default_rng(8).normal(size=(80, 3))
    single = KMeans(5, init="random", n_init=1, random_state=42).fit(X)
    multiple = KMeans(5, init="random", n_init=8, random_state=42).fit(X)
    assert multiple.inertia_ <= single.inertia_


def test_single_cluster_is_sample_mean():
    X = np.array([[0.0, 3.0], [2.0, 1.0], [7.0, -2.0]])
    model = KMeans(1, random_state=1).fit(X)
    assert_allclose(model.cluster_centers_[0], X.mean(axis=0), atol=1e-15)
    assert_array_equal(model.labels_, np.zeros(3, dtype=int))


def test_one_center_per_observation():
    X = np.array([[0.0], [3.0], [9.0], [12.0]])
    model = KMeans(4, random_state=1).fit(X)
    assert model.inertia_ == 0.0
    assert len(np.unique(model.labels_)) == 4


def test_empty_clusters_are_reseeded_without_nan():
    X = np.array([[0.0], [0.0], [0.0], [0.0], [10.0], [20.0]])
    model = KMeans(3, init="random", n_init=1, random_state=1).fit(X)
    assert_allclose(np.sort(model.cluster_centers_[:, 0]), [0.0, 10.0, 20.0])
    assert model.inertia_ == 0.0
    assert len(np.unique(model.labels_)) == 3


@pytest.mark.parametrize("init", ["random", "k-means++"])
def test_duplicate_samples_can_have_fewer_occupied_clusters(init):
    X = np.full((6, 2), 1e308)
    model = KMeans(4, init=init, n_init=1, random_state=2).fit(X)
    assert np.all(np.isfinite(model.cluster_centers_))
    assert model.inertia_ == 0.0
    assert model.converged_
    assert_array_equal(model.predict(X), model.labels_)


def test_large_offset_does_not_destroy_squared_distances():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [20.0, 21.0], [21.0, 20.0]])
    offset = 1e12
    base = KMeans(2, random_state=17).fit(X)
    translated = KMeans(2, random_state=17).fit(X + offset)
    assert adjusted_rand_score(base.labels_, translated.labels_) == 1.0
    assert_allclose(base.inertia_, translated.inertia_, atol=1e-10)
    assert_array_equal(translated.predict(X + offset), translated.labels_)


def test_budget_exhaustion_warns_and_final_labels_match_predict():
    X = np.random.default_rng(18).normal(size=(100, 2))
    model = KMeans(4, init="random", n_init=1, max_iter=1, tol=0, random_state=7)
    with pytest.warns(ConvergenceWarning, match="max_iter"):
        model.fit(X)
    assert not model.converged_
    assert model.n_iter_ == 1
    assert_array_equal(model.labels_, model.predict(X))
    assert_allclose(model.inertia_, np.sum((X - model.cluster_centers_[model.labels_]) ** 2))


def test_large_tolerance_can_stop_after_one_update():
    X = np.random.default_rng(18).normal(size=(100, 2))
    model = KMeans(4, n_init=1, tol=1e6, random_state=7).fit(X)
    assert model.n_iter_ == 1
    assert model.converged_
    assert_array_equal(model.labels_, model.predict(X))


def test_input_is_not_mutated_and_global_rng_is_not_used(blobs):
    X, _ = blobs
    original = X.copy()
    global_state = np.random.get_state()
    KMeans(3, random_state=7).fit(X)
    after = np.random.get_state()
    assert_array_equal(X, original)
    assert_array_equal(global_state[1], after[1])
    assert global_state[2:] == after[2:]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_clusters": 0},
        {"n_clusters": -1},
        {"n_clusters": 1.5},
        {"n_clusters": True},
        {"n_init": 0},
        {"max_iter": 0},
        {"tol": -1},
        {"tol": np.inf},
        {"init": "unknown"},
        {"random_state": -1},
        {"random_state": 1.2},
        {"random_state": True},
    ],
)
def test_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        KMeans(**kwargs)


@pytest.mark.parametrize("X", [[1, 2], [[np.nan]], [[np.inf]], [["bad"]], np.empty((0, 2))])
def test_invalid_training_data(X):
    with pytest.raises(ValueError):
        KMeans(1).fit(X)


def test_more_clusters_than_samples_is_rejected():
    with pytest.raises(ValueError, match="number of samples"):
        KMeans(3).fit([[0.0], [1.0]])


def test_prediction_validation():
    with pytest.raises(NotFittedError, match="not fitted"):
        KMeans(1).predict([[0.0]])
    model = KMeans(1, random_state=0).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="features"):
        model.predict([[0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        model.predict([[np.nan]])


def test_unrepresentable_training_distances_raise_clear_error():
    with pytest.raises(ValueError, match="rescale"):
        KMeans(1, n_init=1).fit([[0.0], [1e200]])


def test_unrepresentable_prediction_distances_raise_clear_error():
    model = KMeans(1, random_state=0).fit([[0.0], [1.0]])
    with pytest.raises(ValueError, match="rescale"):
        model.predict([[1e200]])


@pytest.mark.parametrize("prefit", [False, True])
def test_failed_later_start_preserves_complete_fitted_state(monkeypatch, prefit):
    original_data = np.array([[0.0], [1.0], [9.0], [10.0]])
    model = KMeans(2, n_init=2, random_state=7)
    if prefit:
        model.fit(original_data)
    prior_state = {
        name: deepcopy(value) for name, value in vars(model).items() if name.endswith("_")
    }
    single_run = model._single_run
    starts = 0

    def fail_on_second_start(X, rng):
        nonlocal starts
        starts += 1
        if starts == 2:
            raise FloatingPointError("Injected failure after one successful start")
        return single_run(X, rng)

    monkeypatch.setattr(model, "_single_run", fail_on_second_start)
    new_data = np.array([[0.0, 2.0], [1.0, 3.0], [20.0, 5.0], [21.0, 6.0], [22.0, 7.0]])
    with pytest.raises(ValueError, match="rescale"):
        model.fit(new_data)
    assert starts == 2
    current_state = {name: value for name, value in vars(model).items() if name.endswith("_")}
    assert current_state.keys() == prior_state.keys()
    for name in prior_state:
        assert_array_equal(current_state[name], prior_state[name])
    if prefit:
        assert_array_equal(model.predict(original_data), prior_state["labels_"])
    else:
        with pytest.raises(NotFittedError):
            model.predict(original_data)
