"""Sample variance, reconstruction, and sign-invariant reference checks for PCA."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.decomposition import PCA as SklearnPCA

from ml_from_scratch.decomposition import PCA
from ml_from_scratch.utils.validation import NotFittedError


@pytest.fixture
def correlated_data():
    rng = np.random.default_rng(24)
    latent = rng.normal(size=(100, 3))
    return latent @ np.array([[3.0, 1.0, 0.0], [0.0, 2.0, 0.5], [0.2, 0.0, 0.3]]) + 7.0


def test_learned_attributes_and_projection_shape(correlated_data):
    model = PCA(2)
    assert model.fit(correlated_data) is model
    transformed = model.transform(correlated_data)
    assert transformed.shape == (100, 2)
    assert model.components_.shape == (2, 3)
    assert model.explained_variance_.shape == (2,)
    assert model.singular_values_.shape == (2,)
    assert model.n_components_ == 2
    assert model.n_features_in_ == 3
    assert np.all(np.diff(model.explained_variance_) <= 0.0)
    assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-14)
    assert_allclose(model.components_ @ model.components_.T, np.eye(2), atol=1e-14)


def test_known_sample_variance():
    X = np.array([[-2.0, 4.0], [-1.0, 4.0], [1.0, 4.0], [2.0, 4.0]])
    model = PCA(1).fit(X)
    assert_allclose(model.mean_, [0.0, 4.0])
    assert_allclose(model.explained_variance_, [10.0 / 3.0])
    assert_allclose(model.explained_variance_ratio_, [1.0])
    assert_allclose(model.singular_values_, [np.sqrt(10.0)])
    assert_allclose(model.components_.T @ model.components_, [[1.0, 0.0], [0.0, 0.0]])


def test_sklearn_agreement_without_assuming_component_signs(correlated_data):
    model = PCA(2).fit(correlated_data)
    reference = SklearnPCA(n_components=2, svd_solver="full").fit(correlated_data)
    assert_allclose(model.explained_variance_, reference.explained_variance_, rtol=1e-12)
    assert_allclose(
        model.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-12
    )
    assert_allclose(model.singular_values_, reference.singular_values_, rtol=1e-12)
    # Orthogonal projectors identify the subspace even if a basis changes sign.
    assert_allclose(
        model.components_.T @ model.components_,
        reference.components_.T @ reference.components_,
        atol=1e-12,
    )
    assert_allclose(
        model.inverse_transform(model.transform(correlated_data)),
        reference.inverse_transform(reference.transform(correlated_data)),
        atol=1e-12,
    )


@pytest.mark.parametrize("shape", [(25, 4), (4, 25)])
def test_all_components_reconstruct_tall_and_wide_data(shape):
    X = np.random.default_rng(8).normal(size=shape)
    model = PCA()
    transformed = model.fit_transform(X)
    assert transformed.shape == (shape[0], min(shape))
    assert_allclose(model.inverse_transform(transformed), X, atol=1e-13)
    assert_allclose(np.sum(model.explained_variance_ratio_), 1.0)


def test_retained_variance_and_reconstruction_error(correlated_data):
    X = correlated_data
    reduced = PCA(1).fit(X)
    full = PCA().fit(X)
    reconstruction = reduced.inverse_transform(reduced.transform(X))
    squared_error = np.sum((X - reconstruction) ** 2)
    expected_error = (X.shape[0] - 1) * np.sum(full.explained_variance_[1:])
    assert_allclose(squared_error, expected_error, rtol=1e-12)
    assert_allclose(
        reduced.explained_variance_ratio_[0],
        reduced.explained_variance_[0] / np.sum(full.explained_variance_),
    )
    assert 0 < reduced.explained_variance_ratio_.sum() < 1


def test_rank_deficient_data_reconstruct_with_true_rank():
    x = np.linspace(-3.0, 3.0, 30)
    X = np.column_stack([x, 2.0 * x, -5.0 * x]) + 4.0
    model = PCA(1)
    assert_allclose(model.inverse_transform(model.fit_transform(X)), X, atol=1e-13)
    assert_allclose(model.explained_variance_ratio_, [1.0])


@pytest.mark.parametrize("value", [0.0, 0.1, 1e308])
def test_constant_features_have_zero_variance_ratios(value):
    X = np.full((5, 3), value)
    model = PCA(2)
    transformed = model.fit_transform(X)
    assert_array_equal(transformed, np.zeros((5, 2)))
    assert_array_equal(model.explained_variance_, np.zeros(2))
    assert_array_equal(model.explained_variance_ratio_, np.zeros(2))
    assert_array_equal(model.inverse_transform(transformed), X)


def test_representable_variance_avoids_squaring_singular_values_first():
    X = np.tile([[-1e154], [1e154]], (50, 1))
    model = PCA(1).fit(X)
    assert np.isfinite(model.explained_variance_[0])
    assert_allclose(model.explained_variance_, [1e308 * (100.0 / 99.0)], rtol=1e-14)
    assert_allclose(model.explained_variance_ratio_, [1.0])


def test_large_translation_preserves_variance_and_centered_projections():
    X = np.array([[0.0], [2.0], [0.0], [2.0]])
    shifted = X + 1e16
    original = PCA(1).fit(X)
    translated = PCA(1).fit(shifted)
    assert_allclose(translated.explained_variance_, original.explained_variance_)
    assert_allclose(translated.transform(shifted), original.transform(X))
    assert_allclose(translated.transform(shifted).mean(axis=0), 0.0, atol=1e-15)
    assert_array_equal(translated.inverse_transform(translated.transform(shifted)), shifted)


def test_transform_uses_training_mean(correlated_data):
    model = PCA(2).fit(correlated_data)
    sample = correlated_data[:1] + 100.0
    expected = (sample - model.mean_) @ model.components_.T
    assert_allclose(model.transform(sample), expected)
    assert not np.allclose(model.transform(sample), 0.0)


def test_input_is_not_mutated_and_signs_are_repeatable(correlated_data):
    X = correlated_data.copy()
    first = PCA().fit(X)
    second = PCA().fit(X)
    assert_array_equal(X, correlated_data)
    assert_array_equal(first.components_, second.components_)
    loadings = first.components_[np.arange(3), np.argmax(np.abs(first.components_), axis=1)]
    assert np.all(loadings >= 0)


@pytest.mark.parametrize("components", [0, -1, 1.5, True, "two"])
def test_invalid_component_parameter(components):
    with pytest.raises(ValueError, match="n_components"):
        PCA(components)


@pytest.mark.parametrize("shape", [(3, 10), (10, 3)])
def test_component_count_cannot_exceed_thin_svd_dimension(shape):
    with pytest.raises(ValueError, match="n_components"):
        PCA(4).fit(np.ones(shape))


@pytest.mark.parametrize(
    "X", [[[1.0, 2.0]], [1.0, 2.0], [[np.nan], [1.0]], [[np.inf], [1.0]], [["a"], ["b"]]]
)
def test_invalid_training_data(X):
    with pytest.raises(ValueError):
        PCA().fit(X)


@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_transformations_require_fitting(method):
    with pytest.raises(NotFittedError, match="not fitted"):
        getattr(PCA(1), method)([[0.0]])


def test_transform_feature_validation(correlated_data):
    model = PCA(2).fit(correlated_data)
    with pytest.raises(ValueError, match="features"):
        model.transform([[0.0, 0.0]])
    with pytest.raises(ValueError, match="features"):
        model.inverse_transform([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="finite"):
        model.inverse_transform([[0.0, np.inf]])


def test_unrepresentable_variance_is_rejected():
    with pytest.raises(ValueError, match="rescale"):
        PCA().fit([[-1e200], [1e200]])


def test_overflowing_transform_is_rejected():
    model = PCA(1).fit([[1.7e308], [1.7e308]])
    with pytest.raises(ValueError, match="rescale"):
        model.transform([[-1.7e308]])


def test_overflowing_reconstruction_is_rejected():
    model = PCA(1).fit([[1.7e308], [1.7e308]])
    with pytest.raises(ValueError, match="rescale"):
        model.inverse_transform([[1e308]])
