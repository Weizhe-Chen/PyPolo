import pytest
import numpy as np

from pypolo.utils.scalers import MinMaxScaler, StandardScaler


def test_standard_scaler_init():
    rng = np.random.RandomState(0)
    scale = 2.0
    mean = 1.0
    values = scale * rng.randn(1000, 2) + mean
    standard_scaler = StandardScaler(values)
    np.testing.assert_almost_equal(standard_scaler.mean, mean, decimal=1)
    np.testing.assert_almost_equal(standard_scaler.scale, scale, decimal=1)


def test_standard_scaler_init_with_exception():
    values = np.array([0, 1, 2], dtype=np.float64)
    with pytest.raises(ValueError):
        StandardScaler(values)
    values = np.ones((10, 1))
    with pytest.raises(ValueError):
        StandardScaler(values)


def test_standard_scaler_preprocess():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=1.0, scale=2.0, size=(100, 1))
    standard_scaler = StandardScaler(values)
    standardized = standard_scaler.preprocess(values)
    mean = standardized.mean(axis=0, keepdims=False)
    std = standardized.std(axis=0, keepdims=False)
    np.testing.assert_almost_equal(mean, 0.0)
    np.testing.assert_almost_equal(std, 1.0)


def test_standard_scaler_postprocess_mean():
    rng = np.random.RandomState(0)
    expected = rng.normal(loc=1.0, scale=2.0, size=(100, 1))
    standard_scaler = StandardScaler(expected)
    standardized = standard_scaler.preprocess(expected)
    actual = standard_scaler.postprocess_mean(standardized)
    np.testing.assert_allclose(actual, expected)


def test_standard_scaler_postprocess_std():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=1.0, scale=2.0, size=(100, 1))
    standard_scaler = StandardScaler(values)
    standardized = standard_scaler.preprocess(values)
    expected = standardized * standard_scaler.scale
    actual = standard_scaler.postprocess_std(standardized)
    np.testing.assert_allclose(actual, expected)


def test_min_max_scaler_init():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=1.0, scale=2.0, size=(100, 2))
    scaler = MinMaxScaler(values, expected_range=(-1.0, 1.0))
    maxs = values.max(axis=0, keepdims=True)
    mins = values.min(axis=0, keepdims=True)
    expected = maxs - mins
    np.testing.assert_almost_equal(scaler.data_ptp, expected)


def test_min_max_scaler_init_with_exception():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=1.0, scale=2.0, size=(100, 2))
    with pytest.raises(ValueError):
        MinMaxScaler(values, expected_range=(1.0, 1.0))
    ones = np.ones(shape=(100, 2))
    with pytest.raises(ValueError):
        MinMaxScaler(ones, expected_range=(-1.0, 1.0))


def test_min_max_scaler_preprocess_postprocess():
    rng = np.random.RandomState(0)
    values = rng.normal(loc=1.0, scale=2.0, size=(100, 2))
    scaler = MinMaxScaler(values, expected_range=(-1.0, 1.0))
    preprocessed = scaler.preprocess(values)
    expected = -1.0 * np.ones(shape=(1, 2))
    np.testing.assert_almost_equal(
        preprocessed.min(axis=0, keepdims=True),
        expected,
    )
    expected = 1.0 * np.ones(shape=(1, 2))
    np.testing.assert_almost_equal(
        preprocessed.max(axis=0, keepdims=True),
        expected,
    )
    actual = scaler.postprocess(preprocessed)
    np.testing.assert_allclose(actual, values)
