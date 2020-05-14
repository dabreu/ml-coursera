import pytest
import numpy as np
from machine_learning.preprocessing import StandardScaler
from machine_learning.preprocessing import MinMaxScaler


@pytest.fixture()
def std_scaler():
    return StandardScaler()


@pytest.fixture()
def minmax_scaler():
    return MinMaxScaler()


def test_normalize_features(minmax_scaler):
    x = np.array([1, 2, 3]).reshape(3, 1)
    xn = minmax_scaler.transform(x)
    smin = minmax_scaler.min
    smax = minmax_scaler.max
    np.testing.assert_array_equal(xn, (x-smin)/(smax-smin))
    np.testing.assert_array_equal(smin, [1])
    np.testing.assert_array_equal(smax, [3])


def test_normalize_multiple_features(minmax_scaler):
    x = np.array([[1, 5, 3], [4, 2, 7]])
    xn = minmax_scaler.transform(x)
    smin = minmax_scaler.min
    smax = minmax_scaler.max
    np.testing.assert_array_equal(xn, (x-smin)/(smax-smin))
    np.testing.assert_array_equal(smin, [1, 2, 3])
    np.testing.assert_array_equal(smax, [4, 5, 7])


def test_normalize_single_value(minmax_scaler):
    x = np.array([[1, 5, 3], [4, 2, 7]])
    xn = minmax_scaler.transform(x)
    np.testing.assert_array_equal(minmax_scaler.transform([10, 14, 23]), [3, 4, 5])


def test_standardize_features(std_scaler):
    x = np.array([1, 2, 3]).reshape(3,1)
    xn = std_scaler.transform(x)
    mu = std_scaler.mu
    sigma = std_scaler.sigma
    np.testing.assert_array_almost_equal(xn, np.array([-1/0.816, 0, 1/0.816]).reshape(3, 1), decimal=3)
    np.testing.assert_array_equal(mu, [2])
    np.testing.assert_array_almost_equal(sigma, [0.816], decimal=3)


def test_standardize_multiple_features(std_scaler):
    x = np.array([[1, 2, 3], [10, 4, 6]])
    xn = std_scaler.transform(x)
    mu = std_scaler.mu
    sigma = std_scaler.sigma
    np.testing.assert_array_almost_equal(xn, np.array([[-1, -1, -1], [1, 1, 1]]))
    np.testing.assert_array_almost_equal(mu, [5.5, 3, 4.5], decimal=1)
    np.testing.assert_array_almost_equal(sigma, [4.5, 1, 1.5], decimal=1)


def test_standardize_single_value(std_scaler):
    x = np.array([[1, 2, 3], [10, 4, 6]])
    xn = std_scaler.transform(x)
    mu = std_scaler.mu
    sigma = std_scaler.sigma
    np.testing.assert_array_equal(std_scaler.transform([10, 5, 6]), [1, 2, 1])