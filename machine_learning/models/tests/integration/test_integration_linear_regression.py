import pytest
import numpy as np
import pandas as pd
import datatest as dt
from machine_learning.models import LinearRegression
from machine_learning.preprocessing import add_intercept_term
from machine_learning.preprocessing import StandardScaler


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def samples1():
    df = pd.read_csv('./fixtures/ex1data1.txt', header=None)
    return df.iloc[:, 0], df.iloc[:, 1]


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def samples2():
    df = pd.read_csv('./fixtures/ex1data2.txt', header=None)
    return df.iloc[:, 0:2].to_numpy(), df.iloc[:, 2]


@pytest.fixture()
def linear_regression_model():
    return LinearRegression()


@pytest.mark.integration
def test_linear_regression_cost_theta_zero(samples1, linear_regression_model):
    x, y = samples1
    x = add_intercept_term(x)
    assert np.round(linear_regression_model.cost(x, y, [0, 0]), 2) == 32.07


@pytest.mark.integration
def test_linear_regression_cost_theta_non_zero(samples1, linear_regression_model):
    x, y = samples1
    x = add_intercept_term(x)
    assert np.round(linear_regression_model.cost(x, y, [-1, 2]), 2) == 54.24


@pytest.mark.integration
def test_linear_regression_fit_and_predict_sample1(samples1, linear_regression_model):
    x, y = samples1
    linear_regression_model.fit(x, y, alpha=0.01, iterations=1500, add_intercept=True)
    theta = linear_regression_model.theta
    np.testing.assert_array_almost_equal(theta, [-3.6303, 1.1664], decimal=4)
    assert np.round(linear_regression_model.predict(np.array([3.5])), 6) == 0.451977
    assert np.round(linear_regression_model.predict(np.array([7])), 6) == 4.534245


@pytest.mark.integration
def test_linear_regression_fit_and_predict_sample2(samples2, linear_regression_model):
    x, y = samples2
    scaler = StandardScaler()
    x = scaler.transform(x)
    linear_regression_model.fit(x, y, alpha=0.1, iterations=400, add_intercept=True)
    value = scaler.transform(np.array([[1650, 3]]))
    assert np.round(linear_regression_model.predict(value, add_intercept=1), 3) == 293081.465
