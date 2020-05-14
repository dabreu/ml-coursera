import pytest
import numpy as np
from machine_learning.models import LinearRegression
from machine_learning.preprocessing import add_intercept_term

@pytest.fixture()
def linear_regression_model():
    return LinearRegression()


def test_cost_function_with_theta_zeros(linear_regression_model):
    x = add_intercept_term(np.array([1, 1, 1]).reshape(3, 1))
    y = np.array([1, 2, 3])
    theta = np.array([0, 0])
    assert np.round(linear_regression_model.cost(x, y, theta), decimals=2) ==  2.33


def test_cost_function_with_theta_ones(linear_regression_model):
    x = add_intercept_term(np.array([1, 1, 1]).reshape(3, 1))
    y = np.array([1, 2, 3])
    theta = np.array([1, 1])
    assert np.round(linear_regression_model.cost(x, y, theta), decimals=2) ==  0.33


def test_cost_function_returns_zero_when_x_and_y_fit(linear_regression_model):
    x = add_intercept_term(np.array([1, 2, 3]).reshape(3, 1))
    y = np.array([1, 2, 3])
    theta = np.array([0, 1])
    assert linear_regression_model.cost(x, y, theta) == 0


def test_cost_function_with_multiple_variables(linear_regression_model):
    x = add_intercept_term(np.array([[1, 2], [2, 4], [2, 3]]))
    y = np.array([4, 7, 7])
    theta = np.array([0, 1, 1])
    assert linear_regression_model.cost(x, y, theta) == 1


def test_cost_function_with_multiple_variables_returns_zero_when_x_and_y_fit(linear_regression_model):
    x = add_intercept_term(np.array([[1, 2], [2, 4], [2, 3]]))
    y = np.array([4, 7, 6])
    theta = np.array([1, 1, 1])
    assert linear_regression_model.cost(x, y, theta) == 0


def test_fit_calculates_theta_vector(linear_regression_model):
    x = add_intercept_term(np.array([1, 1, 1]).reshape(3, 1))
    y = np.array([1, 2, 3])
    J_history = linear_regression_model.fit(x, y, alpha=0.01, iterations=10)
    assert linear_regression_model.theta is not None
    assert np.size(linear_regression_model.theta) == 2


def test_fit_cost_function_history(linear_regression_model):
    x = add_intercept_term(np.array([1, 1, 1]).reshape(3, 1))
    y = np.array([1, 2, 3])
    J_history = linear_regression_model.fit(x, y, alpha=0.01, iterations=10)
    assert J_history is not None
    assert np.size(J_history) == 10


def test_fit_should_decrease_cost_function_after_each_iteration(linear_regression_model):
    x = add_intercept_term(np.array([1, 2, 3]).reshape(3, 1))
    y = np.array([4, 8, 12])
    J_history = linear_regression_model.fit(x, y, alpha=0.01, iterations=10)
    assert np.size(J_history) == 10
    assert all(J_history[i] > J_history[i + 1] for i in range(len(J_history) - 1))


def test_predict_value(linear_regression_model):
    x = add_intercept_term(np.array([1, 2, 3]).reshape(3, 1))
    y = np.array([4, 8, 12])
    linear_regression_model.fit(x, y, alpha=0.01, iterations=10)
    assert linear_regression_model.predict(np.array([4])) == np.array([1, 4]).dot(linear_regression_model.theta)

