import numpy as np
from preprocessing import add_intercept_term


def test_cost_function_with_zero_decision_boundary(logistic_regression_model):
    x = add_intercept_term(np.array([1, 1, 1]).reshape(3, 1))
    y = np.array([0, 0, 0])
    theta = np.array([0, 0])
    cost = (1 / 3) * (-np.log(1 / 2) * 3)
    assert logistic_regression_model.cost(x, y, theta) == cost


def test_cost_function_with_good_decision_boundary(logistic_regression_model):
    x = add_intercept_term(np.array([1, 2, 3, 7, 8, 9]).reshape(6, 1))
    y = np.array([0, 0, 0, 1, 1, 1])
    theta = np.array([-4, 1])
    assert logistic_regression_model.cost(x, y, theta) < 0.1


def test_fit_calculates_theta_vector(logistic_regression_model):
    x = np.array([1, 2, 3, 7, 8, 9])
    y = np.array([0, 0, 0, 1, 1, 1])
    logistic_regression_model.fit(x, y, alpha=0.01, iterations=10, add_intercept=True)
    assert logistic_regression_model.theta is not None
    assert np.size(logistic_regression_model.theta) == 2


def test_fit_cost_function_history(logistic_regression_model):
    x = np.array([1, 2, 3, 7, 8, 9])
    y = np.array([0, 0, 0, 1, 1, 1])
    J_history = logistic_regression_model.fit(x, y, alpha=0.01, iterations=10, add_intercept=True)
    assert J_history is not None
    assert np.size(J_history) == 10


def test_fit_should_decrease_cost_function_after_each_iteration(logistic_regression_model):
    x = np.array([1, 2, 3, 7, 8, 9])
    y = np.array([0, 0, 0, 1, 1, 1])
    J_history = logistic_regression_model.fit(x, y, alpha=0.01, iterations=10, add_intercept=True)
    assert all(J_history[i] > J_history[i + 1] for i in range(len(J_history) - 1))


def test_predict_probability(logistic_regression_model):
    x = np.array([1, 5, 8, 9, 10, 19])
    y = np.array([0, 0, 0, 1, 1, 1])
    logistic_regression_model.fit(x, y, alpha=0.1, iterations=30, add_intercept=True)
    assert logistic_regression_model.predict_probability(np.array([3])) < 0.5
    assert logistic_regression_model.predict_probability(np.array([15])) >= 0.5


def test_predict_with_classification(logistic_regression_model):
    x = np.array([1, 5, 8, 9, 10, 19])
    y = np.array([0, 0, 0, 1, 1, 1])
    logistic_regression_model.fit(x, y, alpha=0.1, iterations=1000, add_intercept=True)
    assert logistic_regression_model.predict(np.array([3])) == 0
    assert logistic_regression_model.predict(np.array([15])) == 1
