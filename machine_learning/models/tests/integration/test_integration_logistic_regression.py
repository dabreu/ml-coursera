import pytest
import numpy as np
import pandas as pd
import datatest as dt
from models import LogisticRegression
from models import accuracy
from preprocessing import add_intercept_term
from preprocessing import StandardScaler


@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def samples1():
    df = pd.read_csv('./fixtures/ex2data1.txt', header=None)
    return df.iloc[:, 0:2], df.iloc[:, 2]


@pytest.fixture()
def logistic_regression_model():
    return LogisticRegression()


@pytest.mark.integration
def test_logistic_regression_cost_theta_zero(samples1, logistic_regression_model):
    x, y = samples1
    x = add_intercept_term(x)
    assert np.round(logistic_regression_model.cost(x, y, [0, 0, 0]), 3) == 0.693


@pytest.mark.integration
def test_logistic_regression_cost_theta_non_zero(samples1, logistic_regression_model):
    x, y = samples1
    x = add_intercept_term(x)
    assert np.round(logistic_regression_model.cost(x, y, [-24, 0.2, 0.2]), 3) == 0.218


@pytest.mark.integration
def test_logistic_regression_predict(samples1, logistic_regression_model):
    x, y = samples1
    x = StandardScaler().transform(x)
    iterations = 100
    logistic_regression_model.fit(x, y, 1, iterations, add_intercept=True)
    predictions = [logistic_regression_model.predict(x.iloc[i, :]) for i in range(np.size(x, 0))]
    assert accuracy(predictions, y) > 80
