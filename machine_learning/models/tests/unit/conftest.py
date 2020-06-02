import pytest
from models import LinearRegression
from models import LogisticRegression


@pytest.fixture()
def linear_regression_model():
    return LinearRegression()


@pytest.fixture()
def logistic_regression_model():
    return LogisticRegression()
