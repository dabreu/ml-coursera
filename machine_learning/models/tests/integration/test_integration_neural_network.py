import pytest
import numpy as np
import pandas as pd
import datatest as dt
import matplotlib.pyplot as plt
from machine_learning.models import NeuralNetwork
from machine_learning.models import accuracy

@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def train_data():
    df = pd.read_csv('./fixtures/mnist_train.csv', header=None)
    x = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0]
    y_onehot = pd.get_dummies(y).to_numpy()
    return x, y.to_numpy(), y_onehot

@pytest.fixture(scope='module')
@dt.working_directory(__file__)
def test_data():
    df = pd.read_csv('./fixtures/mnist_test.csv', header=None)
    x = df.iloc[:, 1:].to_numpy()
    y = df.iloc[:, 0].to_numpy()
    return x, y

def test_cost_function(train_data, test_data):
    x, y, y_onehot,  = train_data
    x_test, y_test = test_data
    network = NeuralNetwork(layers=[784, 25, 10])
    network.fit(x, y_onehot, alpha=0.1, iterations=30)
    prediction_test = network.predict(x_test)
    accuracy_test = accuracy(prediction_test, y_test)
    assert accuracy_test > 90

