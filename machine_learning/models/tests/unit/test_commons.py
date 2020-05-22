import pytest
import numpy as np
from machine_learning.models import sigmoid, sigmoid_gradient, accuracy


def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(1000) == 1
    assert sigmoid(-1000) == 0


def test_sigmoid_on_vector():
    value = np.array([-1000, 0, 1000])
    np.testing.assert_array_equal(sigmoid(value), np.array([0, 0.5, 1]))


def test_sigmoid_on_matrix():
    value = np.array([[-1000, 0, 1000], [0, -2000, 0]])
    np.testing.assert_array_equal(sigmoid(value), np.array([[0, 0.5, 1], [0.5, 0, 0.5]]))


def test_sigmoid_gradient():
    assert sigmoid_gradient(0) == 0.25
    assert sigmoid_gradient(1000) == 0
    assert sigmoid_gradient(-1000) == 0


def test_sigmoid_on_vector():
    value = np.array([-1, - 0.5, 0, 0.5, 1])
    np.testing.assert_array_almost_equal(sigmoid_gradient(value),
                                         np.array([0.196612, 0.235004, 0.250000, 0.235004, 0.196612]), 10 ** -7)


def test_sigmoid_on_matrix():
    value = np.array([[-1, - 0.5, 0, 0.5, 1], [0, 0.5, 1, -1, -0.5]])
    np.testing.assert_array_almost_equal(sigmoid_gradient(value),
                                         np.array([[0.196612, 0.235004, 0.250000, 0.235004, 0.196612],
                                                  [0.250000, 0.235004, 0.196612, 0.196612, 0.235004]]), 10 ** -7)


def test_accuracy():
    prediction = np.array([1, 2, 3, 8, 8, 8])
    y = np.array([1, 2, 3, 4, 5, 6])
    assert accuracy(prediction, y) == 50

def test_no_accuracy():
    prediction = np.array([8, 8, 8, 8, 8, 8])
    y = np.array([1, 2, 3, 4, 5, 6])
    assert accuracy(prediction, y) == 0

def test_full_accuracy():
    prediction = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([1, 2, 3, 4, 5, 6])
    assert accuracy(prediction, y) == 100