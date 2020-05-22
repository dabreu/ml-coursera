import pytest
import numpy as np
from machine_learning.models import NeuralNetwork
from machine_learning.models import sigmoid

def test_network_initialization():
    network = NeuralNetwork(layers=[400, 25, 10])
    assert network.num_layers == 3
    assert network.layers[0] == 400
    assert network.layers[1] == 25
    assert network.layers[2] == 10
    assert network.weights != []

def test_network_random_initialization_of_weights_by_layer():
    network = NeuralNetwork(layers=[400, 25, 10])
    weights_layer1 = network.weights[0]
    weights_layer2 = network.weights[1]
    init_epsilon1 = np.sqrt(6) / np.sqrt(425)
    init_epsilon2 = np.sqrt(6) / np.sqrt(35)
    assert np.shape(weights_layer1) == (25, 401)
    assert np.shape(weights_layer2) == (10, 26)
    assert are_values_in_range(weights_layer1, -init_epsilon1, init_epsilon1)
    assert are_values_in_range(weights_layer2, -init_epsilon2, init_epsilon2)

def test_network_feed_forward_computes_units_by_layer():
    network = NeuralNetwork(layers=[2, 3, 2])
    network.weights = [np.ones((3, 3)), np.ones((2, 4))]
    z_, a_ = network.feed_forward([4, 6])
    assert len(z_) == len(a_)
    assert len(z_) == network.num_layers
    assert len(a_[0]) == 2
    assert len(z_[1]) == len(a_[1])
    assert len(z_[1]) == 3
    assert len(z_[2]) == len(a_[2])
    assert len(z_[2]) == 2

def test_network_feed_forward_computes_hidden_units():
    network = NeuralNetwork(layers=[2, 3, 2])
    network.weights = [np.array([[1, 1, 2], [1, 2, 1], [1, 1, 1]]), np.ones((2, 4))]
    z_, a_ = network.feed_forward(np.array([2, 3]))
    np.testing.assert_array_equal(a_[0], np.array([2, 3]))
    np.testing.assert_array_equal(z_[1], np.array([9, 8, 6]))
    np.testing.assert_array_equal(a_[1], np.array([sigmoid(9), sigmoid(8), sigmoid(6)]))
    value = sigmoid(9) + sigmoid(8) + sigmoid(6) + 1 # weights of theta2 are 1's
    np.testing.assert_array_equal(z_[2], np.array([value, value]))

def test_network_backpropagation():
    network = NeuralNetwork(layers=[2, 3, 1])
    network.weights = [np.array([[1, 1, 2], [1, 2, 1], [1, 1, 1]]), np.ones((1, 4))]
    x = np.array([[2, 3], [1, 1]])
    y = np.array([2, 1]).reshape(2, 1)
    gradients = network.backpropagation(x, y)
    assert len(gradients) == network.num_layers - 1
    assert np.shape(gradients[0]) == (3, 3)
    assert np.shape(gradients[1]) == (1, 4)

def test_network_fit():
    network = NeuralNetwork(layers=[2, 3, 1])
    x = np.array([[2, 3], [1, 1]])
    y = np.array([2, 1]).reshape(2, 1)
    cost_history = network.fit(x, y, alpha=0.01, iterations=1000)
    print(cost_history)

def are_values_in_range(values, init, end):
    return (values > init).all() and (values < end).all()