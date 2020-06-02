import numpy as np

from .commons import sigmoid, sigmoid_gradient


class NeuralNetwork:
    """
    Class that models a neural network with multiple layers
    """

    BIAS_TERM = 1

    def __init__(self, layers):
        """
        Creates an empty neural network with the given layers, having the #units per layer as
        indicated on the parameter
        :param layers: vector indicating the number of activation units per layer
        (i.e. layers[0] = 5000 indicates 5000 activation units on layer 1)
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.weights = self._get_initialized_weights()

    def fit(self, x, y, alpha, iterations, vlambda=0, batch_size=1000):
        """ Trains the neural network with the given x and y samples"""
        for iteration in range(iterations):
            batches = self._create_batches(x, y, batch_size)
            for xi, yi in batches:
                gradients = self.backpropagation(xi, yi, vlambda)
                for i, gradient in enumerate(gradients):
                    self.weights[i] = self.weights[i] - alpha * gradient

    def feed_forward(self, xi):
        """ Computes the forward propagation for sample xi """
        z_ = [0]
        a_ = [xi]
        a = xi
        for weight in self.weights:
            z = weight.dot(add_intercept_term_vector(a))
            a = sigmoid(z)
            z_.append(z)
            a_.append(a)
        return a_, z_

    def backpropagation(self, x, y, vlambda=0):
        """ Backpropagation to compute weights gradients per layer """
        gradient_ = self._initialize_gradients()  # returns a list with the gradient for each layer
        for xi, yi in zip(x, y):
            # Forward propagation to compute activation units on each layer for sample xi
            a_, z_ = self.feed_forward(xi)
            # Compute the delta/error for the last layer (a(L) - y)
            delta = a_[-1] - yi
            # For each layer computes the delta and the corresponding gradient
            for layer in range(self.num_layers - 2, -1, -1):
                # Calculate the gradient for the layer
                gradient_l = self._calculate_gradient(delta, a_, layer)
                gradient_[layer] = gradient_[layer] + gradient_l
                # Computes the delta/error for the layer
                delta = self._calculate_delta(delta, z_, layer)

        return [self._regularize(gradient, i, np.size(x, 0), vlambda)
                for i, gradient in enumerate(gradient_)]

    def cost(self, x, y):
        m = np.size(x, 0)
        cost = 0
        for xi, yi in zip(x, y):
            a = self._hypothesis(xi)
            error = (-yi * np.log(a)) - ((1 - yi) * np.log(1 - a))
            cost = cost + np.sum(error)
        return (1 / m) * cost

    def _create_batches(self, x, y, batch_size):
        return [(x[i:i + batch_size, :], y[i:i + batch_size, :])
                for i in range(0, len(x), batch_size)]

    def _initialize_gradients(self):
        """ Initializes the gradients with zero """
        return [np.zeros(np.shape(weight)) for weight in self.weights]

    def _calculate_delta(self, delta, z, layer):
        if layer == 0:
            return 0  # the first layer doesn't have delta/err as it corresponds to the X features
        # Calculate the delta/error as delta(l) = theta(l)^T * delta(l+1) .* sigmoid'(z(l))
        z_l = add_intercept_term_vector(z[layer])
        return (self.weights[layer].transpose().dot(delta)) * sigmoid_gradient(z_l)

    def _calculate_gradient(self, delta, a, layer):
        delta = delta.reshape(-1, 1)
        if layer != self.num_layers - 2:
            # delete delta_0, which is the bias term, for all deltas except the latest one
            delta = delta[1:, :]
        a_l = add_intercept_term_vector(a[layer]).reshape(-1, 1)
        # Calculate the gradient as G(l) = delta(l+1) * a(l)ˆT
        return delta.dot(a_l.transpose())

    def _regularize(self, gradient, layer, m, vlambda):
        weights = self.weights[layer]
        weights[:, 0] = 0  # set to zero the bias term of weights as it should not be regularized
        regularization = (vlambda / m) * weights
        return (1 / m) * gradient + regularization

    def _get_initialized_weights(self):
        """ Returns random weights within an epsilon interval, for symmetry breaking."""
        return [self._get_initialized_weight(i) for i in range(self.num_layers - 1)]

    def _get_initialized_weight(self, i):
        """ Creates a matrix of weights randomly initialized with values in interval
        (-epsilon, epsilon), where epsilon is calculated based on the number of inputs and outputs
        of the layer """
        inputs = self.layers[i]
        outputs = self.layers[i + 1]
        init_epsilon = np.sqrt(6) / np.sqrt(inputs + outputs)
        return np.random.rand(outputs,
                              inputs + self.BIAS_TERM) * init_epsilon * 2 - init_epsilon

    def _hypothesis(self, x):
        a_, _ = self.feed_forward(x)
        return a_[-1]

    def predict(self, x):
        return [np.argmax(self._hypothesis(xi)) for xi in x]


def add_intercept_term_vector(x):
    return np.append([1], x)
