import numpy as np

def sigmoid(z):
    # Sigmoid function
    return 1 / (1 + np.exp(-z))

def sigmoid_gradient(z):
    # Derivative of the sigmoid function
    return sigmoid(z) * (1 - sigmoid(z))

def accuracy(prediction, y):
    # Returns the accuracy of the prediction vs the output y
    return np.mean(prediction == y) * 100