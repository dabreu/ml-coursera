import numpy as np
from machine_learning.preprocessing import add_intercept_term


class LogisticRegression:
    """
    Class for logistic regression model
    """
    def __init__(self):
        self.theta = None

    def fit(self, x, y, alpha, iterations, add_intercept=False):
        if add_intercept:
            x = add_intercept_term(x)

        m, n = x.shape
        self.theta = np.zeros(n)
        cost_history = np.zeros(iterations)
        for i in range(iterations):
            self.theta = self.theta - alpha * (1 / m) * (x.transpose().dot(self._hypothesis(x, self.theta) - y))
            cost_history[i] = self.cost(x, y, self.theta)
        return cost_history

    def cost(self, x, y, theta):
        """
        Computes the cost function based on cross-entropy loss or log loss

        :param x: matrix(m,n) with the values of features x, where m = #training samples and n = #features
        :param y: vector(m) with the target values
        :param theta: vector(n) with theta/coefficients values (decision boundary)
        :return: cost for logistic regression
        """
        m = np.size(x, 0)
        hypothesis = self._hypothesis(x, theta)
        error = (-y * np.log(hypothesis)) - ((1 - y) * np.log(1 - hypothesis))
        cost = (1 / m) * np.sum(error)
        return cost

    def gradient(self, x, y, theta):
        # Computes the gradient of the cost function at theta
        m = x.shape[0]
        return (1 / m) * (x.transpose().dot(self._hypothesis(x, theta) - y))

    @classmethod
    def _hypothesis(cls, x, theta):
        # Computes the hypothesis function for the given sample and theta
        return cls._sigmoid(x.dot(theta))

    @classmethod
    def _sigmoid(cls, z):
        return 1 / (1 + np.exp(-z))

    def predict_probability(self, x, add_intercept=True):
        """
        Function that predicts the target based on the sample and the model's theta vector

        :param x: the sample value
        :param add_intercept: indicates whether it is needed to add the intercept term to features x
        :return: the predicted/target value
        """
        if add_intercept:
            x = np.append([1], x)

        return self._hypothesis(x, self.theta)

    def predict(self, x, add_intercept=True):
        # Predicts the probability of target being 1 and then classify the value
        probability = self.predict_probability(x, add_intercept)
        return 1 if probability >= 0.5 else 0

    def accuracy(self, prediction, y):
        return np.mean(prediction == y) * 100