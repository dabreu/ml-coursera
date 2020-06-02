import numpy as np

from preprocessing import add_intercept_term


class LinearRegression:
    """
    Class for linear regression model
    """

    def __init__(self):
        self.theta = None

    def fit(self, x, y, alpha, iterations, add_intercept=False):
        """
        Fits the training samples (x) to the targets (y) using linear regression training algorithm

        :param x: the matrix(m,n) with the training samples, where m = #samples and n = #features
        :param y: the vector(m) with the target values
        :param alpha: the learning rate used by the algorithm
        :param iterations: the number of iterations to perform
        :return: returns the history of the cost function calculated for each iteration.
        The theta coefficients are stored as part of the model's state
        """
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
        Computes the cost function as the mean squared error between the predicted/hypothesis
        and target values

        :param x: matrix(m,n) with the values of features x, where m = #training samples and
                n = #features
        :param y: vector(m) with the target values
        :param theta: vector(n) with theta/coefficients values
        :return: cost for linear regression of using theta to fit data points in x and y
        """
        m = np.size(x, 0)
        error = self._hypothesis(x, theta) - y  # error between the prediction and the target
        cost = (1 / (2 * m)) * np.sum(np.square(error))  # mean squared error
        return cost

    @classmethod
    def _hypothesis(cls, x, theta):
        # Computes the hypothesis function for the given sample and theta
        return x.dot(theta)

    def predict(self, x, add_intercept=True):
        """
        Function that predicts the target based on the sample and the model's theta vector

        :param x: the sample value
        :param add_intercept: indicates whether it is needed to add the intercept term to features x
        :return: the predicted/target value
        """
        if add_intercept:
            x = add_intercept_term(x)

        return self._hypothesis(x, self.theta)
