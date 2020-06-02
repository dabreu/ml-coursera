from abc import ABC, abstractmethod

import numpy as np


class Scaler(ABC):
    """
    Abstract class for data scaling
    """

    @abstractmethod
    def transform(self, x):
        """
        Transforms X by applying scaling

        :param x: the values to scale
        :return: the input scaled
        """
        pass


class StandardScaler(Scaler):
    """
    Data scaling via standardization
    """

    def __init__(self):
        self.mu = None
        self.sigma = None

    def transform(self, x):
        """
        Function to standardize the features. It returns scaled X,  where the mean of each feature
        is 0 and the standard deviation is 1.
        The calculated mean and standard deviation by feature are stored as part of the
        normalizer's state.

        :param x: matrix(m,n) with the values of features x to normalize, where
                m = #training samples and n = #features
        :return: the normalized x
        """
        if self.mu is None:
            self.mu = self._calculate_mean(x)

        if self.sigma is None:
            self.sigma = self._calculate_std(x)

        return (x - self.mu) / self.sigma

    @classmethod
    def _calculate_mean(cls, x):
        return np.mean(x, axis=0)

    @classmethod
    def _calculate_std(cls, x):
        return np.std(x, axis=0)


class MinMaxScaler(Scaler):
    """
    Data scaling via min/max normalization
    """

    def __init__(self):
        self.min = None
        self.max = None

    def transform(self, x):
        """
        Function to rescale features. It returns a normalized X using min-max normalization.
        The calculated min and max values by feature are stored as part of the normalizer'state.

        :param x: matrix(m,n) with the values of features x to normalize,
                where m = #training samples and n = #features
        :return: the normalized x
        """
        if self.min is None:
            self.min = self._calculate_min(x)

        if self.max is None:
            self.max = self._calculate_max(x)

        return (x - self.min) / (self.max - self.min)

    @classmethod
    def _calculate_min(cls, x):
        return np.min(x, axis=0)

    @classmethod
    def _calculate_max(cls, x):
        return np.max(x, axis=0)


def add_intercept_term(x):
    return np.c_[np.ones((np.shape(x)[0], 1)), x]
